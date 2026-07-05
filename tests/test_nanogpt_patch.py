from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

_LOCAL_NANOGPT = Path("/Users/alokbeniwal/nanogpt/nanogpt/__init__.py")
requires_local_nanogpt = pytest.mark.skipif(
    not _LOCAL_NANOGPT.exists(),
    reason="local nanogpt checkout not available",
)

from lct_activation.integrations.nanogpt import (
    NonlinearLCTActivation,
    make_lct_activation_factory,
    make_lct_linear_factory,
    patch_feedforward_class,
)
from lct_activation.layers import LCTLinear


def _make_feedforward_class() -> type[nn.Module]:
    class DummyFeedForward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return DummyFeedForward


def test_patch_feedforward_replaces_activation() -> None:
    feedforward_cls = _make_feedforward_class()
    patch_feedforward_class(feedforward_cls, variant="activation")
    module = feedforward_cls()
    assert isinstance(module.net[1], NonlinearLCTActivation)


def test_patch_feedforward_replaces_first_linear() -> None:
    feedforward_cls = _make_feedforward_class()
    patch_feedforward_class(
        feedforward_cls,
        variant="linear",
        linear_factory=make_lct_linear_factory(),
    )
    module = feedforward_cls()
    assert isinstance(module.net[0], LCTLinear)
    assert isinstance(module.net[1], nn.ReLU)


def test_patch_feedforward_hybrid_replaces_both() -> None:
    feedforward_cls = _make_feedforward_class()
    patch_feedforward_class(
        feedforward_cls,
        variant="hybrid",
        linear_factory=make_lct_linear_factory(),
    )
    module = feedforward_cls()
    assert isinstance(module.net[0], LCTLinear)
    assert isinstance(module.net[1], NonlinearLCTActivation)


def test_activation_factory_accepts_normalization_kwargs() -> None:
    factory = make_lct_activation_factory(
        normalization="compositional", unitary_projection=False
    )
    activation = factory()
    y = activation(torch.randn(2, 16))
    assert y.shape == (2, 16)


def test_lazy_activation_params_materialize_and_train() -> None:
    """The lazy activation wrapper creates its trainable parameters on first
    forward; an optimizer built before that would silently train the model
    with a frozen activation (this happened - see the tune harness)."""

    feedforward_cls = _make_feedforward_class()
    patch_feedforward_class(
        feedforward_cls,
        variant="activation",
        activation_factory=make_lct_activation_factory(),
    )
    module = feedforward_cls()
    before = sum(p.numel() for p in module.parameters())
    module(torch.randn(4, 8))
    after = sum(p.numel() for p in module.parameters())
    assert after > before

    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)
    wrapper = next(m for m in module.modules() if isinstance(m, NonlinearLCTActivation))
    assert wrapper.activation is not None
    bias_before = wrapper.activation.bias.detach().clone()
    for _ in range(3):
        loss = module(torch.randn(4, 8)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert not torch.allclose(bias_before, wrapper.activation.bias)


@requires_local_nanogpt
def test_build_local_nanogpt_seed_controls_init() -> None:
    """The exec'd nanogpt source reseeds torch to 1337 at import time; the
    seed kwarg must reseed after that, or every 'seed' is the same run."""

    from lct_activation.integrations.nanogpt import build_local_nanogpt

    def fingerprint(seed: int) -> torch.Tensor:
        model, _ = build_local_nanogpt(
            variant="baseline",
            batch_size=2,
            ctx_len=16,
            n_heads=2,
            embed_dim=32,
            n_layers=1,
            vocab_size=65,
            seed=seed,
        )
        return next(model.parameters()).detach().clone()

    assert torch.equal(fingerprint(1), fingerprint(1))
    assert not torch.equal(fingerprint(1), fingerprint(2))


@requires_local_nanogpt
def test_paired_get_batch_is_identical_across_configs() -> None:
    from lct_activation.cli import _make_paired_get_batch
    from lct_activation.integrations.nanogpt import (
        build_local_nanogpt,
        make_lct_linear_factory,
    )

    batches = []
    for variant in ("baseline", "linear"):
        model, namespace = build_local_nanogpt(
            variant=variant,
            linear_factory=make_lct_linear_factory(),
            batch_size=4,
            ctx_len=16,
            n_heads=2,
            embed_dim=32,
            n_layers=1,
            vocab_size=65,
            seed=7,
        )
        # Consume some global RNG differently per variant, as training would.
        torch.randn(10 if variant == "baseline" else 33)
        get_batch = _make_paired_get_batch(namespace, seed=7, batch_size=4, ctx_len=16)
        xb, _ = get_batch("train")
        batches.append(xb)
    assert torch.equal(batches[0], batches[1])


def test_lowrank_factory_replaces_and_matches_lct_budget() -> None:
    from lct_activation.integrations.nanogpt import make_lowrank_linear_factory

    dense = nn.Linear(256, 1024)
    replacement = make_lowrank_linear_factory(rank=1)(dense)
    y = replacement(torch.randn(3, 256))
    assert y.shape == (3, 1024)

    lowrank_params = sum(p.numel() for p in replacement.parameters())
    lct = LCTLinear(256, 1024)
    lct_params = sum(p.numel() for p in lct.parameters())
    # rank-1 is the closest factorized budget to LCTLinear's parameter count
    assert abs(lowrank_params - lct_params) / lct_params < 0.25


@requires_local_nanogpt
def test_attention_scaling_changes_forward_only_when_enabled() -> None:
    from lct_activation.integrations.nanogpt import build_local_nanogpt

    outputs = []
    for scaling in (False, True, True):
        model, _ = build_local_nanogpt(
            variant="baseline",
            batch_size=2,
            ctx_len=16,
            n_heads=2,
            embed_dim=32,
            n_layers=1,
            vocab_size=65,
            seed=3,
            attention_scaling=scaling,
        )
        model.eval()
        with torch.no_grad():
            logits, _ = model(torch.zeros(1, 16, dtype=torch.long), torch.zeros(1, 16, dtype=torch.long))
        outputs.append(logits)
    assert not torch.allclose(outputs[0], outputs[1])   # scaling changes the function
    assert torch.allclose(outputs[1], outputs[2])       # deterministically


@requires_local_nanogpt
def test_attention_scaling_preserves_causality() -> None:
    from lct_activation.integrations.nanogpt import build_local_nanogpt

    model, _ = build_local_nanogpt(
        variant="baseline",
        batch_size=2,
        ctx_len=16,
        n_heads=2,
        embed_dim=32,
        n_layers=1,
        vocab_size=65,
        seed=3,
        attention_scaling=True,
    )
    model.eval()
    x = torch.randint(0, 65, (1, 16))
    with torch.no_grad():
        logits, _ = model(x, x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % 65
        logits2, _ = model(x2, x2)
    # changing the last token must not affect earlier positions
    assert torch.allclose(logits[0, :-1], logits2[0, :-1], atol=1e-5)
