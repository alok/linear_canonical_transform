from __future__ import annotations

import torch
from torch import nn

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
