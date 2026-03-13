from __future__ import annotations

import torch
from torch import nn

from lct_activation.integrations.nanogpt import NonlinearLCTActivation, make_lct_linear_factory, patch_feedforward_class
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
