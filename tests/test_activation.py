from __future__ import annotations

import torch

from lct_activation import LCTActivation


def test_lct_activation_preserves_shape_and_dtype() -> None:
    act = LCTActivation(15)
    x = torch.randn(2, 4, 15, dtype=torch.float32)
    y = act(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_lct_activation_is_nonlinear() -> None:
    act = LCTActivation(16)
    x = torch.randn(2, 16)
    y = torch.randn(2, 16)
    lhs = act(x + y)
    rhs = act(x) + act(y)
    assert not torch.allclose(lhs, rhs, atol=1e-5, rtol=0.0)


def test_lct_activation_zero_stays_zero_with_default_bias() -> None:
    act = LCTActivation(16)
    x = torch.zeros(3, 16)
    y = act(x)
    assert torch.allclose(y, x, atol=1e-6, rtol=0.0)


def test_lct_activation_preserves_half_precision_dtype() -> None:
    act = LCTActivation(16)
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.randn(2, 16, dtype=dtype)
        assert act(x).dtype == dtype

