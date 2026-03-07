from __future__ import annotations

import torch

from lct_activation import LCTActivation


def test_activation_preserves_shape_dtype_and_reality() -> None:
    torch.manual_seed(3)
    activation = LCTActivation(15)
    x = torch.randn(2, 4, 15, dtype=torch.float32)

    y = activation(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert not torch.is_complex(y)
    assert torch.isfinite(y).all()


def test_activation_is_genuinely_nonlinear() -> None:
    torch.manual_seed(4)
    activation = LCTActivation(16, bias_init=0.1, residual_mix=0.2)
    x = torch.randn(3, 16, dtype=torch.float32)
    y = torch.randn(3, 16, dtype=torch.float32)

    lhs = activation(x + y)
    rhs = activation(x) + activation(y)

    assert not torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


def test_activation_zero_stays_zero_with_default_bias() -> None:
    activation = LCTActivation(16)
    x = torch.zeros(3, 16, dtype=torch.float32)
    y = activation(x)
    assert torch.allclose(y, x, atol=1e-6, rtol=0.0)
