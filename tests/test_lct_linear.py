from __future__ import annotations

import torch
import torch.nn.functional as F

from lct_activation import LCTLinear


def test_lct_linear_preserves_expected_shape() -> None:
    layer = LCTLinear(15, 11)
    x = torch.randn(4, 7, 15, dtype=torch.float32)
    y = layer(x)
    assert y.shape == (4, 7, 11)
    assert y.dtype == x.dtype


def test_lct_linear_is_identity_like_at_init() -> None:
    layer = LCTLinear(16, 16, bias=False)
    x = torch.randn(3, 16, dtype=torch.float32)
    y = layer(x)
    assert torch.allclose(y, x, atol=1e-4, rtol=0.0)


def test_materialized_weight_matches_forward() -> None:
    layer = LCTLinear(12, 9)
    x = torch.randn(5, 12, dtype=torch.float32)
    weight = layer.materialize_weight()
    y_direct = layer(x)
    y_dense = F.linear(x, weight, layer.bias)
    assert torch.allclose(y_direct, y_dense, atol=1e-4, rtol=0.0)


def test_lct_linear_is_linear_without_bias() -> None:
    layer = LCTLinear(10, 10, bias=False)
    x = torch.randn(2, 10, dtype=torch.float32)
    y = torch.randn(2, 10, dtype=torch.float32)
    lhs = layer(x + y)
    rhs = layer(x) + layer(y)
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=0.0)


def test_to_linear_round_trips_dense_equivalent() -> None:
    layer = LCTLinear(14, 9)
    x = torch.randn(6, 14, dtype=torch.float32)
    dense = layer.to_linear()
    assert torch.allclose(layer(x), dense(x), atol=1e-4, rtol=0.0)


def test_fourier_fast_path_backward_matches_dense_equivalent() -> None:
    layer = LCTLinear(16, 16, bias=True, a=0.0, b=1.0, c=0.0)
    dense = layer.to_linear()

    x1 = torch.randn(4, 16, dtype=torch.float32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = layer(x1).square().sum()
    y2 = dense(x2).square().sum()

    y1.backward()
    y2.backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-4, rtol=0.0)
    assert torch.allclose(layer(x1.detach()), dense(x2.detach()), atol=1e-4, rtol=0.0)


def test_compositional_mode_materializes_consistently() -> None:
    layer = LCTLinear(12, 12, bias=False, a=0.0, b=1.0, c=0.0, normalization="compositional")
    x = torch.randn(3, 12, dtype=torch.float32)
    dense = layer.to_linear()
    assert torch.allclose(layer(x), dense(x), atol=1e-4, rtol=0.0)


def test_fourier_conv_backend_matches_fft_backend() -> None:
    fft_layer = LCTLinear(16, 16, bias=False, direct_fourier_backend="fft")
    conv_layer = LCTLinear(16, 16, bias=False, direct_fourier_backend="conv")
    conv_layer.load_state_dict(fft_layer.state_dict())

    x = torch.randn(4, 16, dtype=torch.float32)
    assert torch.allclose(conv_layer(x), fft_layer(x), atol=1e-4, rtol=0.0)


def test_fourier_conv_backend_backward_matches_fft_backend() -> None:
    fft_layer = LCTLinear(16, 16, bias=True, direct_fourier_backend="fft")
    conv_layer = LCTLinear(16, 16, bias=True, direct_fourier_backend="conv")
    conv_layer.load_state_dict(fft_layer.state_dict())

    x1 = torch.randn(3, 16, dtype=torch.float32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = fft_layer(x1).square().sum()
    y2 = conv_layer(x2).square().sum()
    y1.backward()
    y2.backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-4, rtol=0.0)
    assert torch.allclose(fft_layer.spectral_real.grad, conv_layer.spectral_real.grad, atol=1e-4, rtol=0.0)
    assert torch.allclose(fft_layer.spectral_imag.grad, conv_layer.spectral_imag.grad, atol=1e-4, rtol=0.0)
