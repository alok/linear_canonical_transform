from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lct_activation import LCTLinear, SymplecticLCTLayer
from lct_activation.triton_ops import HAS_TRITON


CUDA_TRITON = torch.cuda.is_available() and HAS_TRITON


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


def test_learnable_lct_linear_uses_symplectic_transform_with_gradients() -> None:
    torch.manual_seed(12)
    layer = LCTLinear(
        16,
        16,
        bias=False,
        a=2**-0.5,
        b=2**-0.5,
        c=-(2**-0.5),
        inverse_after_multiply=False,
        learnable_transform=True,
    )
    assert isinstance(layer.transform, SymplecticLCTLayer)

    x = torch.randn(4, 16)
    target = torch.randn(4, 16)
    (layer(x) - target).square().mean().backward()

    for parameter in (
        layer.transform.angle,
        layer.transform.log_scale,
        layer.transform.shear,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)
        assert float(parameter.grad.abs()) > 1e-7

    a, b, c, d = layer.transform.canonical_matrix
    assert torch.allclose(a * d - b * c, torch.ones_like(a), atol=1e-6, rtol=0.0)


def test_learnable_fourier_linear_is_identity_like_with_inverse() -> None:
    layer = LCTLinear(16, 16, bias=False, learnable_transform=True)
    x = torch.randn(3, 16)
    assert torch.allclose(layer(x), x, atol=2e-4, rtol=0.0)


def test_fixed_and_learned_symplectic_controls_share_initial_function() -> None:
    fixed = LCTLinear(
        16,
        16,
        bias=False,
        learnable_transform=False,
        transform_parameterization="symplectic",
    )
    learned = LCTLinear(
        16,
        16,
        bias=False,
        learnable_transform=True,
        transform_parameterization="symplectic",
    )
    learned.load_state_dict(fixed.state_dict())
    x = torch.randn(3, 16)

    assert torch.allclose(fixed(x), learned(x), atol=1e-6, rtol=0.0)
    assert not any(parameter.requires_grad for parameter in fixed.transform.parameters())
    assert all(parameter.requires_grad for parameter in learned.transform.parameters())


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


@torch.no_grad()
def _break_identity_initialization(layer: LCTLinear) -> None:
    real = torch.linspace(
        0.75,
        1.25,
        layer.complex_features,
        device=layer.spectral_real.device,
        dtype=layer.spectral_real.dtype,
    )
    imag = torch.linspace(
        -0.2,
        0.2,
        layer.complex_features,
        device=layer.spectral_imag.device,
        dtype=layer.spectral_imag.dtype,
    )
    layer.spectral_real.copy_(real)
    layer.spectral_imag.copy_(imag)


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA with Triton")
def test_cuda_triton_learned_symplectic_path_preserves_autograd() -> None:
    torch.manual_seed(21)
    layer = LCTLinear(
        64,
        64,
        bias=False,
        learnable_transform=True,
        transform_parameterization="symplectic",
        use_triton_kernels=True,
    ).cuda()
    _break_identity_initialization(layer)
    x = torch.randn(8, 64, device="cuda", requires_grad=True)
    target = torch.randn_like(x)

    (layer(x) - target).square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in (
        layer.transform.angle,
        layer.transform.log_scale,
        layer.transform.shear,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)
    assert any(
        float(parameter.grad.abs()) > 1e-7
        for parameter in (
            layer.transform.angle,
            layer.transform.log_scale,
            layer.transform.shear,
        )
    )


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA with Triton")
def test_cuda_triton_generic_frft_path_preserves_input_and_spectral_gradients() -> None:
    root_half = 2**-0.5
    layer = LCTLinear(
        64,
        64,
        bias=False,
        a=root_half,
        b=root_half,
        c=-root_half,
        transform_parameterization="legacy",
        use_triton_kernels=True,
    ).cuda()
    _break_identity_initialization(layer)
    x = torch.randn(8, 64, device="cuda", requires_grad=True)

    layer(x).square().mean().backward()

    for gradient in (x.grad, layer.spectral_real.grad, layer.spectral_imag.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().max()) > 1e-7


@pytest.mark.skipif(not CUDA_TRITON, reason="requires CUDA with Triton")
def test_cuda_direct_fft_triton_backward_matches_native_torch() -> None:
    torch.manual_seed(22)
    triton_layer = LCTLinear(64, 64, bias=True, use_triton_kernels=True).cuda()
    native_layer = LCTLinear(64, 64, bias=True, use_triton_kernels=False).cuda()
    _break_identity_initialization(triton_layer)
    native_layer.load_state_dict(triton_layer.state_dict())
    x_triton = torch.randn(8, 64, device="cuda", requires_grad=True)
    x_native = x_triton.detach().clone().requires_grad_(True)

    output_triton = triton_layer(x_triton)
    output_native = native_layer(x_native)
    output_triton.square().mean().backward()
    output_native.square().mean().backward()

    torch.testing.assert_close(output_triton, output_native, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(x_triton.grad, x_native.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        triton_layer.spectral_real.grad,
        native_layer.spectral_real.grad,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        triton_layer.spectral_imag.grad,
        native_layer.spectral_imag.grad,
        atol=2e-5,
        rtol=2e-5,
    )
