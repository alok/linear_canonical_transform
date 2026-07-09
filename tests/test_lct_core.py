from __future__ import annotations

import math

import torch

from lct_activation import (
    LCTLayer,
    SymplecticLCTLayer,
    linear_canonical_transform,
    symplectic_d,
)


def test_symplectic_d_is_stable_near_zero_a() -> None:
    a = torch.tensor(1e-10)
    b = torch.tensor(0.9)
    c = torch.tensor(-1.1)
    d = symplectic_d(a, b, c)
    assert torch.isfinite(torch.as_tensor(d)).all()


def test_lctlayer_default_is_identity_for_real_input() -> None:
    layer = LCTLayer()
    x = torch.randn(3, 32, dtype=torch.float32)
    y = layer(x)
    assert torch.allclose(y, x, atol=1e-4, rtol=0.0)


def test_lctlayer_fourier_matches_torch_fft() -> None:
    layer = LCTLayer(a=0.0, b=1.0, c=0.0, normalized=True)
    x = torch.randn(4, 64, dtype=torch.complex64)
    y = layer(x)
    ref = torch.fft.fft(x, dim=-1, norm="ortho")
    assert torch.allclose(y, ref, atol=1e-4, rtol=0.0)


def test_true_canonical_fourier_round_trips_through_inverse() -> None:
    layer = LCTLayer(a=0.0, b=1.0, c=-1.0, normalized=True)
    x = torch.randn(4, 64, dtype=torch.complex64)
    assert torch.allclose(layer.inverse(layer(x)), x, atol=2e-4, rtol=0.0)


def test_fast_path_matches_dense_reference() -> None:
    x = torch.randn(2, 384, dtype=torch.complex64)
    a, b, c = 0.73, 0.61, -0.42
    d = symplectic_d(a, b, c)

    dense = linear_canonical_transform(
        x,
        a=a,
        b=b,
        c=c,
        d=d,
        normalized=True,
        dense_threshold=4096,
        unitary_projection=False,
    )
    fast = linear_canonical_transform(
        x,
        a=a,
        b=b,
        c=c,
        d=d,
        normalized=True,
        dense_threshold=64,
        unitary_projection=False,
    )
    assert torch.allclose(fast, dense, atol=3e-5, rtol=0.0)


def test_symplectic_layer_promotes_historical_fourier_shorthand() -> None:
    layer = SymplecticLCTLayer.from_abc(a=0.0, b=1.0, c=0.0)
    a, b, c, d = layer.canonical_matrix
    actual = torch.stack((a, b, c, d))
    expected = torch.tensor((0.0, 1.0, -1.0, 0.0))
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)


def test_symplectic_layer_preserves_determinant_while_parameters_train() -> None:
    torch.manual_seed(11)
    layer = SymplecticLCTLayer(
        angle=math.pi / 4.0,
        log_scale=0.15,
        shear=-0.2,
        dense_threshold=64,
        unitary_projection=False,
    )
    x = torch.randn(3, 16, dtype=torch.complex64)
    target = torch.randn(3, 16, dtype=torch.complex64)
    loss = torch.view_as_real(layer(x) - target).square().mean()
    loss.backward()

    for parameter in (layer.angle, layer.log_scale, layer.shear):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)
        assert float(parameter.grad.abs()) > 1e-7

    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-3)
    optimizer.step()
    a, b, c, d = layer.canonical_matrix
    assert torch.allclose(a * d - b * c, torch.ones_like(a), atol=1e-6, rtol=0.0)
