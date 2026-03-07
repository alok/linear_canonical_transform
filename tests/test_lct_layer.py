from __future__ import annotations

import math

import torch

from lct_activation import LCTLayer


def test_default_layer_is_identity_for_real_input() -> None:
    layer = LCTLayer()
    x = torch.randn(3, 32, dtype=torch.float32)
    y = layer(x)
    assert y.dtype == torch.float32
    assert torch.allclose(y, x, atol=1e-4, rtol=0.0)


def test_fourier_inverse_matches_ifft() -> None:
    layer = LCTLayer(a=0.0, b=1.0, c=0.0)
    x = torch.randn(3, 64, dtype=torch.complex64)
    y = layer(x)
    z = layer.inverse(y)
    expected = torch.fft.ifft(y, dim=-1, norm="ortho")
    assert torch.allclose(z, expected, atol=1e-4, rtol=0.0)


def test_matrix_materialization_matches_forward_on_eye() -> None:
    layer = LCTLayer(a=0.3, b=0.8, c=-0.2)
    eye = torch.eye(16, dtype=torch.complex64)
    actual = layer(eye)
    expected = layer.matrix(16)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)
