from __future__ import annotations

import math

import pytest
import torch

from lct_activation import LCTLayer


def _centered_indices(n: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    idx = torch.arange(n, device=device, dtype=dtype)
    return idx - (n - 1) / 2.0


def _fractional_fourier_kernel_reference(
    n: int,
    theta: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    dev = device or torch.device("cpu")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    idx = _centered_indices(n, dtype=real_dtype, device=dev)
    n_idx = idx.view(n, 1)
    k_idx = idx.view(1, n)

    a = math.cos(theta)
    b = math.sin(theta)
    d = a

    a_t = torch.tensor(a, dtype=dtype, device=dev)
    b_t = torch.tensor(b, dtype=dtype, device=dev)
    d_t = torch.tensor(d, dtype=dtype, device=dev)
    pi_t = torch.tensor(math.pi, dtype=dtype, device=dev)
    s = torch.tensor((n - 1) / 2.0, dtype=real_dtype, device=dev)

    phase = (
        1j * pi_t * (a_t / b_t) * n_idx.square()
        - 1j * 2.0 * pi_t * n_idx * k_idx / (b_t * n)
        + 1j * pi_t * (d_t / b_t) * k_idx.square()
    )
    lin_phase = (
        1j
        * 2.0
        * pi_t
        * s
        / b_t
        * ((a_t - 1.0 / n) * n_idx + (d_t - 1.0 / n) * k_idx)
    )
    const_phase = torch.exp(1j * pi_t * (s**2) * (a_t + d_t - 2.0 / n) / b_t)
    amp = torch.exp(-1j * pi_t / 4.0 * torch.sign(torch.real(b_t))) / math.sqrt(n)
    return (amp * const_phase * torch.exp(phase + lin_phase)).to(dtype)


def test_fourier_kernel_matches_unitary_dft_matrix() -> None:
    n = 16
    layer = LCTLayer(a=0.0, b=1.0, c=0.0, normalization="unitary")
    actual = layer.matrix(n).detach()
    expected = torch.fft.fft(torch.eye(n, dtype=torch.complex64), norm="ortho")
    assert torch.allclose(actual, expected, atol=4e-5, rtol=0.0)


def test_inverse_fourier_kernel_matches_unitary_idft_matrix() -> None:
    n = 16
    layer = LCTLayer(a=0.0, b=-1.0, c=0.0, normalization="unitary")
    actual = layer.matrix(n).detach()
    expected = torch.fft.ifft(torch.eye(n, dtype=torch.complex64), norm="ortho")
    assert torch.allclose(actual, expected, atol=4e-5, rtol=0.0)


def test_laplace_kernel_matches_minus_i_dft_matrix() -> None:
    n = 16
    layer = LCTLayer(a=0j, b=1j, c=1j, normalization="unitary")
    actual = layer.matrix(n).detach()
    expected = -1j * torch.fft.fft(torch.eye(n, dtype=torch.complex64), norm="ortho")
    assert torch.allclose(actual, expected, atol=3e-5, rtol=0.0)


@pytest.mark.parametrize("theta", [math.pi / 6, math.pi / 4, math.pi / 3])
def test_fractional_fourier_kernel_matches_reference(theta: float) -> None:
    n = 16
    layer = LCTLayer.fractional_fourier(
        theta,
        normalization="unitary",
        dense_threshold=256,
        unitary_projection=False,
    )
    actual = layer.matrix(n).detach()
    expected = _fractional_fourier_kernel_reference(n, theta)
    assert torch.allclose(actual, expected, atol=4e-5, rtol=0.0)


@pytest.mark.parametrize("theta", [math.pi / 6, math.pi / 4, math.pi / 3])
def test_fractional_fourier_kernel_is_unitary_in_unitary_mode(theta: float) -> None:
    n = 16
    layer = LCTLayer.fractional_fourier(theta, normalization="unitary", dense_threshold=256)
    kernel = layer.matrix(n).detach()
    ident = torch.eye(n, dtype=torch.complex64)
    assert torch.allclose(kernel.conj().T @ kernel, ident, atol=5e-5, rtol=0.0)


def test_fourier_kernel_matches_backward_norm_in_compositional_mode() -> None:
    n = 16
    layer = LCTLayer(a=0.0, b=1.0, c=0.0, normalization="compositional", normalized=False)
    actual = layer.matrix(n).detach()
    expected = torch.fft.fft(torch.eye(n, dtype=torch.complex64), norm="backward")
    assert torch.allclose(actual, expected, atol=1e-5, rtol=0.0)
