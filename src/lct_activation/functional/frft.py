from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "spectral_fractional_fourier_matrix",
    "spectral_fractional_fourier_transform",
]


def _dft_matrix(length: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    eye = torch.eye(length, dtype=dtype, device=device)
    return torch.fft.fft(eye, norm="ortho")


def spectral_fractional_fourier_matrix(
    length: int,
    angle: float,
    *,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a finite spectral FrFT matrix.

    This constructs ``F ** alpha`` from the DFT matrix's four spectral
    projectors, with ``alpha = 2 * angle / pi``. It is a finite-grid FrFT
    discretization whose main virtue is algebraic: it is unitary and composes
    on the finite grid up to floating-point error.
    """

    if length <= 0:
        raise ValueError("length must be positive")
    if not torch.is_complex(torch.empty((), dtype=dtype)):
        raise TypeError("dtype must be a complex dtype")

    dev = torch.device("cpu") if device is None else torch.device(device)
    alpha = 2.0 * float(angle) / math.pi
    fourier = _dft_matrix(length, dtype=dtype, device=dev)
    ident = torch.eye(length, dtype=dtype, device=dev)
    powers = [ident, fourier, fourier @ fourier, fourier @ fourier @ fourier]

    out = torch.zeros_like(fourier)
    for eigenvalue in (1.0 + 0.0j, -1.0 + 0.0j, 1.0j, -1.0j):
        projector = sum((eigenvalue ** (-power)) * powers[power] for power in range(4)) / 4.0
        out = out + (eigenvalue**alpha) * projector
    return out


def spectral_fractional_fourier_transform(
    x: Tensor,
    angle: float,
    *,
    dim: int = -1,
) -> Tensor:
    """Apply the finite spectral FrFT along ``dim``."""

    input_was_complex = torch.is_complex(x)
    x_complex = x.to(torch.complex64)
    matrix = spectral_fractional_fourier_matrix(
        x_complex.size(dim),
        angle,
        dtype=x_complex.dtype,
        device=x_complex.device,
    )
    if dim != -1:
        x_perm = x_complex.movedim(dim, -1)
        out = torch.matmul(x_perm, matrix).movedim(-1, dim)
    else:
        out = torch.matmul(x_complex, matrix)
    return out if input_was_complex else out.real.to(x.dtype)
