from __future__ import annotations

import math
from typing import Final, Literal

import torch
from torch import Tensor

from .chirpz import chirpz_lct

PI: Final[float] = math.pi
NormMode = Literal["unitary", "compositional"]

__all__ = [
    "NormMode",
    "linear_canonical_transform",
    "symplectic_d",
]


def _dense_work_dtypes(device: torch.device) -> tuple[torch.dtype, torch.dtype]:
    if device.type == "mps":
        return torch.complex64, torch.float32
    return torch.complex128, torch.float64


def _complex_scalar(value: Tensor | float, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _mode(normalized: bool, normalization: NormMode | None) -> NormMode:
    if normalization is not None:
        return normalization
    return "unitary" if normalized else "compositional"


def _global_amplitude(
    *,
    b: Tensor,
    length: int,
    mode: NormMode,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if mode == "unitary":
        phase = torch.exp(
            -1j
            * torch.as_tensor(PI / 4.0, dtype=dtype, device=device)
            * torch.sign(torch.real(b))
        )
        return phase / math.sqrt(length)

    n = torch.as_tensor(float(length), dtype=dtype, device=device)
    return 1.0 / torch.sqrt(1j * b * n)


def _broadcast_vector(vector: Tensor, *, ndim: int, dim: int) -> Tensor:
    actual_dim = dim if dim >= 0 else ndim + dim
    shape = [1] * ndim
    shape[actual_dim] = vector.numel()
    return vector.reshape(shape)


def _apply_kernel(x: Tensor, kernel: Tensor, dim: int) -> Tensor:
    if dim != -1:
        x_perm = x.movedim(dim, -1)
        out = torch.matmul(x_perm, kernel)
        return out.movedim(-1, dim)
    return torch.matmul(x, kernel)


def _chirp(
    *,
    length: int,
    coeff: Tensor,
    dtype: torch.dtype,
    device: torch.device,
    centered: bool,
) -> Tensor:
    idx = torch.arange(length, device=device, dtype=torch.float32)
    if centered:
        idx = idx - (length - 1) / 2.0
    return torch.exp(1j * torch.as_tensor(PI, dtype=dtype, device=device) * coeff * idx**2)


def _resample_b0_branch(
    x: Tensor,
    *,
    c: Tensor,
    d: Tensor,
    dim: int,
    centered: bool,
    mode: NormMode,
) -> Tensor:
    if dim != -1:
        x = x.movedim(dim, -1)

    original_shape = x.shape
    n_features = x.size(-1)
    device = x.device
    dtype = x.dtype

    is_identity_like = (
        float(torch.abs(d - 1.0).detach().cpu()) <= 1e-6
        and float(torch.abs(c * d).detach().cpu()) <= 1e-6
    )
    if is_identity_like:
        return x.movedim(-1, dim) if dim != -1 else x

    idx = torch.arange(n_features, device=device, dtype=torch.float32)
    if centered:
        centered_idx = idx - (n_features - 1) / 2.0
        sample_points = torch.real(d) * centered_idx
        denom = (n_features - 1) / 2.0 if n_features > 1 else 1.0
        grid_x = sample_points / denom if denom != 0 else torch.zeros_like(sample_points)
    else:
        sample_points = torch.real(d) * idx
        if n_features > 1:
            grid_x = 2.0 * sample_points / (n_features - 1.0) - 1.0
        else:
            grid_x = torch.zeros_like(sample_points)

    x_reshaped = x.reshape(-1, 1, 1, n_features)
    grid = torch.zeros(x_reshaped.shape[0], 1, n_features, 2, device=device, dtype=torch.float32)
    grid[..., 0] = grid_x
    grid[..., 1] = 0.0

    resampled_real = torch.nn.functional.grid_sample(
        x_reshaped.real,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    resampled_imag = torch.nn.functional.grid_sample(
        x_reshaped.imag,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    resampled = torch.complex(resampled_real.squeeze(2), resampled_imag.squeeze(2)).reshape(original_shape)
    chirp = _chirp(length=n_features, coeff=c * d, dtype=dtype, device=device, centered=centered)

    amp = torch.as_tensor(1.0, dtype=dtype, device=device) if mode == "unitary" else torch.sqrt(d)
    out = amp * resampled * chirp
    return out.movedim(-1, dim) if dim != -1 else out


def linear_canonical_transform(
    x: Tensor,
    *,
    a: Tensor | float,
    b: Tensor | float,
    c: Tensor | float,
    d: Tensor | float,
    dim: int = -1,
    normalized: bool = True,
    normalization: NormMode | None = None,
    centered: bool = True,
    dense_threshold: int = 256,
    b_eps: float = 1e-6,
    unitary_projection: bool = True,
) -> Tensor:
    """Apply a finite-dimensional approximation of the 1-D discrete LCT."""

    x = x.to(torch.complex64)
    mode = _mode(normalized, normalization)

    n_features = x.size(dim)
    dtype = x.dtype
    device = x.device

    a_c = _complex_scalar(a, dtype=dtype, device=device)
    b_c = _complex_scalar(b, dtype=dtype, device=device)
    c_c = _complex_scalar(c, dtype=dtype, device=device)
    d_c = _complex_scalar(d, dtype=dtype, device=device)

    abs_b = float(torch.abs(b_c).detach().cpu())
    if abs_b <= b_eps:
        return _resample_b0_branch(
            x,
            c=c_c,
            d=d_c,
            dim=dim,
            centered=centered,
            mode=mode,
        )

    if (
        abs(abs_b - 1.0) <= 1e-6
        and float(torch.abs(a_c).detach().cpu()) <= 1e-6
        and float(torch.abs(c_c).detach().cpu()) <= 1e-6
        and float(torch.abs(torch.imag(b_c)).detach().cpu()) <= 1e-6
    ):
        b_sign = float(torch.sign(torch.real(b_c)).detach().cpu())
        if b_sign >= 0:
            norm = "ortho" if mode == "unitary" else "backward"
            return torch.fft.fft(x, dim=dim, norm=norm)
        norm = "ortho" if mode == "unitary" else "forward"
        return torch.fft.ifft(x, dim=dim, norm=norm)

    if (
        torch.isclose(a_c, torch.tensor(0j, dtype=dtype, device=device))
        and torch.isclose(b_c, torch.tensor(1j, dtype=dtype, device=device))
        and torch.isclose(c_c, torch.tensor(1j, dtype=dtype, device=device))
    ):
        idx = torch.arange(n_features, device=device)
        n_idx = idx.view(n_features, 1)
        k_idx = idx.view(1, n_features)
        expo = -1j * 2.0 * PI * n_idx * k_idx / n_features
        dft = torch.exp(expo.to(dtype)) / math.sqrt(n_features)
        kernel = torch.tensor(-1j, dtype=dtype, device=device) * dft
        return _apply_kernel(x, kernel, dim)

    if abs(abs_b - 1.0) <= 1e-6:
        chirp_in = _chirp(length=n_features, coeff=a_c / b_c, dtype=dtype, device=device, centered=centered)
        x_chirped = x * _broadcast_vector(chirp_in, ndim=x.ndim, dim=dim)
        x_fft = torch.fft.fft(x_chirped, dim=dim, norm="ortho")

        chirp_out = _chirp(length=n_features, coeff=d_c / b_c, dtype=dtype, device=device, centered=centered)
        out = x_fft * _broadcast_vector(chirp_out, ndim=x.ndim, dim=dim)
        amp = _global_amplitude(b=b_c, length=n_features, mode=mode, dtype=dtype, device=device)
        return out * (amp * math.sqrt(n_features))

    if n_features > dense_threshold:
        return chirpz_lct(
            x,
            a=a_c,
            b=b_c,
            c=c_c,
            d=d_c,
            dim=dim,
            centered=centered,
            normalization=mode,
            unitary_projection=unitary_projection,
            dense_fallback_threshold=dense_threshold,
        )

    dense_complex_dtype, dense_real_dtype = _dense_work_dtypes(device)

    idx = torch.arange(n_features, device=device, dtype=dense_real_dtype)
    if centered:
        idx = idx - (n_features - 1) / 2.0

    n_idx = idx.view(n_features, 1)
    k_idx = idx.view(1, n_features)

    a128 = torch.as_tensor(a_c, dtype=dense_complex_dtype, device=device)
    b128 = torch.as_tensor(b_c, dtype=dense_complex_dtype, device=device)
    d128 = torch.as_tensor(d_c, dtype=dense_complex_dtype, device=device)
    pi128 = torch.as_tensor(PI, dtype=dense_complex_dtype, device=device)

    phase = (
        1j * pi128 * (a128 / b128) * n_idx**2
        - 1j * 2.0 * pi128 * n_idx * k_idx / (b128 * n_features)
        + 1j * pi128 * (d128 / b128) * k_idx**2
    )

    if centered:
        s = torch.tensor((n_features - 1) / 2.0, dtype=dense_real_dtype, device=device)
        lin_phase = (
            1j
            * 2.0
            * pi128
            * s
            / b128
            * ((a128 - 1.0 / n_features) * n_idx + (d128 - 1.0 / n_features) * k_idx)
        )
        const_phase = torch.exp(
            1j * pi128 * (s**2) * (a128 + d128 - 2.0 / n_features) / b128
        )
        kernel = const_phase * torch.exp(phase + lin_phase)
    else:
        kernel = torch.exp(phase)

    amp = _global_amplitude(
        b=torch.as_tensor(b128, dtype=dense_complex_dtype, device=device),
        length=n_features,
        mode=mode,
        dtype=dense_complex_dtype,
        device=device,
    )
    kernel = (amp * kernel).to(dtype)

    if mode == "unitary" and unitary_projection:
        q, _ = torch.linalg.qr(kernel)
        kernel = q.to(dtype)

    return _apply_kernel(x, kernel, dim)


def symplectic_d(
    a: Tensor | float,
    b: Tensor | float,
    c: Tensor | float,
    *,
    eps: float = 1e-8,
) -> Tensor | float:
    """Solve ``ad - bc = 1`` for ``d`` with a stable fallback near ``a = 0``."""

    if not any(isinstance(v, torch.Tensor) for v in (a, b, c)):
        if isinstance(a, complex) or isinstance(b, complex) or isinstance(c, complex):
            a_c = complex(a)
            b_c = complex(b)
            c_c = complex(c)
            if abs(a_c) > eps:
                return (1.0 + b_c * c_c) / a_c
            denom = (a_c.conjugate() * a_c).real + eps * eps
            return ((1.0 + b_c * c_c) * a_c.conjugate()) / denom

        a_f = float(a)
        b_f = float(b)
        c_f = float(c)
        if abs(a_f) > eps:
            return (1.0 + b_f * c_f) / a_f
        return ((1.0 + b_f * c_f) * a_f) / (a_f * a_f + eps * eps)

    ref_tensor = next(v for v in (a, b, c) if isinstance(v, torch.Tensor))
    device = ref_tensor.device
    any_complex = any(
        isinstance(v, complex) or (isinstance(v, torch.Tensor) and torch.is_complex(v))
        for v in (a, b, c)
    )
    dtype: torch.dtype = torch.complex64 if any_complex else torch.float32

    a_t = torch.as_tensor(a, dtype=dtype, device=device)
    b_t = torch.as_tensor(b, dtype=dtype, device=device)
    c_t = torch.as_tensor(c, dtype=dtype, device=device)

    mask = torch.abs(a_t) > eps
    a_safe = torch.where(mask, a_t, torch.ones_like(a_t))
    exact = (1.0 + b_t * c_t) / a_safe

    if torch.is_complex(a_t):
        reg = (1.0 + b_t * c_t) * torch.conj(a_t) / (torch.abs(a_t) ** 2 + eps * eps)
    else:
        reg = (1.0 + b_t * c_t) * a_t / (a_t * a_t + eps * eps)
    return torch.where(mask, exact, reg)
