from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .lct import NormMode

__all__ = ["chirpz_lct"]


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
            * torch.as_tensor(math.pi / 4.0, dtype=dtype, device=device)
            * torch.sign(torch.real(b))
        )
        return phase / math.sqrt(length)

    n = torch.as_tensor(float(length), dtype=dtype, device=device)
    return 1.0 / torch.sqrt(1j * b * n)


def chirpz_lct(
    x: Tensor,
    *,
    a: Tensor | float,
    b: Tensor | float,
    c: Tensor | float,
    d: Tensor | float,
    dim: int = -1,
    centered: bool = True,
    normalization: NormMode = "unitary",
    unitary_projection: bool = True,
    dense_fallback_threshold: int = 256,
) -> Tensor:
    """Fast Chirp-Z implementation for generic ``|b| != 1`` LCTs."""

    del centered

    if not torch.is_complex(x):
        x = x.to(torch.complex64)

    n_features = x.size(dim)
    if n_features <= dense_fallback_threshold:
        from .lct import linear_canonical_transform

        return linear_canonical_transform(
            x,
            a=a,
            b=b,
            c=c,
            d=d,
            dim=dim,
            normalization=normalization,
            centered=True,
            dense_threshold=max(dense_fallback_threshold, n_features),
            unitary_projection=unitary_projection,
        )

    if dim != -1:
        x = x.movedim(dim, -1)

    device = x.device
    in_dtype = x.dtype
    work_dtype = torch.complex128

    a64 = torch.as_tensor(a, dtype=torch.complex64, device=device)
    b64 = torch.as_tensor(b, dtype=torch.complex64, device=device)
    d64 = torch.as_tensor(d, dtype=torch.complex64, device=device)

    a_w = torch.as_tensor(a64, dtype=work_dtype, device=device)
    b_w = torch.as_tensor(b64, dtype=work_dtype, device=device)
    d_w = torch.as_tensor(d64, dtype=work_dtype, device=device)

    x_w = x.to(work_dtype)
    n = torch.arange(n_features, device=device, dtype=torch.float64)
    pi_w = torch.as_tensor(math.pi, dtype=work_dtype, device=device)

    alpha = 1.0 / (b_w * n_features)
    chirp_in = torch.exp(1j * pi_w * ((a_w / b_w) - alpha) * n**2)
    u = x_w * chirp_in

    conv_len = 1 << (2 * n_features - 1).bit_length()
    u_pad = torch.zeros(*u.shape[:-1], conv_len, dtype=work_dtype, device=device)
    u_pad[..., :n_features] = u

    k = torch.arange(n_features, device=device, dtype=torch.float64)
    q_first = torch.exp(1j * pi_w * alpha * k**2)
    q = torch.zeros(conv_len, dtype=work_dtype, device=device)
    q[:n_features] = q_first
    if n_features > 1:
        q[conv_len - (n_features - 1) :] = q_first[1:].flip(0)

    conv = torch.fft.ifft(
        torch.fft.fft(u_pad, n=conv_len, dim=-1) * torch.fft.fft(q, n=conv_len),
        n=conv_len,
        dim=-1,
    )[..., :n_features]

    chirp_out = torch.exp(1j * pi_w * ((d_w / b_w) - alpha) * n**2)
    amp = _global_amplitude(
        b=b_w,
        length=n_features,
        mode=normalization,
        dtype=work_dtype,
        device=device,
    )

    y = (amp * conv * chirp_out).to(in_dtype)
    return y.movedim(-1, dim) if dim != -1 else y

