from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - local macOS env intentionally lacks Triton
    triton = None
    tl = None
    HAS_TRITON = False

__all__ = ["HAS_TRITON", "complex_pointwise_mul", "complex_mul_conj_diag_and_grad"]


if HAS_TRITON:

    @triton.jit
    def _complex_mul_kernel(
        x_ptr,
        d_ptr,
        out_ptr,
        total_elements,
        width,
        conjugate_d: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total_elements

        x_base = offs * 2
        col = offs % width
        d_base = col * 2

        xr = tl.load(x_ptr + x_base, mask=mask, other=0.0)
        xi = tl.load(x_ptr + x_base + 1, mask=mask, other=0.0)
        dr = tl.load(d_ptr + d_base, mask=mask, other=0.0)
        di = tl.load(d_ptr + d_base + 1, mask=mask, other=0.0)

        if conjugate_d:
            di = -di

        out_r = xr * dr - xi * di
        out_i = xr * di + xi * dr

        tl.store(out_ptr + x_base, out_r, mask=mask)
        tl.store(out_ptr + x_base + 1, out_i, mask=mask)


    @triton.jit
    def _complex_mul_conj_diag_grad_kernel(
        spectral_ptr,
        grad_ptr,
        diag_ptr,
        out_ptr,
        grad_real_ptr,
        grad_imag_ptr,
        rows,
        width,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < width)
        base = (row_offsets[:, None] * width + col_offsets[None, :]) * 2
        diag_base = col_offsets * 2

        sr = tl.load(spectral_ptr + base, mask=mask, other=0.0)
        si = tl.load(spectral_ptr + base + 1, mask=mask, other=0.0)
        gr = tl.load(grad_ptr + base, mask=mask, other=0.0)
        gi = tl.load(grad_ptr + base + 1, mask=mask, other=0.0)

        dr = tl.load(diag_ptr + diag_base, mask=col_offsets < width, other=0.0)[None, :]
        di = tl.load(diag_ptr + diag_base + 1, mask=col_offsets < width, other=0.0)[None, :]

        out_r = gr * dr + gi * di
        out_i = gi * dr - gr * di

        tl.store(out_ptr + base, out_r, mask=mask)
        tl.store(out_ptr + base + 1, out_i, mask=mask)

        partial_real = tl.sum(sr * gr + si * gi, axis=0)
        partial_imag = tl.sum(sr * gi - si * gr, axis=0)

        tl.atomic_add(grad_real_ptr + col_offsets, partial_real, mask=col_offsets < width)
        tl.atomic_add(grad_imag_ptr + col_offsets, partial_imag, mask=col_offsets < width)


def complex_pointwise_mul(
    x: Tensor,
    diag: Tensor,
    *,
    conjugate_diag: bool = False,
    use_triton: bool = False,
) -> Tensor:
    """Multiply a complex tensor by a complex diagonal vector.

    On CUDA with Triton installed, this uses a small custom kernel. Otherwise it
    falls back to standard PyTorch complex multiplication.
    """

    if not (use_triton and HAS_TRITON and x.is_cuda):
        diag_term = torch.conj(diag) if conjugate_diag else diag
        return x * diag_term.view(*([1] * (x.ndim - 1)), x.shape[-1])

    x_contig = x.contiguous()
    diag_contig = diag.contiguous()
    total_elements = x_contig.numel()
    width = x_contig.shape[-1]

    x_realimag = torch.view_as_real(x_contig).reshape(-1)
    diag_realimag = torch.view_as_real(diag_contig).reshape(-1)
    out_realimag = torch.empty_like(x_realimag)

    grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK"]),)
    _complex_mul_kernel[grid](
        x_realimag,
        diag_realimag,
        out_realimag,
        total_elements,
        width,
        conjugate_d=conjugate_diag,
        BLOCK=256,
    )

    out_view = out_realimag.reshape(*x_contig.shape, 2)
    return torch.view_as_complex(out_view)


def complex_mul_conj_diag_and_grad(
    spectral: Tensor,
    spectral_grad: Tensor,
    diag: Tensor,
    *,
    use_triton: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute ``spectral_grad * conj(diag)`` and the diagonal gradient.

    Returns ``(grad_spectral, grad_real, grad_imag)`` where the last two terms
    are the real and imaginary parts of ``sum(conj(spectral) * spectral_grad)``.
    """

    if not (use_triton and HAS_TRITON and spectral.is_cuda):
        grad_spectral = complex_pointwise_mul(
            spectral_grad,
            diag,
            conjugate_diag=True,
            use_triton=False,
        )
        spec_ri = torch.view_as_real(spectral)
        grad_ri = torch.view_as_real(spectral_grad)
        spec_r, spec_i = spec_ri[..., 0], spec_ri[..., 1]
        grad_r, grad_i = grad_ri[..., 0], grad_ri[..., 1]
        reduce_dims = tuple(range(spectral.ndim - 1))
        grad_real = torch.sum(spec_r * grad_r + spec_i * grad_i, dim=reduce_dims)
        grad_imag = torch.sum(spec_r * grad_i - spec_i * grad_r, dim=reduce_dims)
        return grad_spectral, grad_real, grad_imag

    spectral_contig = spectral.contiguous()
    grad_contig = spectral_grad.contiguous()
    diag_contig = diag.contiguous()

    rows = spectral_contig.numel() // spectral_contig.shape[-1]
    width = spectral_contig.shape[-1]

    spectral_ri = torch.view_as_real(spectral_contig).reshape(rows, width, 2).contiguous()
    grad_ri = torch.view_as_real(grad_contig).reshape(rows, width, 2).contiguous()
    diag_ri = torch.view_as_real(diag_contig).reshape(width, 2).contiguous()

    out_ri = torch.empty_like(grad_ri)
    grad_real = torch.zeros(width, dtype=spectral_contig.real.dtype, device=spectral_contig.device)
    grad_imag = torch.zeros(width, dtype=spectral_contig.real.dtype, device=spectral_contig.device)

    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]), triton.cdiv(width, meta["BLOCK_N"]))
    _complex_mul_conj_diag_grad_kernel[grid](
        spectral_ri.reshape(-1),
        grad_ri.reshape(-1),
        diag_ri.reshape(-1),
        out_ri.reshape(-1),
        grad_real,
        grad_imag,
        rows,
        width,
        BLOCK_M=8,
        BLOCK_N=64,
    )

    grad_spectral = torch.view_as_complex(out_ri.reshape(*spectral_contig.shape, 2))
    return grad_spectral, grad_real, grad_imag
