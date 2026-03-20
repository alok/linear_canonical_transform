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

__all__ = ["HAS_TRITON", "complex_pointwise_mul"]


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
