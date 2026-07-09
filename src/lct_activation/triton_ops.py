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

__all__ = [
    "HAS_TRITON",
    "complex_mul_conj_diag_and_grad",
    "complex_pointwise_mul",
    "pack_real_pairs",
    "reduce_unpacked_grad",
    "unpack_real_pairs",
]


def _raw_triton_enabled(use_triton: bool, *tensors: Tensor) -> bool:
    """Use raw Triton kernels only when autograd does not need their outputs.

    These kernels write into freshly allocated tensors and intentionally do
    not register an autograd formula. ``_DirectFFTLinearFn`` calls them inside
    a custom ``torch.autograd.Function`` with an explicit backward, so grad
    mode is disabled there and the fast path remains available. Generic LCT
    paths must fall back to PyTorch operations whenever an input requires
    gradients; otherwise CUDA training silently severs the graph.
    """

    if not (use_triton and HAS_TRITON and tensors and tensors[0].is_cuda):
        return False
    return not (
        torch.is_grad_enabled()
        and any(tensor.requires_grad for tensor in tensors)
    )


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


    @triton.jit
    def _pack_real_pairs_kernel(
        x_ptr,
        out_ptr,
        rows,
        in_features,
        target_channels,
        repeat_mode: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        complex_cols = target_channels // 2
        mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < complex_cols)

        src_even = 2 * col_offsets
        src_odd = 2 * col_offsets + 1

        if repeat_mode:
            src_even = src_even % in_features
            src_odd = src_odd % in_features

        even_mask = mask & (src_even[None, :] < in_features)
        odd_mask = mask & (src_odd[None, :] < in_features)

        even_idx = row_offsets[:, None] * in_features + src_even[None, :]
        odd_idx = row_offsets[:, None] * in_features + src_odd[None, :]

        xr = tl.load(x_ptr + even_idx, mask=even_mask, other=0.0)
        xi = tl.load(x_ptr + odd_idx, mask=odd_mask, other=0.0)

        out_base = (row_offsets[:, None] * complex_cols + col_offsets[None, :]) * 2
        tl.store(out_ptr + out_base, xr, mask=mask)
        tl.store(out_ptr + out_base + 1, xi, mask=mask)


    @triton.jit
    def _unpack_real_pairs_kernel(
        z_ptr,
        out_ptr,
        rows,
        original_channels,
        complex_cols,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < original_channels)

        pair_idx = col_offsets // 2
        part_idx = col_offsets % 2
        src_base = (row_offsets[:, None] * complex_cols + pair_idx[None, :]) * 2 + part_idx[None, :]
        values = tl.load(z_ptr + src_base, mask=mask, other=0.0)
        out_idx = row_offsets[:, None] * original_channels + col_offsets[None, :]
        tl.store(out_ptr + out_idx, values, mask=mask)


    @triton.jit
    def _reduce_unpacked_grad_kernel(
        grad_ptr,
        out_ptr,
        rows,
        original_channels,
        expanded_channels,
        repeat_mode: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < original_channels)

        out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        if repeat_mode:
            repeats = (expanded_channels + original_channels - 1) // original_channels
            for r in range(0, 32):
                pass
        # fallback for non-repeat mode is a single contiguous slice
        if not repeat_mode:
            src_idx = row_offsets[:, None] * expanded_channels + col_offsets[None, :]
            out = tl.load(grad_ptr + src_idx, mask=mask, other=0.0)
        else:
            repeats = (expanded_channels + original_channels - 1) // original_channels
            for rep in range(0, 16):
                src_col = col_offsets + rep * original_channels
                rep_mask = mask & (src_col[None, :] < expanded_channels)
                src_idx = row_offsets[:, None] * expanded_channels + src_col[None, :]
                out += tl.load(grad_ptr + src_idx, mask=rep_mask, other=0.0)

        out_idx = row_offsets[:, None] * original_channels + col_offsets[None, :]
        tl.store(out_ptr + out_idx, out, mask=mask)


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

    if not _raw_triton_enabled(use_triton, x, diag):
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


def pack_real_pairs(
    x: Tensor,
    target_channels: int,
    *,
    mode: str = "zero",
    use_triton: bool = False,
) -> Tensor:
    if not _raw_triton_enabled(use_triton, x):
        current = x.size(-1)
        pad = target_channels - current
        if pad > 0:
            if mode == "tile" and current > 0:
                repeats = (target_channels + current - 1) // current
                x = x.repeat(*([1] * (x.ndim - 1)), repeats)[..., :target_channels]
            else:
                x = torch.nn.functional.pad(x, (0, pad))
        x_work = x.contiguous()
        return torch.view_as_complex(x_work.reshape(*x_work.shape[:-1], target_channels // 2, 2))

    x_contig = x.contiguous()
    rows = x_contig.numel() // x_contig.shape[-1]
    in_features = x_contig.shape[-1]
    complex_cols = target_channels // 2
    out_ri = torch.empty((rows, complex_cols, 2), device=x_contig.device, dtype=x_contig.dtype)

    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]), triton.cdiv(complex_cols, meta["BLOCK_N"]))
    _pack_real_pairs_kernel[grid](
        x_contig.reshape(-1),
        out_ri.reshape(-1),
        rows,
        in_features,
        target_channels,
        repeat_mode=(mode == "tile"),
        BLOCK_M=8,
        BLOCK_N=64,
    )
    return torch.view_as_complex(out_ri.reshape(*x_contig.shape[:-1], complex_cols, 2))


def unpack_real_pairs(
    z: Tensor,
    *,
    original_channels: int,
    use_triton: bool = False,
) -> Tensor:
    if not _raw_triton_enabled(use_triton, z):
        out = torch.view_as_real(z).reshape(*z.shape[:-1], -1)
        return out[..., :original_channels]

    z_contig = z.contiguous()
    rows = z_contig.numel() // z_contig.shape[-1]
    complex_cols = z_contig.shape[-1]
    z_ri = torch.view_as_real(z_contig).reshape(rows, complex_cols, 2).contiguous()
    out = torch.empty((rows, original_channels), device=z_contig.device, dtype=z_contig.real.dtype)

    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]), triton.cdiv(original_channels, meta["BLOCK_N"]))
    _unpack_real_pairs_kernel[grid](
        z_ri.reshape(-1),
        out.reshape(-1),
        rows,
        original_channels,
        complex_cols,
        BLOCK_M=8,
        BLOCK_N=64,
    )
    return out.reshape(*z_contig.shape[:-1], original_channels)


def reduce_unpacked_grad(
    grad: Tensor,
    *,
    original_channels: int,
    expanded_channels: int,
    mode: str = "zero",
    use_triton: bool = False,
) -> Tensor:
    if expanded_channels == original_channels:
        return grad

    if not (use_triton and HAS_TRITON and grad.is_cuda):
        if mode == "zero":
            return grad[..., :original_channels]

        # Adjoint of the tile expansion expanded[i] = x[i % original_channels]
        # used by pack_real_pairs (repeat, not repeat_interleave).
        indices = (
            torch.arange(expanded_channels, device=grad.device) % original_channels
        )
        out = torch.zeros(*grad.shape[:-1], original_channels, dtype=grad.dtype, device=grad.device)
        out.index_add_(-1, indices, grad)
        return out

    grad_contig = grad.contiguous()
    rows = grad_contig.numel() // grad_contig.shape[-1]
    out = torch.empty((rows, original_channels), device=grad_contig.device, dtype=grad_contig.dtype)

    grid = lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]), triton.cdiv(original_channels, meta["BLOCK_N"]))
    _reduce_unpacked_grad_kernel[grid](
        grad_contig.reshape(-1),
        out.reshape(-1),
        rows,
        original_channels,
        expanded_channels,
        repeat_mode=(mode == "tile"),
        BLOCK_M=8,
        BLOCK_N=64,
    )
    return out.reshape(*grad_contig.shape[:-1], original_channels)
