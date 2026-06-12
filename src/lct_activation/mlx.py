"""MLX (Apple-silicon-native) port of the LCT activation and linear layers.

This module mirrors the PyTorch implementation in ``lct_activation.layers`` and
``lct_activation.functional.lct`` closely enough that the two backends agree
numerically on the same inputs (to roughly float32 ulp-level differences in
the FFTs; dense kernels and chirp tables are taken verbatim from the torch
reference at plan-compile time), with one deliberate difference:

- Transform parameters ``(a, b, c)`` are **fixed at construction time** in the
  MLX backend. MLX traces computation lazily, so the data-dependent branch
  selection the PyTorch code performs per forward (``b ~= 0`` vs FFT vs chirp
  vs dense) cannot depend on traced parameter values. Instead, each layer
  compiles a *plan* per feature length: chirps, Bluestein tables, and dense
  kernels are precomputed once (reusing the torch reference's own arithmetic
  where its float32 rounding is observable) and the runtime forward is pure
  MLX FFTs, elementwise complex multiplies, and matmuls.

The genuinely learnable parts remain learnable: the modReLU bias, output gain,
and residual mix of :class:`LCTModReLU`, and the spectral diagonal and bias of
:class:`LCTLinear`.

Requires the optional ``mlx`` dependency::

    uv sync --extra dev          # on Apple silicon, pulls mlx automatically
    uv add "lct-activation[mlx]" # in downstream projects
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
except ImportError as exc:  # pragma: no cover - exercised only without mlx
    raise ImportError(
        "lct_activation.mlx requires the optional 'mlx' dependency. "
        "Install it with `uv sync --extra mlx` (or `pip install mlx`)."
    ) from exc

PI = math.pi
NormMode = Literal["unitary", "compositional"]
Plan = Callable[[mx.array], mx.array]

__all__ = [
    "LCTActivation",
    "LCTLayer",
    "LCTLinear",
    "LCTModReLU",
    "NormMode",
    "linear_canonical_transform",
    "symplectic_d",
]


def symplectic_d(
    a: complex | float,
    b: complex | float,
    c: complex | float,
    *,
    eps: float = 1e-8,
) -> complex:
    """Solve ``ad - bc = 1`` for ``d`` with a stable fallback near ``a = 0``."""

    a_c = complex(a)
    b_c = complex(b)
    c_c = complex(c)
    if abs(a_c) > eps:
        return (1.0 + b_c * c_c) / a_c
    denom = (a_c.conjugate() * a_c).real + eps * eps
    return ((1.0 + b_c * c_c) * a_c.conjugate()) / denom


def _mode(normalized: bool, normalization: NormMode | None) -> NormMode:
    if normalization is not None:
        return normalization
    return "unitary" if normalized else "compositional"


def _global_amplitude(*, b: complex, length: int, mode: NormMode) -> complex:
    if mode == "unitary":
        sign = 0.0 if b.real == 0.0 else math.copysign(1.0, b.real)
        return complex(np.exp(-1j * (PI / 4.0) * sign)) / math.sqrt(length)
    return complex(1.0 / np.sqrt(1j * b * length))


def _torch_layer_d(
    a: complex | float, b: complex | float, c: complex | float
) -> complex:
    """Mirror the torch ``LCTLayer.d`` property bit-for-bit.

    The torch layer stores ``(a, b, c)`` as float32/complex64 parameters and
    solves for ``d`` in that precision. The MLX layer must reproduce the same
    rounded ``d``: the dense+QR plan downstream amplifies even 1-ulp parameter
    differences into O(1) kernel differences (the kernel is numerically
    rank-deficient), so a float64-accurate ``d`` would *break* parity.
    """

    import torch

    from .functional.lct import symplectic_d as torch_symplectic_d

    def as_param(value: complex | float) -> "torch.Tensor":
        dtype = torch.complex64 if isinstance(value, complex) else torch.float32
        return torch.tensor(value, dtype=dtype)

    d = torch_symplectic_d(as_param(a), as_param(b), as_param(c))
    return complex(torch.as_tensor(d, dtype=torch.complex64).item())


def _const(values: np.ndarray) -> mx.array:
    return mx.array(np.ascontiguousarray(values.astype(np.complex64)))


def _fft(x: mx.array, norm: str) -> mx.array:
    n = x.shape[-1]
    y = mx.fft.fft(x, axis=-1)
    if norm == "ortho":
        return y * (1.0 / math.sqrt(n))
    return y


def _ifft(x: mx.array, norm: str) -> mx.array:
    n = x.shape[-1]
    y = mx.fft.ifft(x, axis=-1)
    if norm == "ortho":
        return y * math.sqrt(n)
    if norm == "forward":
        return y * float(n)
    return y


def _torch_reference_kernel(
    *,
    length: int,
    a: complex,
    b: complex,
    c: complex,
    d: complex,
    mode: NormMode,
    centered: bool,
    unitary_projection: bool,
) -> np.ndarray:
    """Dense LCT kernel taken verbatim from the PyTorch reference.

    Built by applying the torch implementation to an identity matrix, so the
    MLX backend reuses the exact same bits, QR projection included. This is
    not pedantry: the centered dense kernel is numerically rank-deficient at
    moderate lengths (condition number ~1e9 at N=64), so the QR projection
    amplifies even 1-ulp build differences (np.exp vs torch.exp) into O(1)
    kernel differences. Sharing the construction is the only robust way to
    keep the backends in agreement on this branch. Precompute-only; torch is
    already a hard dependency of the package.
    """

    import torch

    from .functional.lct import linear_canonical_transform as torch_lct

    eye = torch.eye(length, dtype=torch.complex64)
    kernel = torch_lct(
        eye,
        a=a,
        b=b,
        c=c,
        d=d,
        dim=-1,
        normalization=mode,
        centered=centered,
        dense_threshold=length,
        unitary_projection=unitary_projection,
    )
    return kernel.numpy()


def _torch_chirp_table(*, length: int, coeff: complex, centered: bool) -> np.ndarray:
    """Chirp table built with the torch reference's own float32 arithmetic.

    The torch backend computes chirp phases in complex64; at Bluestein-scale
    phases the float32 rounding is the dominant numerical effect, so the MLX
    plan must bake in the *same* tables (not more accurate float64 ones) for
    the backends to agree at large N. Precompute-only.
    """

    import torch

    from .functional.lct import _chirp as torch_chirp

    return torch_chirp(
        length=length,
        coeff=torch.tensor(coeff, dtype=torch.complex64),
        dtype=torch.complex64,
        device=torch.device("cpu"),
        centered=centered,
    ).numpy()


def _resample_plan(
    *,
    length: int,
    c: complex,
    d: complex,
    centered: bool,
    mode: NormMode,
) -> Plan:
    """Mirror of the PyTorch ``b ~= 0`` grid-sample branch via gather + lerp."""

    if abs(d - 1.0) <= 1e-6 and abs(c * d) <= 1e-6:
        return lambda x: x

    idx = np.arange(length, dtype=np.float64)
    if centered:
        centered_idx = idx - (length - 1) / 2.0
        pos = d.real * centered_idx + (length - 1) / 2.0
    else:
        pos = d.real * idx

    lo = np.floor(pos)
    w_hi = pos - lo
    w_lo = 1.0 - w_hi
    i_lo = lo.astype(np.int64)
    i_hi = i_lo + 1

    # grid_sample zero padding: out-of-range taps contribute nothing.
    w_lo = np.where((i_lo >= 0) & (i_lo < length), w_lo, 0.0)
    w_hi = np.where((i_hi >= 0) & (i_hi < length), w_hi, 0.0)
    i_lo = np.clip(i_lo, 0, length - 1)
    i_hi = np.clip(i_hi, 0, length - 1)

    chirp = _torch_chirp_table(
        length=length, coeff=complex(np.complex64(c * d)), centered=centered
    )
    amp = 1.0 if mode == "unitary" else complex(np.sqrt(np.complex128(d)))
    scale = _const(amp * chirp)

    idx_lo = mx.array(i_lo.astype(np.int32))
    idx_hi = mx.array(i_hi.astype(np.int32))
    weight_lo = mx.array(w_lo.astype(np.float32))
    weight_hi = mx.array(w_hi.astype(np.float32))

    def plan(x: mx.array) -> mx.array:
        # Gather the float32 real/imag planes separately: the vjp of take is
        # a scatter, and MLX's Metal backend has no complex64 scatter, so a
        # complex gather would make this branch non-differentiable on GPU.
        x_re, x_im = mx.real(x), mx.imag(x)
        re = (
            mx.take(x_re, idx_lo, axis=-1) * weight_lo
            + mx.take(x_re, idx_hi, axis=-1) * weight_hi
        )
        im = (
            mx.take(x_im, idx_lo, axis=-1) * weight_lo
            + mx.take(x_im, idx_hi, axis=-1) * weight_hi
        )
        return (re + 1j * im).astype(mx.complex64) * scale

    return plan


def _chirpz_plan(
    *,
    length: int,
    a: complex,
    b: complex,
    d: complex,
    mode: NormMode,
) -> Plan:
    """Bluestein chirp-z plan for generic ``|b| != 1`` LCTs (matches torch)."""

    n = np.arange(length, dtype=np.float64)
    alpha = 1.0 / (b * length)
    chirp_in = np.exp(1j * PI * ((a / b) - alpha) * n**2)

    conv_len = 1 << (2 * length - 1).bit_length()
    q = np.zeros(conv_len, dtype=np.complex128)
    q_first = np.exp(1j * PI * alpha * n**2)
    q[:length] = q_first
    if length > 1:
        q[conv_len - (length - 1):] = q_first[1:][::-1]

    chirp_out = np.exp(1j * PI * ((d / b) - alpha) * n**2)
    amp = _global_amplitude(b=b, length=length, mode=mode)

    chirp_in_c = _const(chirp_in)
    q_fft_c = _const(np.fft.fft(q))
    out_scale_c = _const(amp * chirp_out)
    pad = conv_len - length

    def plan(x: mx.array) -> mx.array:
        u = x * chirp_in_c
        u_pad = mx.pad(u, [(0, 0)] * (u.ndim - 1) + [(0, pad)])
        conv = mx.fft.ifft(mx.fft.fft(u_pad, axis=-1) * q_fft_c, axis=-1)
        return conv[..., :length] * out_scale_c

    return plan


def _compile_plan(
    *,
    length: int,
    a: complex,
    b: complex,
    c: complex,
    d: complex,
    mode: NormMode,
    centered: bool,
    dense_threshold: int,
    b_eps: float,
    unitary_projection: bool,
) -> Plan:
    """Pick the same branch the PyTorch implementation would, precomputed.

    Parameters are rounded through complex64 first, exactly like the PyTorch
    path casts them to ``torch.complex64`` before use; for Bluestein-scale
    phases (``pi * n**2``) that rounding is the dominant numerical effect, so
    matching it is what makes the two backends agree.
    """

    a = complex(np.complex64(a))
    b = complex(np.complex64(b))
    c = complex(np.complex64(c))
    d = complex(np.complex64(d))

    abs_b = abs(b)
    if abs_b <= b_eps:
        return _resample_plan(length=length, c=c, d=d, centered=centered, mode=mode)

    if (
        abs(abs_b - 1.0) <= 1e-6
        and abs(a) <= 1e-6
        and abs(c) <= 1e-6
        and abs(b.imag) <= 1e-6
    ):
        if b.real >= 0:
            norm = "ortho" if mode == "unitary" else "backward"
            return lambda x: _fft(x, norm)
        norm = "ortho" if mode == "unitary" else "forward"
        return lambda x: _ifft(x, norm)

    if abs(a) <= 1e-8 and abs(b - 1j) <= 1e-5 and abs(c - 1j) <= 1e-5:
        kernel = _const(
            _torch_reference_kernel(
                length=length,
                a=a,
                b=b,
                c=c,
                d=d,
                mode=mode,
                centered=centered,
                unitary_projection=unitary_projection,
            )
        )
        return lambda x: mx.matmul(x, kernel)

    if abs(abs_b - 1.0) <= 1e-6:
        coeff_in = complex(np.complex64(a / b))
        coeff_out = complex(np.complex64(d / b))
        chirp_in = _const(
            _torch_chirp_table(length=length, coeff=coeff_in, centered=centered)
        )
        amp = _global_amplitude(b=b, length=length, mode=mode)
        chirp_out = _const(
            amp
            * math.sqrt(length)
            * _torch_chirp_table(length=length, coeff=coeff_out, centered=centered)
        )
        return lambda x: _fft(x * chirp_in, "ortho") * chirp_out

    if length > dense_threshold:
        return _chirpz_plan(length=length, a=a, b=b, d=d, mode=mode)

    kernel = _const(
        _torch_reference_kernel(
            length=length,
            a=a,
            b=b,
            c=c,
            d=d,
            mode=mode,
            centered=centered,
            unitary_projection=unitary_projection,
        )
    )
    return lambda x: mx.matmul(x, kernel)


def _is_complex(x: mx.array) -> bool:
    return x.dtype == mx.complex64


def linear_canonical_transform(
    x: mx.array,
    *,
    a: complex | float,
    b: complex | float,
    c: complex | float,
    d: complex | float | None = None,
    axis: int = -1,
    normalized: bool = True,
    normalization: NormMode | None = None,
    centered: bool = True,
    dense_threshold: int = 256,
    b_eps: float = 1e-6,
    unitary_projection: bool = True,
) -> mx.array:
    """Finite-dimensional 1-D discrete LCT along ``axis`` (fixed parameters)."""

    a_c = complex(a)
    b_c = complex(b)
    c_c = complex(c)
    d_c = symplectic_d(a_c, b_c, c_c) if d is None else complex(d)

    if axis != -1 and axis != x.ndim - 1:
        x = mx.moveaxis(x, axis, -1)
        moved = True
    else:
        moved = False

    plan = _compile_plan(
        length=x.shape[-1],
        a=a_c,
        b=b_c,
        c=c_c,
        d=d_c,
        mode=_mode(normalized, normalization),
        centered=centered,
        dense_threshold=dense_threshold,
        b_eps=b_eps,
        unitary_projection=unitary_projection,
    )
    y = plan(x.astype(mx.complex64))
    return mx.moveaxis(y, -1, axis) if moved else y


def _pack_real_pairs(
    x: mx.array,
    target_channels: int,
    *,
    mode: Literal["zero", "tile"] = "zero",
) -> mx.array:
    current = x.shape[-1]
    pad = target_channels - current
    if pad > 0:
        if mode == "tile" and current > 0:
            repeats = (target_channels + current - 1) // current
            x = mx.tile(x, [1] * (x.ndim - 1) + [repeats])[..., :target_channels]
        else:
            x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    pairs = x.reshape(*x.shape[:-1], target_channels // 2, 2)
    return (pairs[..., 0] + 1j * pairs[..., 1]).astype(mx.complex64)


def _unpack_real_pairs(z: mx.array, *, original_channels: int) -> mx.array:
    out = mx.stack([mx.real(z), mx.imag(z)], axis=-1)
    out = out.reshape(*z.shape[:-1], z.shape[-1] * 2)
    return out[..., :original_channels]


def _real_work_dtype(dtype: mx.Dtype) -> mx.Dtype:
    if dtype in (mx.float16, mx.bfloat16):
        return mx.float32
    return dtype


class LCTLayer(mlx_nn.Module):
    """Fixed-parameter finite LCT with precompiled FFT/chirp/dense plans."""

    def __init__(
        self,
        *,
        a: complex | float = 1.0,
        b: complex | float = 0.0,
        c: complex | float = 0.0,
        normalized: bool = True,
        normalization: NormMode | None = None,
        centered: bool = True,
        dense_threshold: int = 256,
        b_eps: float = 1e-6,
        unitary_projection: bool = True,
    ) -> None:
        super().__init__()
        self._a = complex(a)
        self._b = complex(b)
        self._c = complex(c)
        self._d = _torch_layer_d(a, b, c)
        self._mode: NormMode = _mode(normalized, normalization)
        self._centered = centered
        self._dense_threshold = dense_threshold
        self._b_eps = b_eps
        self._unitary_projection = unitary_projection
        self._plans: dict[int, Plan] = {}
        self._inverse_plans: dict[int, Plan] = {}

    @property
    def canonical_matrix(self) -> tuple[complex, complex, complex, complex]:
        return self._a, self._b, self._c, self._d

    def _plan(self, length: int, *, inverse: bool) -> Plan:
        cache = self._inverse_plans if inverse else self._plans
        plan = cache.get(length)
        if plan is None:
            if inverse:
                a, b, c, d = self._d, -self._b, -self._c, self._a
            else:
                a, b, c, d = self._a, self._b, self._c, self._d
            plan = _compile_plan(
                length=length,
                a=a,
                b=b,
                c=c,
                d=d,
                mode=self._mode,
                centered=self._centered,
                dense_threshold=self._dense_threshold,
                b_eps=self._b_eps,
                unitary_projection=self._unitary_projection,
            )
            cache[length] = plan
        return plan

    def _apply(self, x: mx.array, *, inverse: bool) -> mx.array:
        input_was_complex = _is_complex(x)
        y = self._plan(x.shape[-1], inverse=inverse)(x.astype(mx.complex64))
        if input_was_complex:
            return y
        return mx.real(y).astype(x.dtype)

    def __call__(self, x: mx.array) -> mx.array:
        return self._apply(x, inverse=False)

    def inverse(self, x: mx.array) -> mx.array:
        return self._apply(x, inverse=True)

    def matrix(self, length: int) -> mx.array:
        # mx.eye cannot build complex64 on GPU (scatter limitation); go via numpy.
        eye = mx.array(np.eye(length, dtype=np.complex64))
        return self._plan(length, inverse=False)(eye)

    @staticmethod
    def fractional_fourier(angle: float, **kwargs) -> "LCTLayer":
        # Mirror the torch staticmethod exactly: float32 cos/sin, because the
        # dense+QR path amplifies even 1-ulp parameter differences.
        import torch

        theta = torch.tensor(float(angle))
        a = torch.cos(theta).item()
        b = torch.sin(theta).item()
        return LCTLayer(a=a, b=b, c=-b, **kwargs)


class LCTModReLU(mlx_nn.Module):
    """LCT-domain modReLU activation (MLX port of the PyTorch layer)."""

    def __init__(
        self,
        channels: int,
        *,
        a: float = 0.0,
        b: float = 1.0,
        c: float = 0.0,
        bias_init: float = 0.1,
        inverse_after_nonlinearity: bool = False,
        residual_mix: float = 0.0,
        eps: float = 1e-6,
        dense_threshold: int = 256,
        normalization: NormMode = "unitary",
        unitary_projection: bool = True,
    ) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("channels must be at least 2")

        self.channels = channels
        self.padded_channels = channels if channels % 2 == 0 else channels + 1
        self.complex_channels = self.padded_channels // 2
        self.inverse_after_nonlinearity = inverse_after_nonlinearity
        self.eps = eps

        self.transform = LCTLayer(
            a=a,
            b=b,
            c=c,
            normalization=normalization,
            dense_threshold=dense_threshold,
            unitary_projection=unitary_projection,
        )
        self.bias = mx.full((self.complex_channels,), float(bias_init))
        self.residual_mix = mx.array(float(residual_mix))
        self.output_gain = mx.array(1.0)

    def _apply_modrelu(self, z: mx.array) -> mx.array:
        radius = mx.abs(z)
        bias = self.bias.reshape([1] * (z.ndim - 1) + [self.complex_channels])
        gain = mx.maximum(radius + bias, 0.0) / (radius + self.eps)
        return z * gain.astype(mx.complex64)

    def __call__(self, x: mx.array) -> mx.array:
        if _is_complex(x):
            spectral = self.transform(x)
            activated = self._apply_modrelu(spectral)
            if self.inverse_after_nonlinearity:
                activated = self.transform.inverse(activated)
            return activated

        work_dtype = _real_work_dtype(x.dtype)
        residual = x
        packed = _pack_real_pairs(x.astype(work_dtype), self.padded_channels)
        spectral = self.transform(packed)
        activated = self._apply_modrelu(spectral)
        if self.inverse_after_nonlinearity:
            activated = self.transform.inverse(activated)

        out = _unpack_real_pairs(activated, original_channels=self.channels)
        out = self.output_gain.astype(out.dtype) * out
        out = out + self.residual_mix.astype(out.dtype) * residual.astype(out.dtype)
        return out.astype(x.dtype)


LCTActivation = LCTModReLU


class LCTLinear(mlx_nn.Module):
    """Structured ``nn.Linear``-style layer built from LCT spectral mixing."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        a: float = 0.0,
        b: float = 1.0,
        c: float = 0.0,
        inverse_after_multiply: bool = True,
        dense_threshold: int = 32,
        normalization: NormMode = "unitary",
        unitary_projection: bool = False,
        expansion_mode: Literal["zero", "tile"] = "tile",
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.inverse_after_multiply = inverse_after_multiply
        self.expansion_mode: Literal["zero", "tile"] = expansion_mode
        self.normalization: NormMode = normalization
        self._direct_fft = (
            abs(a) <= 1e-6 and abs(b - 1.0) <= 1e-6 and abs(c) <= 1e-6
        )

        base_features = max(self.in_features, self.out_features)
        self.padded_features = (
            base_features if base_features % 2 == 0 else base_features + 1
        )
        self.complex_features = self.padded_features // 2

        self.transform = LCTLayer(
            a=a,
            b=b,
            c=c,
            normalization=normalization,
            dense_threshold=dense_threshold,
            unitary_projection=unitary_projection,
        )

        self.spectral_real = mx.ones((self.complex_features,))
        self.spectral_imag = mx.zeros((self.complex_features,))
        self.bias = mx.zeros((self.out_features,)) if bias else None

    @property
    def spectral_diag(self) -> mx.array:
        return self.spectral_real + 1j * self.spectral_imag

    def _apply_linear_map(self, x: mx.array) -> mx.array:
        if _is_complex(x):
            raise TypeError("LCTLinear expects a real-valued tensor")

        selected_mode: Literal["zero", "tile"] = (
            self.expansion_mode if self.out_features > self.in_features else "zero"
        )
        work_dtype = _real_work_dtype(x.dtype)
        packed = _pack_real_pairs(
            x.astype(work_dtype),
            self.padded_features,
            mode=selected_mode,
        )

        norm: NormMode = self.normalization
        if self._direct_fft:
            spectral = _fft(packed, "ortho" if norm == "unitary" else "backward")
        else:
            spectral = self.transform(packed)

        diag = self.spectral_diag.astype(mx.complex64)
        mixed = spectral * diag.reshape([1] * (spectral.ndim - 1) + [self.complex_features])

        if self.inverse_after_multiply:
            if self._direct_fft:
                mixed = _ifft(mixed, "ortho" if norm == "unitary" else "forward")
            else:
                mixed = self.transform.inverse(mixed)

        out = _unpack_real_pairs(mixed, original_channels=self.padded_features)
        return out[..., : self.out_features]

    def __call__(self, x: mx.array) -> mx.array:
        out = self._apply_linear_map(x)
        if self.bias is None:
            return out.astype(x.dtype)
        return (out + self.bias.astype(out.dtype)).astype(x.dtype)

    def materialize_weight(self) -> mx.array:
        basis = mx.eye(self.in_features, dtype=mx.float32)
        weight = self._apply_linear_map(basis).T
        return mx.stop_gradient(weight)

    def to_linear(self) -> mlx_nn.Linear:
        linear = mlx_nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
        )
        linear.weight = self.materialize_weight()
        if self.bias is not None:
            linear.bias = mx.stop_gradient(self.bias)
        return linear
