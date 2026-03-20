from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .functional import NormMode, linear_canonical_transform, symplectic_d
from .triton_ops import HAS_TRITON, complex_mul_conj_diag_and_grad, complex_pointwise_mul

__all__ = [
    "LCTActivation",
    "LCTLinear",
    "LCTLayer",
    "LCTModReLU",
]


class LCTLayer(nn.Module):
    """Differentiable finite-dimensional LCT with fast FFT/Chirp-Z paths."""

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

        def _param(v: complex | float) -> nn.Parameter:
            dtype = torch.complex64 if isinstance(v, complex) else torch.float32
            return nn.Parameter(torch.tensor(v, dtype=dtype))

        self.a = _param(a)
        self.b = _param(b)
        self.c = _param(c)
        self.normalized = normalized
        self.normalization = normalization
        self.centered = centered
        self.dense_threshold = dense_threshold
        self.b_eps = b_eps
        self.unitary_projection = unitary_projection

    @property
    def d(self) -> Tensor:
        return torch.as_tensor(
            symplectic_d(self.a, self.b, self.c),
            dtype=torch.complex64,
            device=self.a.device,
        )

    @property
    def canonical_matrix(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.a, self.b, self.c, self.d

    def forward(self, x: Tensor) -> Tensor:
        input_was_complex = torch.is_complex(x)
        x_complex = x.to(torch.complex64)
        y = linear_canonical_transform(
            x_complex,
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            dim=-1,
            normalized=self.normalized,
            normalization=self.normalization,
            centered=self.centered,
            dense_threshold=self.dense_threshold,
            b_eps=self.b_eps,
            unitary_projection=self.unitary_projection,
        )
        if input_was_complex:
            return y.to(x.dtype)
        return y.real.to(x.dtype)

    def inverse(self, x: Tensor) -> Tensor:
        input_was_complex = torch.is_complex(x)
        x_complex = x.to(torch.complex64)
        y = linear_canonical_transform(
            x_complex,
            a=self.d,
            b=-self.b,
            c=-self.c,
            d=self.a,
            dim=-1,
            normalized=self.normalized,
            normalization=self.normalization,
            centered=self.centered,
            dense_threshold=self.dense_threshold,
            b_eps=self.b_eps,
            unitary_projection=self.unitary_projection,
        )
        if input_was_complex:
            return y.to(x.dtype)
        return y.real.to(x.dtype)

    def inverse_layer(self) -> "LCTLayer":
        return LCTLayer(
            a=complex(self.d.detach().cpu().item()),
            b=complex((-self.b).detach().cpu().item()),
            c=complex((-self.c).detach().cpu().item()),
            normalized=self.normalized,
            normalization=self.normalization,
            centered=self.centered,
            dense_threshold=self.dense_threshold,
            b_eps=self.b_eps,
            unitary_projection=self.unitary_projection,
        )

    def matrix(self, length: int, *, device: torch.device | None = None) -> Tensor:
        dev = device or self.a.device
        eye = torch.eye(length, dtype=torch.complex64, device=dev)
        return self.forward(eye)

    @staticmethod
    def fresnel(
        length: int,
        *,
        wavelength: float = 1.0,
        distance: float = 1.0,
        **kwargs,
    ) -> "LCTLayer":
        del length
        return LCTLayer(a=1.0, b=wavelength * distance, c=0.0, **kwargs)

    @staticmethod
    def fractional_fourier(angle: float | Tensor, **kwargs) -> "LCTLayer":
        theta = float(angle) if isinstance(angle, Tensor) else angle
        a = torch.cos(torch.tensor(theta)).item()
        b = torch.sin(torch.tensor(theta)).item()
        return LCTLayer(a=a, b=b, c=-b, **kwargs)


def _real_work_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _pack_real_pairs(x: Tensor, target_channels: int, *, mode: Literal["zero", "tile"] = "zero") -> Tensor:
    x = _expand_real_features(x, target_channels, mode=mode)
    work_dtype = _real_work_dtype(x.dtype)
    x_work = x.to(work_dtype).contiguous()
    return torch.view_as_complex(x_work.reshape(*x_work.shape[:-1], target_channels // 2, 2))


def _expand_real_features(x: Tensor, target_channels: int, *, mode: Literal["zero", "tile"] = "zero") -> Tensor:
    current = x.size(-1)
    pad = target_channels - current
    if pad > 0:
        if mode == "tile" and current > 0:
            repeats = (target_channels + current - 1) // current
            x = x.repeat(*([1] * (x.ndim - 1)), repeats)[..., :target_channels]
        else:
            x = F.pad(x, (0, pad))
    return x


def _reduce_expanded_features(
    grad: Tensor,
    *,
    original_channels: int,
    expanded_channels: int,
    mode: Literal["zero", "tile"] = "zero",
) -> Tensor:
    if expanded_channels == original_channels:
        return grad
    if mode == "zero":
        return grad[..., :original_channels]

    repeats = (expanded_channels + original_channels - 1) // original_channels
    indices = torch.div(
        torch.arange(expanded_channels, device=grad.device),
        repeats,
        rounding_mode="floor",
    ).clamp(max=original_channels - 1)
    out = torch.zeros(*grad.shape[:-1], original_channels, dtype=grad.dtype, device=grad.device)
    out.index_add_(-1, indices, grad)
    return out


def _unpack_real_pairs(z: Tensor, *, original_channels: int) -> Tensor:
    out = torch.view_as_real(z).reshape(*z.shape[:-1], -1)
    return out[..., :original_channels]


def _norm_mode(normalization: NormMode | None) -> NormMode:
    return "unitary" if normalization is None else normalization


def _fft_norm(mode: NormMode) -> str:
    return "ortho" if mode == "unitary" else "backward"


def _ifft_norm(mode: NormMode) -> str:
    return "ortho" if mode == "unitary" else "forward"


class _DirectFFTLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        spectral_real: Tensor,
        spectral_imag: Tensor,
        in_features: int,
        out_features: int,
        padded_features: int,
        expansion_mode: str,
        inverse_after_multiply: bool,
        norm_mode: str,
        use_triton: bool,
    ) -> Tensor:
        work_dtype = _real_work_dtype(x.dtype)
        expanded = _expand_real_features(
            x.to(work_dtype),
            padded_features,
            mode=expansion_mode,
        )
        packed = torch.view_as_complex(expanded.contiguous().reshape(*expanded.shape[:-1], padded_features // 2, 2))
        spectral = torch.fft.fft(packed, dim=-1, norm=_fft_norm(norm_mode))
        diag = torch.complex(spectral_real.to(work_dtype), spectral_imag.to(work_dtype))
        mixed = complex_pointwise_mul(spectral, diag, use_triton=use_triton)
        transformed = (
            torch.fft.ifft(mixed, dim=-1, norm=_ifft_norm(norm_mode))
            if inverse_after_multiply
            else mixed
        )
        out = _unpack_real_pairs(transformed, original_channels=padded_features)[..., :out_features]

        ctx.save_for_backward(spectral, diag)
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.padded_features = padded_features
        ctx.expansion_mode = expansion_mode
        ctx.inverse_after_multiply = inverse_after_multiply
        ctx.norm_mode = norm_mode
        ctx.use_triton = use_triton
        ctx.input_dtype = x.dtype
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        spectral, diag = ctx.saved_tensors
        work_dtype = _real_work_dtype(grad_out.dtype)

        grad_expanded = F.pad(
            grad_out.to(work_dtype),
            (0, ctx.padded_features - ctx.out_features),
        )
        grad_packed = torch.view_as_complex(
            grad_expanded.contiguous().reshape(*grad_expanded.shape[:-1], ctx.padded_features // 2, 2)
        )

        spectral_grad = (
            torch.fft.fft(grad_packed, dim=-1, norm=_fft_norm(ctx.norm_mode))
            if ctx.inverse_after_multiply
            else grad_packed
        )
        grad_spectral, grad_real, grad_imag = complex_mul_conj_diag_and_grad(
            spectral,
            spectral_grad,
            diag,
            use_triton=ctx.use_triton,
        )
        grad_input_packed = torch.fft.ifft(
            grad_spectral,
            dim=-1,
            norm=_ifft_norm(ctx.norm_mode),
        )
        grad_input_expanded = _unpack_real_pairs(grad_input_packed, original_channels=ctx.padded_features)
        grad_input = _reduce_expanded_features(
            grad_input_expanded,
            original_channels=ctx.in_features,
            expanded_channels=ctx.padded_features,
            mode=ctx.expansion_mode,
        ).to(ctx.input_dtype)
        grad_real = grad_real.to(spectral.real.dtype)
        grad_imag = grad_imag.to(spectral.real.dtype)

        return grad_input, grad_real, grad_imag, None, None, None, None, None, None, None


class LCTModReLU(nn.Module):
    """LCT-domain modReLU activation for real or complex inputs."""

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
        self.bias = nn.Parameter(torch.full((self.complex_channels,), float(bias_init)))
        self.residual_mix = nn.Parameter(torch.tensor(float(residual_mix)))
        self.output_gain = nn.Parameter(torch.tensor(1.0))

    def _apply_modrelu(self, z: Tensor) -> Tensor:
        radius = torch.abs(z)
        bias = self.bias.to(radius.dtype).view(*([1] * (z.ndim - 1)), self.complex_channels)
        gain = F.relu(radius + bias) / (radius + self.eps)
        return z * gain.to(z.dtype)

    def forward(self, x: Tensor) -> Tensor:
        if torch.is_complex(x):
            spectral = self.transform(x)
            activated = self._apply_modrelu(spectral)
            if self.inverse_after_nonlinearity:
                activated = self.transform.inverse(activated)
            return activated

        residual = x
        packed = _pack_real_pairs(x, self.padded_channels)
        spectral = self.transform(packed)
        activated = self._apply_modrelu(spectral)
        if self.inverse_after_nonlinearity:
            activated = self.transform.inverse(activated)

        out = _unpack_real_pairs(activated, original_channels=self.channels)
        out = self.output_gain.to(out.dtype) * out
        return out + self.residual_mix.to(out.dtype) * residual


LCTActivation = LCTModReLU


class LCTLinear(nn.Module):
    """Structured ``nn.Linear``-style layer built from LCT spectral mixing.

    The layer packs real features into complex pairs, applies an LCT, multiplies
    by a learnable complex diagonal in the transform domain, optionally maps
    back through the inverse transform, then crops to ``out_features``.
    """

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
        learnable_transform: bool = False,
        expansion_mode: Literal["zero", "tile"] = "tile",
        use_triton_kernels: bool | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.inverse_after_multiply = inverse_after_multiply
        self.expansion_mode = expansion_mode
        self.normalization = normalization
        self.use_triton_kernels = HAS_TRITON if use_triton_kernels is None else use_triton_kernels
        self._direct_fft = (
            not learnable_transform
            and abs(a) <= 1e-6
            and abs(b - 1.0) <= 1e-6
            and abs(c) <= 1e-6
        )

        base_features = max(self.in_features, self.out_features)
        self.padded_features = base_features if base_features % 2 == 0 else base_features + 1
        self.complex_features = self.padded_features // 2

        self.transform = LCTLayer(
            a=a,
            b=b,
            c=c,
            normalization=normalization,
            dense_threshold=dense_threshold,
            unitary_projection=unitary_projection,
        )
        if not learnable_transform:
            self.transform.a.requires_grad_(False)
            self.transform.b.requires_grad_(False)
            self.transform.c.requires_grad_(False)

        self.spectral_real = nn.Parameter(torch.ones(self.complex_features))
        self.spectral_imag = nn.Parameter(torch.zeros(self.complex_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

    @property
    def spectral_diag(self) -> Tensor:
        return torch.complex(self.spectral_real, self.spectral_imag)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.spectral_real.fill_(1.0)
            self.spectral_imag.zero_()
            if self.bias is not None:
                self.bias.zero_()

    def _apply_linear_map(self, x: Tensor) -> Tensor:
        if torch.is_complex(x):
            raise TypeError("LCTLinear expects a real-valued tensor")

        selected_mode = self.expansion_mode if self.out_features > self.in_features else "zero"
        norm_mode = _norm_mode(self.normalization)

        if self._direct_fft and norm_mode == "unitary":
            return _DirectFFTLinearFn.apply(
                x,
                self.spectral_real,
                self.spectral_imag,
                self.in_features,
                self.out_features,
                self.padded_features,
                selected_mode,
                self.inverse_after_multiply,
                norm_mode,
                bool(self.use_triton_kernels and x.is_cuda),
            )

        packed = _pack_real_pairs(
            x,
            self.padded_features,
            mode=selected_mode,
        )
        if self._direct_fft:
            spectral = torch.fft.fft(packed, dim=-1, norm=_fft_norm(norm_mode))
        else:
            spectral = self.transform(packed)

        diag = self.spectral_diag.to(spectral.dtype)
        mixed = complex_pointwise_mul(
            spectral,
            diag,
            use_triton=bool(self.use_triton_kernels and spectral.is_cuda),
        )
        if self.inverse_after_multiply:
            if self._direct_fft:
                mixed = torch.fft.ifft(mixed, dim=-1, norm=_ifft_norm(norm_mode))
            else:
                mixed = self.transform.inverse(mixed)

        out = _unpack_real_pairs(mixed, original_channels=self.padded_features)
        return out[..., : self.out_features]

    def forward(self, x: Tensor) -> Tensor:
        out = self._apply_linear_map(x)
        if self.bias is None:
            return out.to(x.dtype)
        return (out + self.bias.to(out.dtype)).to(x.dtype)

    @torch.no_grad()
    def materialize_weight(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        dev = device or self.spectral_real.device
        basis = torch.eye(self.in_features, device=dev, dtype=dtype)
        return self._apply_linear_map(basis).transpose(0, 1).contiguous()

    @torch.no_grad()
    def to_linear(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> nn.Linear:
        dev = device or self.spectral_real.device
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=dev,
            dtype=dtype,
        )
        linear.weight.copy_(self.materialize_weight(device=dev, dtype=dtype))
        if self.bias is not None and linear.bias is not None:
            linear.bias.copy_(self.bias.to(device=dev, dtype=dtype))
        return linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, inverse_after_multiply={self.inverse_after_multiply}, "
            f"complex_features={self.complex_features}, expansion_mode={self.expansion_mode}"
        )
