from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .functional import NormMode, linear_canonical_transform, symplectic_d

__all__ = [
    "LCTActivation",
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


def _pack_real_pairs(x: Tensor, target_channels: int) -> Tensor:
    pad = target_channels - x.size(-1)
    if pad > 0:
        x = F.pad(x, (0, pad))
    work_dtype = _real_work_dtype(x.dtype)
    x_work = x.to(work_dtype).contiguous()
    return torch.view_as_complex(x_work.reshape(*x_work.shape[:-1], target_channels // 2, 2))


def _unpack_real_pairs(z: Tensor, *, original_channels: int) -> Tensor:
    out = torch.view_as_real(z).reshape(*z.shape[:-1], -1)
    return out[..., :original_channels]


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
