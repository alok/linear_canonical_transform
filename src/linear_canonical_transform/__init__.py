"""Compatibility exports for the older package name."""

from lct_activation import (
    LCTActivation,
    LCTLayer,
    LCTLinear,
    LCTModReLU,
    NormMode,
    chirpz_lct,
    linear_canonical_transform,
    symplectic_d,
)

__all__ = [
    "NormMode",
    "LCTActivation",
    "LCTLayer",
    "LCTLinear",
    "LCTModReLU",
    "chirpz_lct",
    "linear_canonical_transform",
    "symplectic_d",
]
