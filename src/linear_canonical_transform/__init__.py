"""Compatibility exports for the older package name."""

from lct_activation import (
    LCTActivation,
    LCTLayer,
    LCTLinear,
    LCTModReLU,
    NormMode,
    canonical_determinant,
    chirpz_lct,
    compose_canonical,
    composition_error,
    finite_lct_matrix,
    linear_canonical_transform,
    property_report,
    relative_frobenius_error,
    symplectic_d,
    unitarity_error,
)

__all__ = [
    "NormMode",
    "LCTActivation",
    "LCTLayer",
    "LCTLinear",
    "LCTModReLU",
    "canonical_determinant",
    "chirpz_lct",
    "compose_canonical",
    "composition_error",
    "finite_lct_matrix",
    "linear_canonical_transform",
    "property_report",
    "relative_frobenius_error",
    "symplectic_d",
    "unitarity_error",
]
