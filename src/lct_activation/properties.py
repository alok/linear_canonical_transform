from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor

from .functional import (
    NormMode,
    linear_canonical_transform,
    spectral_fractional_fourier_matrix,
    symplectic_d,
)

Scalar = float | complex
CanonicalParams = tuple[Scalar, Scalar, Scalar] | tuple[Scalar, Scalar, Scalar, Scalar]
DiscretizationMode = Literal["lct", "spectral-frft"]

__all__ = [
    "CanonicalParams",
    "DiscretizationMode",
    "FiniteLCTPropertyAssessment",
    "FiniteLCTPropertyReport",
    "FiniteLCTPropertySweepRow",
    "FiniteLCTPropertyThresholds",
    "assess_property_report",
    "canonical_determinant",
    "compose_canonical",
    "composition_error",
    "finite_lct_matrix",
    "format_property_sweep_markdown",
    "property_report",
    "property_sweep",
    "relative_frobenius_error",
    "unitarity_error",
]


@dataclass(frozen=True)
class FiniteLCTPropertyReport:
    """Finite-grid diagnostics for a pair of canonical transforms."""

    length: int
    discretization: DiscretizationMode
    normalization: NormMode
    unitary_projection: bool
    centered: bool
    first: tuple[complex, complex, complex, complex]
    second: tuple[complex, complex, complex, complex]
    composed: tuple[complex, complex, complex, complex]
    first_determinant_error: float
    second_determinant_error: float
    composed_determinant_error: float
    first_unitarity_error: float
    second_unitarity_error: float
    composed_unitarity_error: float
    composition_error: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FiniteLCTPropertyThresholds:
    """Thresholds for checking finite-grid preservation claims."""

    max_determinant_error: float = 1e-8
    max_unitarity_error: float = 1e-5
    max_composition_error: float = 1e-5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FiniteLCTPropertyAssessment:
    """Pass/fail assessment for a finite-grid property report."""

    report: FiniteLCTPropertyReport
    thresholds: FiniteLCTPropertyThresholds
    determinant_ok: bool
    unitarity_ok: bool
    composition_ok: bool
    ok: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "determinant_ok": self.determinant_ok,
            "unitarity_ok": self.unitarity_ok,
            "composition_ok": self.composition_ok,
            "thresholds": self.thresholds.as_dict(),
            "report": self.report.as_dict(),
        }


@dataclass(frozen=True)
class FiniteLCTPropertySweepRow:
    """Compact finite-grid diagnostics for one FrFT angle pair."""

    length: int
    first_angle_degrees: float
    second_angle_degrees: float
    discretization: DiscretizationMode
    normalization: NormMode
    unitary_projection: bool
    first_unitarity_error: float
    second_unitarity_error: float
    composed_unitarity_error: float
    composition_error: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_complex(value: Scalar) -> complex:
    return complex(value)


def _complete(params: CanonicalParams) -> tuple[complex, complex, complex, complex]:
    if len(params) == 4:
        a, b, c, d = params
        return _as_complex(a), _as_complex(b), _as_complex(c), _as_complex(d)
    a, b, c = params
    d = symplectic_d(a, b, c)
    return _as_complex(a), _as_complex(b), _as_complex(c), _as_complex(d)  # type: ignore[arg-type]


def canonical_determinant(params: CanonicalParams) -> complex:
    """Return ``ad - bc`` for canonical LCT parameters."""

    a, b, c, d = _complete(params)
    return a * d - b * c


def _frft_angle(params: CanonicalParams) -> float:
    a, b, c, d = _complete(params)
    tol = 1e-5
    if any(abs(value.imag) > tol for value in (a, b, c, d)):
        raise ValueError("spectral-frft discretization expects real FrFT parameters")
    if abs(c.real + b.real) > tol or abs(d.real - a.real) > tol:
        raise ValueError("spectral-frft discretization expects (cos(theta), sin(theta), -sin(theta))")
    if abs(a.real * a.real + b.real * b.real - 1.0) > 1e-4:
        raise ValueError("spectral-frft discretization expects a^2 + b^2 ~= 1")
    return math.atan2(b.real, a.real)


def _frft_params_degrees(angle_degrees: float) -> tuple[float, float, float]:
    theta = math.radians(angle_degrees)
    return math.cos(theta), math.sin(theta), -math.sin(theta)


def compose_canonical(first: CanonicalParams, second: CanonicalParams) -> tuple[complex, complex, complex, complex]:
    """Compose two canonical matrices for applying ``first`` then ``second``.

    The finite LCT implementation applies row vectors as ``x @ K``. If ``K1``
    is built from ``first`` and ``K2`` from ``second``, then the finite
    composition is compared as ``K1 @ K2`` against the matrix produced from this
    returned canonical product.
    """

    a1, b1, c1, d1 = _complete(first)
    a2, b2, c2, d2 = _complete(second)
    return (
        a2 * a1 + b2 * c1,
        a2 * b1 + b2 * d1,
        c2 * a1 + d2 * c1,
        c2 * b1 + d2 * d1,
    )


def finite_lct_matrix(
    length: int,
    params: CanonicalParams,
    *,
    normalization: NormMode = "unitary",
    centered: bool = True,
    unitary_projection: bool = True,
    dense_threshold: int | None = None,
    discretization: DiscretizationMode = "lct",
    device: torch.device | str | None = None,
) -> Tensor:
    """Materialize the finite LCT matrix used by this package.

    This intentionally uses the same public transform implementation as the
    layers, with a dense threshold high enough to make small-grid diagnostics
    inspect the reference kernel unless the caller chooses otherwise.
    """

    if length <= 0:
        raise ValueError("length must be positive")

    dev = torch.device("cpu") if device is None else torch.device(device)
    a, b, c, d = _complete(params)
    if discretization == "spectral-frft":
        if normalization != "unitary":
            raise ValueError("spectral-frft discretization is unitary-only")
        return spectral_fractional_fourier_matrix(
            length,
            _frft_angle((a, b, c, d)),
            dtype=torch.complex64,
            device=dev,
        ).detach()

    eye = torch.eye(length, dtype=torch.complex64, device=dev)
    threshold = max(length + 1, 256) if dense_threshold is None else dense_threshold
    return linear_canonical_transform(
        eye,
        a=a,
        b=b,
        c=c,
        d=d,
        normalization=normalization,
        centered=centered,
        dense_threshold=threshold,
        unitary_projection=unitary_projection,
    ).detach()


def relative_frobenius_error(actual: Tensor, expected: Tensor) -> float:
    """Return ``||actual - expected||_F / ||expected||_F`` as a Python float."""

    denom = torch.linalg.matrix_norm(expected)
    if float(denom.detach().cpu()) == 0.0:
        denom = torch.ones((), dtype=expected.real.dtype, device=expected.device)
    error = torch.linalg.matrix_norm(actual - expected) / denom
    return float(error.detach().cpu())


def unitarity_error(matrix: Tensor) -> float:
    """Return the relative Frobenius error of ``matrixᴴ matrix = I``."""

    ident = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
    return relative_frobenius_error(matrix.conj().transpose(-2, -1) @ matrix, ident)


def composition_error(
    length: int,
    first: CanonicalParams,
    second: CanonicalParams,
    *,
    normalization: NormMode = "unitary",
    centered: bool = True,
    unitary_projection: bool = True,
    dense_threshold: int | None = None,
    discretization: DiscretizationMode = "lct",
    device: torch.device | str | None = None,
) -> float:
    """Measure finite-grid composition error for applying ``first`` then ``second``."""

    first_matrix = finite_lct_matrix(
        length,
        first,
        normalization=normalization,
        centered=centered,
        unitary_projection=unitary_projection,
        dense_threshold=dense_threshold,
        discretization=discretization,
        device=device,
    )
    second_matrix = finite_lct_matrix(
        length,
        second,
        normalization=normalization,
        centered=centered,
        unitary_projection=unitary_projection,
        dense_threshold=dense_threshold,
        discretization=discretization,
        device=device,
    )
    composed_matrix = finite_lct_matrix(
        length,
        compose_canonical(first, second),
        normalization=normalization,
        centered=centered,
        unitary_projection=unitary_projection,
        dense_threshold=dense_threshold,
        discretization=discretization,
        device=device,
    )
    return relative_frobenius_error(first_matrix @ second_matrix, composed_matrix)


def property_report(
    length: int,
    first: CanonicalParams,
    second: CanonicalParams,
    *,
    normalization: NormMode = "unitary",
    centered: bool = True,
    unitary_projection: bool = True,
    dense_threshold: int | None = None,
    discretization: DiscretizationMode = "lct",
    device: torch.device | str | None = None,
) -> FiniteLCTPropertyReport:
    """Summarize determinant, unitarity, and composition diagnostics."""

    first_complete = _complete(first)
    second_complete = _complete(second)
    composed = compose_canonical(first_complete, second_complete)
    first_matrix = finite_lct_matrix(
        length,
        first_complete,
        normalization=normalization,
        centered=centered,
        unitary_projection=unitary_projection,
        dense_threshold=dense_threshold,
        discretization=discretization,
        device=device,
    )
    second_matrix = finite_lct_matrix(
        length,
        second_complete,
        normalization=normalization,
        centered=centered,
        unitary_projection=unitary_projection,
        dense_threshold=dense_threshold,
        discretization=discretization,
        device=device,
    )
    composed_matrix = finite_lct_matrix(
        length,
        composed,
        normalization=normalization,
        centered=centered,
        unitary_projection=unitary_projection,
        dense_threshold=dense_threshold,
        discretization=discretization,
        device=device,
    )

    return FiniteLCTPropertyReport(
        length=length,
        discretization=discretization,
        normalization=normalization,
        unitary_projection=unitary_projection,
        centered=centered,
        first=first_complete,
        second=second_complete,
        composed=composed,
        first_determinant_error=abs(canonical_determinant(first_complete) - 1.0),
        second_determinant_error=abs(canonical_determinant(second_complete) - 1.0),
        composed_determinant_error=abs(canonical_determinant(composed) - 1.0),
        first_unitarity_error=unitarity_error(first_matrix),
        second_unitarity_error=unitarity_error(second_matrix),
        composed_unitarity_error=unitarity_error(composed_matrix),
        composition_error=relative_frobenius_error(first_matrix @ second_matrix, composed_matrix),
    )


def assess_property_report(
    report: FiniteLCTPropertyReport,
    thresholds: FiniteLCTPropertyThresholds | None = None,
) -> FiniteLCTPropertyAssessment:
    """Check determinant, unitarity, and composition errors against thresholds."""

    active_thresholds = thresholds or FiniteLCTPropertyThresholds()
    determinant_ok = (
        report.first_determinant_error <= active_thresholds.max_determinant_error
        and report.second_determinant_error <= active_thresholds.max_determinant_error
        and report.composed_determinant_error <= active_thresholds.max_determinant_error
    )
    unitarity_ok = (
        report.first_unitarity_error <= active_thresholds.max_unitarity_error
        and report.second_unitarity_error <= active_thresholds.max_unitarity_error
        and report.composed_unitarity_error <= active_thresholds.max_unitarity_error
    )
    composition_ok = report.composition_error <= active_thresholds.max_composition_error
    return FiniteLCTPropertyAssessment(
        report=report,
        thresholds=active_thresholds,
        determinant_ok=determinant_ok,
        unitarity_ok=unitarity_ok,
        composition_ok=composition_ok,
        ok=determinant_ok and unitarity_ok and composition_ok,
    )


def property_sweep(
    *,
    lengths: list[int] | tuple[int, ...],
    angle_pairs_degrees: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    discretizations: list[DiscretizationMode] | tuple[DiscretizationMode, ...] = ("lct", "spectral-frft"),
    normalization: NormMode = "unitary",
    centered: bool = True,
    unitary_projection: bool = True,
    device: torch.device | str | None = None,
) -> list[FiniteLCTPropertySweepRow]:
    """Sweep finite-grid diagnostics over FrFT angle pairs and grid lengths."""

    rows: list[FiniteLCTPropertySweepRow] = []
    for length in lengths:
        if length <= 0:
            raise ValueError("lengths must be positive")
        for first_angle, second_angle in angle_pairs_degrees:
            first = _frft_params_degrees(first_angle)
            second = _frft_params_degrees(second_angle)
            for discretization in discretizations:
                report = property_report(
                    length,
                    first,
                    second,
                    normalization=normalization,
                    centered=centered,
                    unitary_projection=unitary_projection,
                    discretization=discretization,
                    device=device,
                )
                rows.append(
                    FiniteLCTPropertySweepRow(
                        length=length,
                        first_angle_degrees=float(first_angle),
                        second_angle_degrees=float(second_angle),
                        discretization=discretization,
                        normalization=normalization,
                        unitary_projection=unitary_projection,
                        first_unitarity_error=report.first_unitarity_error,
                        second_unitarity_error=report.second_unitarity_error,
                        composed_unitarity_error=report.composed_unitarity_error,
                        composition_error=report.composition_error,
                    )
                )
    return rows


def _format_sweep_float(value: float) -> str:
    return f"{value:.3e}"


def format_property_sweep_markdown(rows: list[FiniteLCTPropertySweepRow]) -> str:
    """Format property sweep rows as a compact Markdown table."""

    headers = [
        "length",
        "first deg",
        "second deg",
        "discretization",
        "first unit",
        "second unit",
        "composed unit",
        "composition",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            str(row.length),
            f"{row.first_angle_degrees:g}",
            f"{row.second_angle_degrees:g}",
            row.discretization,
            _format_sweep_float(row.first_unitarity_error),
            _format_sweep_float(row.second_unitarity_error),
            _format_sweep_float(row.composed_unitarity_error),
            _format_sweep_float(row.composition_error),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
