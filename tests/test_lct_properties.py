from __future__ import annotations

import math

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from lct_activation import (
    FiniteLCTPropertyThresholds,
    assess_property_report,
    canonical_determinant,
    compose_canonical,
    composition_error,
    finite_lct_matrix,
    format_property_sweep_markdown,
    property_report,
    property_sweep,
    relative_frobenius_error,
    unitarity_error,
)


def _frft(angle: float) -> tuple[float, float, float]:
    return math.cos(angle), math.sin(angle), -math.sin(angle)


@settings(deadline=None, max_examples=25)
@given(
    length=st.integers(min_value=4, max_value=24),
    angle=st.floats(min_value=-1.2, max_value=1.2, allow_nan=False, allow_infinity=False),
)
def test_unitary_projection_preserves_finite_grid_unitarity(length: int, angle: float) -> None:
    assume(abs(math.sin(angle)) >= 0.15)

    matrix = finite_lct_matrix(
        length,
        _frft(angle),
        normalization="unitary",
        unitary_projection=True,
    )

    assert unitarity_error(matrix) <= 5e-5


@settings(deadline=None, max_examples=30)
@given(
    first_angle=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    second_angle=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_canonical_composition_preserves_symplectic_determinant(
    first_angle: float,
    second_angle: float,
) -> None:
    assume(abs(math.sin(first_angle)) >= 0.1)
    assume(abs(math.sin(second_angle)) >= 0.1)

    composed = compose_canonical(_frft(first_angle), _frft(second_angle))

    assert abs(canonical_determinant(composed) - 1.0) <= 1e-10


@settings(deadline=None, max_examples=20)
@given(length=st.integers(min_value=2, max_value=64))
def test_fourier_shortcut_composes_with_inverse_on_finite_grid(length: int) -> None:
    fourier = finite_lct_matrix(length, (0.0, 1.0, 0.0), normalization="unitary")
    inverse_fourier = finite_lct_matrix(length, (0.0, -1.0, 0.0), normalization="unitary")
    ident = torch.eye(length, dtype=torch.complex64)

    assert relative_frobenius_error(fourier @ inverse_fourier, ident) <= 5e-6


def test_projection_tradeoff_is_measurable_for_generic_frft_pair() -> None:
    length = 16
    first = _frft(0.5)
    second = _frft(-0.5)

    raw = finite_lct_matrix(
        length,
        first,
        normalization="unitary",
        unitary_projection=False,
    )
    projected = finite_lct_matrix(
        length,
        first,
        normalization="unitary",
        unitary_projection=True,
    )
    raw_composition_error = composition_error(
        length,
        first,
        second,
        normalization="unitary",
        unitary_projection=False,
    )
    projected_composition_error = composition_error(
        length,
        first,
        second,
        normalization="unitary",
        unitary_projection=True,
    )

    assert unitarity_error(projected) <= 5e-5
    assert unitarity_error(raw) >= 1e-2
    assert raw_composition_error < projected_composition_error


def test_property_report_exposes_diagnostics() -> None:
    report = property_report(
        16,
        _frft(0.5),
        _frft(-0.5),
        normalization="unitary",
        unitary_projection=True,
    )

    assert report.length == 16
    assert report.normalization == "unitary"
    assert report.unitary_projection is True
    assert report.first_determinant_error <= 1e-10
    assert report.second_determinant_error <= 1e-10
    assert report.composed_determinant_error <= 1e-10
    assert report.first_unitarity_error <= 5e-5
    assert report.second_unitarity_error <= 5e-5
    assert report.composition_error > 1e-2
    assert report.as_dict()["length"] == 16


def test_spectral_frft_report_exposes_low_composition_error() -> None:
    report = property_report(
        16,
        _frft(0.5),
        _frft(-0.5),
        discretization="spectral-frft",
    )

    assert report.discretization == "spectral-frft"
    assert report.first_unitarity_error <= 1e-5
    assert report.second_unitarity_error <= 1e-5
    assert report.composition_error <= 1e-5


def test_property_assessment_marks_spectral_frft_as_passing() -> None:
    report = property_report(
        16,
        _frft(0.5),
        _frft(-0.5),
        discretization="spectral-frft",
    )

    assessment = assess_property_report(report, FiniteLCTPropertyThresholds())

    assert assessment.ok is True
    assert assessment.determinant_ok is True
    assert assessment.unitarity_ok is True
    assert assessment.composition_ok is True
    assert assessment.as_dict()["ok"] is True


def test_property_assessment_keeps_sampled_tradeoff_visible() -> None:
    report = property_report(
        16,
        _frft(0.5),
        _frft(-0.5),
        normalization="unitary",
        unitary_projection=True,
    )

    assessment = assess_property_report(report, FiniteLCTPropertyThresholds(max_composition_error=1e-5))

    assert assessment.ok is False
    assert assessment.determinant_ok is True
    assert assessment.unitarity_ok is True
    assert assessment.composition_ok is False


def test_property_thresholds_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="max_composition_error"):
        FiniteLCTPropertyThresholds(max_composition_error=-1.0)


def test_property_sweep_compares_sampled_and_spectral_discretizations() -> None:
    rows = property_sweep(
        lengths=[8],
        angle_pairs_degrees=[(30.0, -30.0)],
        discretizations=("lct", "spectral-frft"),
    )

    assert [row.discretization for row in rows] == ["lct", "spectral-frft"]
    sampled = rows[0]
    spectral = rows[1]
    assert sampled.first_unitarity_error <= 5e-5
    assert sampled.composition_error > 1e-2
    assert spectral.first_unitarity_error <= 1e-5
    assert spectral.composition_error <= 1e-5


def test_property_sweep_markdown_table() -> None:
    rows = property_sweep(
        lengths=[8],
        angle_pairs_degrees=[(30.0, -30.0)],
        discretizations=("spectral-frft",),
    )

    table = format_property_sweep_markdown(rows)

    assert "| length | first deg | second deg |" in table
    assert "| 8 | 30 | -30 | spectral-frft |" in table
