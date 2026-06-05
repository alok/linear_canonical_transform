from __future__ import annotations

import math

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from lct_activation import (
    canonical_determinant,
    compose_canonical,
    composition_error,
    finite_lct_matrix,
    property_report,
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
