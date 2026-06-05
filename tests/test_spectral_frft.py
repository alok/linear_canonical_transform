from __future__ import annotations

import math

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from lct_activation import (
    composition_error,
    finite_lct_matrix,
    relative_frobenius_error,
    spectral_fractional_fourier_matrix,
    spectral_fractional_fourier_transform,
    unitarity_error,
)


def _frft(angle: float) -> tuple[float, float, float]:
    return math.cos(angle), math.sin(angle), -math.sin(angle)


def test_spectral_frft_integer_powers_match_fft_and_ifft() -> None:
    length = 16
    eye = torch.eye(length, dtype=torch.complex64)

    fourier = spectral_fractional_fourier_matrix(length, math.pi / 2)
    inverse = spectral_fractional_fourier_matrix(length, -math.pi / 2)

    assert torch.allclose(fourier, torch.fft.fft(eye, norm="ortho"), atol=1e-5, rtol=0.0)
    assert torch.allclose(inverse, torch.fft.ifft(eye, norm="ortho"), atol=1e-5, rtol=0.0)


@settings(deadline=None, max_examples=25)
@given(
    length=st.integers(min_value=4, max_value=32),
    first_angle=st.floats(min_value=-1.2, max_value=1.2, allow_nan=False, allow_infinity=False),
    second_angle=st.floats(min_value=-1.2, max_value=1.2, allow_nan=False, allow_infinity=False),
)
def test_spectral_frft_is_unitary_and_compositional(
    length: int,
    first_angle: float,
    second_angle: float,
) -> None:
    assume(abs(first_angle) + abs(second_angle) >= 1e-3)

    first = spectral_fractional_fourier_matrix(length, first_angle)
    second = spectral_fractional_fourier_matrix(length, second_angle)
    composed = spectral_fractional_fourier_matrix(length, first_angle + second_angle)

    assert unitarity_error(first) <= 1e-5
    assert relative_frobenius_error(first @ second, composed) <= 1e-5


def test_spectral_frft_transform_matches_matrix_application() -> None:
    x = torch.randn(3, 10, dtype=torch.complex64)
    angle = 0.3

    y = spectral_fractional_fourier_transform(x, angle)
    matrix = spectral_fractional_fourier_matrix(10, angle)

    assert torch.allclose(y, x @ matrix, atol=1e-5, rtol=0.0)


def test_property_diagnostics_support_spectral_frft_discretization() -> None:
    length = 16
    first = _frft(0.5)
    second = _frft(-0.5)

    error = composition_error(
        length,
        first,
        second,
        discretization="spectral-frft",
    )
    matrix = finite_lct_matrix(
        length,
        first,
        discretization="spectral-frft",
    )

    assert unitarity_error(matrix) <= 1e-5
    assert error <= 1e-5
