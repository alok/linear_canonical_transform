from typing import Tuple
import pytest
import jax.numpy as jnp
from hypothesis import given, strategies as st
from jaxtyping import Array, Complex
from linear_canonical_transform import (
    lct, SL2C, fourier_transform, inverse_fourier_transform,
    laplace_transform, fractional_fourier_transform
)


# Strategy for generating valid SL(2,C) matrices
@st.composite
def sl2c_matrix(draw) -> SL2C:
    """
    Generate a random SL(2,C) matrix with determinant 1.
    We use the fact that any SL(2,C) matrix can be parameterized as:
    [[a, b], [c, (1+bc)/a]] where a,b,c are complex numbers and a ≠ 0
    """
    # Generate complex numbers directly using hypothesis
    complex_strategy = st.complex_numbers(
        min_magnitude=0.1,  # Avoid a=0
        max_magnitude=2.0,
        allow_infinity=False,
        allow_nan=False,
    )

    a = draw(complex_strategy)
    b = draw(complex_strategy)
    c = draw(complex_strategy)

    # Compute d to ensure determinant is 1
    d = (1 + b * c) / a
    mat = jnp.array([[a, b], [c, d]], dtype=jnp.complex64)
    assert jnp.isclose(jnp.linalg.det(mat), 1), "Matrix must have determinant 1"
    return mat


# Strategy for generating test functions and grids
@st.composite
def lct_inputs(
    draw,
) -> Tuple[Complex[Array, "x"], Complex[Array, "x"], Complex[Array, "x"]]:
    """Generate input function and grids for LCT testing"""
    n_points = 32  # Fixed size for stability
    x = jnp.linspace(-5, 5, n_points, dtype=jnp.complex64)
    u = jnp.linspace(-5, 5, n_points, dtype=jnp.complex64)

    # Generate a simple Gaussian with random width
    width = draw(st.floats(min_value=0.5, max_value=2.0))
    f = jnp.exp(-(x**2) / width).astype(jnp.complex64)

    return f, x, u


@given(lct_inputs(), sl2c_matrix(), sl2c_matrix())
def test_lct_composition(inputs, matrix_A: SL2C, matrix_B: SL2C):
    """Test that LCT(A)∘LCT(B) = LCT(A@B)"""
    f, x, u = inputs

    # Compute LCT(B) followed by LCT(A)
    intermediate = lct(f, x, u, matrix_B)
    composition = lct(intermediate, u, x, matrix_A)

    # Compute LCT(A@B) directly
    combined_matrix = matrix_A @ matrix_B
    direct = lct(f, x, u, combined_matrix)

    assert jnp.allclose(
        composition, direct, rtol=1e-4, atol=1e-4
    ), f"LCT composition property failed: max abs diff = {jnp.round(jnp.max(jnp.abs(composition - direct)), 3)}"


@given(lct_inputs(), sl2c_matrix())
def test_lct_inversion(inputs, matrix: SL2C):
    """Test that LCT(A)∘LCT(A^{-1}) = identity"""
    f, x, u = inputs

    # Compute inverse matrix
    inv_matrix = jnp.linalg.inv(matrix)

    # Apply LCT followed by its inverse
    forward = lct(f, x, u, matrix)
    reconstructed = lct(forward, u, x, inv_matrix)

    assert jnp.allclose(
        f, reconstructed, rtol=1e-4, atol=1e-4
    ), "LCT inversion property failed"


def test_special_matrices():
    """Test the special cases mentioned in the requirements"""
    n_points = 32
    x = jnp.linspace(-5, 5, n_points, dtype=jnp.complex64)
    u = jnp.linspace(-5, 5, n_points, dtype=jnp.complex64)
    f = jnp.exp(-(x**2)).astype(jnp.complex64)

    # Fourier matrix
    fourier = jnp.array([[0, 1], [-1, 0]], dtype=jnp.complex64)

    # Laplace matrix
    laplace = jnp.array([[0, 1j], [1j, 0]], dtype=jnp.complex64)

    # Test invertibility for both transforms
    for matrix in [fourier, laplace]:
        forward = lct(f, x, u, matrix)
        inverse = lct(forward, u, x, jnp.linalg.inv(matrix))
        assert jnp.allclose(
            f, inverse, rtol=1e-4, atol=1e-4
        ), f"Invertibility failed for matrix {matrix}"


@given(st.floats(min_value=-jnp.pi, max_value=jnp.pi))
def test_fractional_fourier(theta: float):
    """Test properties of fractional Fourier transform"""
    n_points = 32
    x = jnp.linspace(-5, 5, n_points, dtype=jnp.complex64)
    u = jnp.linspace(-5, 5, n_points, dtype=jnp.complex64)
    f = jnp.exp(-(x**2)).astype(jnp.complex64)

    frft_matrix = jnp.array(
        [[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]],
        dtype=jnp.complex64,
    )

    # Test that inverse angle gives inverse transform
    forward = lct(f, x, u, frft_matrix)
    inverse_matrix = jnp.array(
        [[jnp.cos(-theta), jnp.sin(-theta)], [-jnp.sin(-theta), jnp.cos(-theta)]],
        dtype=jnp.complex64,
    )
    reconstructed = lct(forward, u, x, inverse_matrix)

    assert jnp.allclose(
        f, reconstructed, rtol=1e-4, atol=1e-4
    ), "Fractional Fourier transform inversion failed"


def test_fourier_transform():
    n_points = 16
    x = jnp.linspace(-5, 5, n_points).astype(jnp.complex64)
    dx = x[1] - x[0]
    # Frequency grid matching JAX FFT convention
    u = 2 * jnp.pi * jnp.fft.fftfreq(n_points, dx).astype(jnp.complex64)
    
    f = jnp.exp(-(x**2)).astype(jnp.complex64)
    ft_result = fourier_transform(f, x, u)
    # Scale FFT to match continuous FT convention
    fft_result = jnp.fft.fft(f) * dx / jnp.sqrt(2 * jnp.pi)
    
    assert ft_result.shape == fft_result.shape, \
        f"Shape mismatch: ft_result {ft_result.shape}, fft_result {fft_result.shape}"
    assert jnp.allclose(ft_result, fft_result, rtol=1e-4, atol=1e-4), \
        "Fourier transform results do not match"

def test_inverse_fourier_transform():
    n_points = 16
    x = jnp.linspace(-5, 5, n_points).astype(jnp.complex64)
    dx = x[1] - x[0]
    u = 2 * jnp.pi * jnp.fft.fftfreq(n_points, dx).astype(jnp.complex64)
    
    f = jnp.exp(-(x**2)).astype(jnp.complex64)
    ift_result = inverse_fourier_transform(f, x, u)
    # Scale IFFT to match continuous IFT convention
    ifft_result = jnp.fft.ifft(f) * dx * jnp.sqrt(2 * jnp.pi) * n_points
    
    assert ift_result.shape == ifft_result.shape, \
        f"Shape mismatch: ift_result {ift_result.shape}, ifft_result {ifft_result.shape}"
    assert jnp.allclose(ift_result, ifft_result, rtol=1e-4, atol=1e-4), \
        "Inverse Fourier transform results do not match"

@given(
    x=st.lists(st.floats(min_value=-5, max_value=5), min_size=16, max_size=16),
    u=st.lists(st.floats(min_value=-5, max_value=5), min_size=16, max_size=16),
    a=st.complex_numbers(min_magnitude=0.1, max_magnitude=2.0, allow_infinity=False, allow_nan=False),
    b=st.complex_numbers(min_magnitude=0.1, max_magnitude=2.0, allow_infinity=False, allow_nan=False),
    c=st.complex_numbers(min_magnitude=0.1, max_magnitude=2.0, allow_infinity=False, allow_nan=False),
)
def test_invertible_transforms(x, u, a, b, c):
    # ad - bc = 1
    d = (1 + b * c) / a
    x = jnp.array(x).astype(jnp.complex64)
    dx = x[1] - x[0]
    u = 2 * jnp.pi * jnp.fft.fftfreq(len(x), dx).astype(jnp.complex64)
    f = jnp.exp(-(x**2)).astype(jnp.complex64)

    matrix = jnp.array([[a, b], [c, d]]).astype(jnp.complex64)
    inv_matrix = jnp.linalg.inv(matrix)

    # Test LCT invertibility
    forward = lct(f, x, u, matrix)
    inverse = lct(forward, u, x, inv_matrix)
    assert jnp.allclose(f, inverse, rtol=1e-4, atol=1e-4), \
        "LCT invertibility failed"

    # Test transforms against FFT with proper scaling
    for transform, reference in [
        (fourier_transform(f, x, u), jnp.fft.fft(f) * dx / jnp.sqrt(2 * jnp.pi)),
        (inverse_fourier_transform(f, x, u), jnp.fft.ifft(f) * dx * jnp.sqrt(2 * jnp.pi) * len(x)),
        (laplace_transform(f, x, u), jnp.fft.fft(f) * dx / jnp.sqrt(2 * jnp.pi))
    ]:
        assert transform.shape == reference.shape, \
            f"Shape mismatch: transform {transform.shape}, reference {reference.shape}"
        assert jnp.allclose(transform, reference, rtol=1e-4, atol=1e-4), \
            "Transform results do not match reference"
