from typing import Tuple
import pytest
import jax
import jax.numpy as jnp
from hypothesis import given, strategies as st
from jaxtyping import Array, Complex
from linear_canonical_transform import (
    lct,
    SL2C,
    dft,
    idft,
    inverse_fourier_transform,
    fractional_fourier_transform,
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


@given(lct_inputs())
def test_fourier_transform(inputs):
    f, x, _ = inputs

    ft_result = dft(f)
    fft_result = jnp.fft.fft(f)

    assert jnp.allclose(
        ft_result, fft_result, rtol=1e-4, atol=1e-4
    ), f"Fourier transform results do not match, {jnp.round(jnp.abs(ft_result - fft_result).max(), 4)}"


@given(lct_inputs())
def test_inverse_fourier_transform(inputs):
    f, x, _ = inputs

    ift_result = idft(f)
    ifft_result = jnp.fft.ifft(f)

    assert jnp.allclose(
        ift_result, ifft_result, rtol=1e-4, atol=1e-4
    ), "Inverse Fourier transform results do not match"


def test_specific_dft_vs_fft():
    """Test DFT against JAX FFT for a specific non-uniform random signal"""
    # Generate random complex signal
    key = jax.random.PRNGKey(42)
    f = jax.random.normal(key, (64,), dtype=jnp.complex64) + 1j * jax.random.normal(
        key, (64,), dtype=jnp.complex64
    )

    # Compute both transforms
    dft_result = dft(f)
    fft_result = jnp.fft.fft(f)

    # Print results for comparison
    print("\nRandom signal:")
    print(f"First 5 values: {f[:5]}")
    print("\nDFT result:")
    print(f"First 5 values: {dft_result[:5]}")
    print("\nFFT result:")
    print(f"First 5 values: {fft_result[:5]}")
    print(f"\nMax absolute difference: {jnp.abs(dft_result - fft_result).max()}")

    assert jnp.allclose(
        dft_result, fft_result, rtol=1e-4, atol=1e-4
    ), "DFT and FFT results don't match for specific random signal"
