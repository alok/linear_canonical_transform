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
    FFT,
)
import numpy as np


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
    fft_result = jnp.fft.fft(f, norm='ortho')

    assert jnp.allclose(
        ft_result, fft_result, rtol=1e-4, atol=1e-4
    ), "Fourier transform results do not match"


@given(lct_inputs())
def test_inverse_fourier_transform(inputs):
    f, x, _ = inputs

    ift_result = idft(f)
    ifft_result = jnp.fft.ifft(f, norm='ortho')

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
    fft_result = jnp.fft.fft(f, norm='ortho')

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


def test_fft_against_numpy():
    """Test our FFT implementation against NumPy's FFT for various inputs"""
    
    # Test case 1: Simple sinusoid
    N = 64
    t = np.linspace(0, 1, N)
    f = np.sin(2 * np.pi * 10 * t) + 1j * np.cos(2 * np.pi * 20 * t)
    f = f.astype(np.complex64)
    
    # Compare results
    our_fft = FFT(f)
    np_fft = np.fft.fft(f, norm='ortho')
    
    assert np.allclose(our_fft, np_fft, rtol=1e-5, atol=1e-5), \
        "FFT doesn't match NumPy for sinusoid"
        
    # Test case 2: Gaussian pulse
    x = np.linspace(-2, 2, N)
    f = np.exp(-x**2).astype(np.complex64)
    
    our_fft = FFT(f)
    np_fft = np.fft.fft(f, norm='ortho')
    
    assert np.allclose(our_fft, np_fft, rtol=1e-5, atol=1e-5), \
        "FFT doesn't match NumPy for Gaussian"
    
    # Test case 3: Random complex signal
    key = jax.random.PRNGKey(0)
    f = (jax.random.normal(key, (N,)) + 
         1j * jax.random.normal(key, (N,))).astype(np.complex64)
    
    our_fft = FFT(f)
    np_fft = np.fft.fft(f, norm='ortho')
    
    assert np.allclose(our_fft, np_fft, rtol=1e-5, atol=1e-5), \
        "FFT doesn't match NumPy for random signal"

def test_fft_properties():
    """Test mathematical properties of our FFT implementation"""
    N = 64
    key = jax.random.PRNGKey(1)
    f = (jax.random.normal(key, (N,)) + 
         1j * jax.random.normal(key, (N,))).astype(np.complex64)
    
    # Test linearity
    alpha = 2.0 + 3.0j
    beta = -1.0 + 2.0j
    g = (jax.random.normal(key, (N,)) + 
         1j * jax.random.normal(key, (N,))).astype(np.complex64)
    
    sum_of_ffts = alpha * FFT(f) + beta * FFT(g)
    fft_of_sum = FFT(alpha * f + beta * g)
    
    assert np.allclose(sum_of_ffts, fft_of_sum, rtol=1e-5, atol=1e-5), \
        "FFT linearity property failed"
    
    # Test Parseval's theorem
    # Energy in time domain should equal energy in frequency domain
    energy_time = np.sum(np.abs(f)**2)
    energy_freq = np.sum(np.abs(FFT(f))**2)
    
    assert np.allclose(energy_time, energy_freq, rtol=1e-5, atol=1e-5), \
        "FFT Parseval's theorem failed"
    
    # Test shift theorem
    # Circular shift in time domain = phase shift in frequency domain
    shift = 3
    shifted_f = np.roll(f, shift)
    k = np.fft.fftfreq(N) * N
    expected_phase = np.exp(-2j * np.pi * k * shift / N)
    
    fft_shifted = FFT(shifted_f)
    fft_original = FFT(f)
    
    phase_ratio = fft_shifted / fft_original
    phase_ratio = phase_ratio[np.abs(fft_original) > 1e-10]  # Avoid division by near-zero
    expected_phase = expected_phase[np.abs(fft_original) > 1e-10]
    
    assert np.allclose(np.abs(phase_ratio), np.ones_like(phase_ratio), rtol=1e-5, atol=1e-5), \
        "FFT shift theorem magnitude failed"
    assert np.allclose(phase_ratio / expected_phase, np.ones_like(phase_ratio), rtol=1e-5, atol=1e-5), \
        "FFT shift theorem phase failed"
