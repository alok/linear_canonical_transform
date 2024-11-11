#! /usr/bin/env python3
# %%
from typing import TypeAlias, Callable
from functools import partial
import jax.numpy as jnp
from jaxtyping import Array, Complex

# Type aliases
SL2C: TypeAlias = Complex[Array, "2 2"]  # 2x2 complex matrix with determinant 1
Signal: TypeAlias = Complex[Array, "N"]  # Complex signal array
Grid: TypeAlias = Complex[Array, "N"]  # Complex grid points
TransformedSignal: TypeAlias = Callable[
    [Grid], Signal
]  # Function mapping grid to signal

def dft(f: Signal) -> Signal:
    """
    Compute the discrete Fourier transform from scratch.
    Following numpy.fft convention:
    A_k = sum_{m=0}^{n-1} a_m exp(-2πi*mk/n)
    
    Args:
        f: Input signal
        
    Returns:
        Fourier transformed signal
    """
    n = len(f)
    k = jnp.arange(n)
    m = k.reshape(-1, 1)
    # Create DFT matrix with scaling
    dft_matrix = jnp.exp(-2j * jnp.pi * k * m / n)
    # Compute transform
    return jnp.dot(dft_matrix, f) / jnp.sqrt(n)


def idft(f: Signal) -> Signal:
    """
    Compute the inverse discrete Fourier transform from scratch.
    Following numpy.fft convention:
    a_m = (1/n) sum_{k=0}^{n-1} A_k exp(2πi*mk/n)
    
    Args:
        f: Input signal
        
    Returns:
        Inverse Fourier transformed signal
    """
    n = len(f)
    k = jnp.arange(n)
    m = k.reshape(-1, 1)
    # Create IDFT matrix with scaling
    idft_matrix = jnp.exp(2j * jnp.pi * k * m / n)
    # Compute inverse transform with 1/n normalization
    return jnp.dot(idft_matrix, f) / jnp.sqrt(n)

import jax.numpy as jnp
from jax import jit
from functools import partial

def make_lct(a, b, c, d):
    """Create a Linear Canonical Transform function with fixed parameters.
    
    Args:
        a, b, c, d: LCT parameters satisfying ad-bc=1
        
    Returns:
        Function that takes a signal array and returns its LCT
    """
    # @jit
    def lct(f):
        N = f.shape[0]
        norm_factor = 1/jnp.sqrt(N)  # For 'ortho' normalization
        
        if jnp.abs(b) < 1e-10:  # b ≈ 0 case
            # For b≈0, LCT is just a scaling and chirp multiplication
            n = jnp.arange(N)
            scaling = jnp.sqrt(d) * jnp.exp(1j * c * d/2 * (n/N)**2)
            return norm_factor * scaling * f
            
        # General case
        n = jnp.arange(N)
        
        # Pre-chirp multiplication
        pre_chirp = jnp.exp(1j * jnp.pi * a/(2*b) * (n/N)**2)
        f_chirped = f * pre_chirp
        
        # FFT (using ortho normalization)
        F = jnp.fft.fft(f_chirped) / jnp.sqrt(N)
        
        # Post-chirp multiplication
        m = jnp.arange(N)
        post_chirp = jnp.exp(1j * jnp.pi * d/(2*b) * (m/N)**2)
        
        # Combine with normalization factor
        phase_factor = jnp.sqrt(1/(1j*2*jnp.pi*b)) 
        result = phase_factor * post_chirp * F
        
        return result

    return lct
def test_numpy_fft_normalization():
    """Test that numpy.fft with ortho normalization matches manual normalization"""
    # Generate sample signal
    N = 64
    t = jnp.linspace(0, 1, N)
    f = jnp.sin(2 * jnp.pi * 10 * t) + 1j * jnp.cos(2 * jnp.pi * 20 * t)
    f = f.astype(jnp.complex64)

    # Compute FFTs with different normalizations
    fft_ortho = jnp.fft.fft(f, norm='ortho')
    fft_manual = jnp.fft.fft(f) / jnp.sqrt(N)

    # Verify they match
    assert jnp.allclose(fft_ortho, fft_manual, rtol=1e-5, atol=1e-5), \
        "FFT ortho normalization doesn't match manual normalization"

    # Print comparison for first few values
    print("\nFFT with ortho first 5 values:")
    print(fft_ortho[:5])
    print("\nFFT with manual normalization first 5 values:")
    print(fft_manual[:5])
    print(f"\nMax absolute difference: {jnp.abs(fft_ortho - fft_manual).max()}")

# Run test
test_numpy_fft_normalization()


FFT = make_lct(0, 1, -1, 0)
def test_fft_implementation():
    """Compare our FFT implementation with numpy's FFT for a sample signal"""
    # Generate sample signal
    N = 64
    t = jnp.linspace(0, 1, N)
    f = jnp.sin(2 * jnp.pi * 10 * t) + 1j * jnp.cos(2 * jnp.pi * 20 * t)
    f = f.astype(jnp.complex64)

    # Compute FFTs
    our_fft = FFT(f)
    np_fft = jnp.fft.fft(f, norm='ortho')

    # Print comparison
    print("\nSample signal first 5 values:")
    print(f[:5])
    print("\nOur FFT first 5 values:")
    print(our_fft[:5])
    print("\nNumPy FFT first 5 values:")
    print(np_fft[:5])
    print(f"\nMax absolute difference: {jnp.abs(our_fft - np_fft).max()}")

    # Verify they match
    assert jnp.allclose(our_fft, np_fft, rtol=1e-5, atol=1e-5), \
        "FFT implementation doesn't match numpy"

# Run test
test_fft_implementation()



# Example usage:
# frft = make_lct(cos(alpha), sin(alpha), -sin(alpha), cos(alpha))  # Fractional Fourier Transform
# fft = make_lct(0, 1, -1, 0)  # Regular Fourier Transform


def lct(f: Signal, x: Grid, matrix: SL2C) -> TransformedSignal:
    """
    Compute the Linear Canonical Transform of signal f with respect to the given SL(2,C) matrix.
    Returns a function that maps output grid points to transformed signal values.

    Args:
        f: Input signal
        x: Input domain grid points
        matrix: SL(2,C) matrix [[a,b],[c,d]] specifying the transform

    Returns:
        Function that maps output grid points to the LCT-transformed signal
    """
    

def fourier_transform(f: Signal, x: Grid) -> TransformedSignal:
    """
    Compute the Fourier transform using LCT with matrix [[0,1],[-1,0]]
    Returns a function mapping frequency points to transformed values.
    """
    matrix = jnp.array([[0, 1], [-1, 0]], dtype=jnp.complex64)
    return lct(f, x, matrix)


def inverse_fourier_transform(f: Signal, x: Grid) -> TransformedSignal:
    """
    Compute the inverse Fourier transform using LCT with matrix [[0,-1],[1,0]]
    Returns a function mapping time points to transformed values.
    """
    matrix = jnp.array([[0, -1], [1, 0]], dtype=jnp.complex64)
    return lct(f, x, matrix)


def fractional_fourier_transform(f: Signal, x: Grid, theta: float) -> TransformedSignal:
    """
    Compute the fractional Fourier transform for angle theta
    Returns a function mapping output points to transformed values.

    Args:
        f: Input signal
        x: Input domain grid points
        theta: Rotation angle in radians

    Returns:
        Function that maps output points to the fractionally Fourier transformed signal
    """
    matrix = jnp.array(
        [[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]],
        dtype=jnp.complex64,
    )
    return lct(f, x, matrix)


# %%
