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
    # Create DFT matrix
    dft_matrix = jnp.exp(-2j * jnp.pi * k * m / n)
    # Compute transform
    return jnp.dot(dft_matrix, f)


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
    # Create IDFT matrix
    idft_matrix = jnp.exp(2j * jnp.pi * k * m / n)
    # Compute inverse transform with 1/n normalization
    return jnp.dot(idft_matrix, f) / n




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
