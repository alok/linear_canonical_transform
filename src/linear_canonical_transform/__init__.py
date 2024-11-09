#! /usr/bin/env python3
# %%
from typing import Annotated
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float, Complex
import einops
import tyro

type SL2C = Complex[Array, "2 2"]

def lct_kernel(
    x: Complex[Array, "x"],
    u: Complex[Array, "u"],
    matrix: SL2C,
) -> Complex[Array, "x u"]:
    """
    Compute the kernel of the Linear Canonical Transform (LCT).
    """
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]
    assert jnp.isclose(jnp.linalg.det(matrix), 1), "Matrix must have determinant 1"
    prefactor = 1 / jnp.sqrt(2 * jnp.pi * 1j * b)
    exponent = (
        1j
        * (a * x[:, None] ** 2 - 2 * x[:, None] * u[None, :] + d * u[None, :] ** 2)
        / (2 * b)
    )
    return prefactor * jnp.exp(exponent)

def lct(
    f: Complex[Array, "x"],
    x: Complex[Array, "x"],
    u: Complex[Array, "u"],
    matrix: SL2C,
) -> Complex[Array, "u"]:
    """
    Compute the Linear Canonical Transform (LCT) of function f at points u.
    """
    K = lct_kernel(x, u, matrix)
    integrand = K * f[:, None]
    return jnp.sum(integrand * (x[1] - x[0]), axis=0)  # Trapezoidal rule approximation

def rotation_matrix(angle: float) -> SL2C:
    """
    Compute the rotation matrix for a given angle theta (in radians).
    """
    return jnp.array(
        [
            [jnp.cos(angle), jnp.sin(angle)],
            [-jnp.sin(angle), jnp.cos(angle)],
        ],
        dtype=jnp.complex64,
    )

def fractional_fourier_transform(
    f: Complex[Array, "x"],
    x: Complex[Array, "x"],
    u: Complex[Array, "u"],
    angle: float,
) -> Complex[Array, "u"]:
    """
    Compute the Fractional Fourier Transform (FrFT) using the LCT framework.
    """
    return lct(f, x, u, matrix=rotation_matrix(angle))

def inverse_fractional_fourier_transform(
    f: Complex[Array, "x"],
    x: Complex[Array, "x"],
    u: Complex[Array, "u"],
    angle: float,
) -> Complex[Array, "u"]:
    """
    Compute the Inverse Fractional Fourier Transform using the LCT framework.
    """
    return fractional_fourier_transform(f, x, u, -angle)

def fractional_laplace_transform(
    f: Complex[Array, "x"],
    x: Complex[Array, "x"],
    u: Complex[Array, "u"],
    angle: float,
) -> Complex[Array, "u"]:
    """
    Compute the Fractional Laplace Transform using the LCT framework.
    """
    return lct(f, x, u, matrix=1j * rotation_matrix(angle))

def fourier_transform(
    f: Complex[Array, "x"],
    x: Complex[Array, "x"],
    u: Complex[Array, "u"],
) -> Complex[Array, "u"]:
    return fractional_fourier_transform(f, x, u, jnp.pi / 2)

def inverse_fourier_transform(
    f: Complex[Array, "x"],
    x: Complex[Array, "x"],
    u: Complex[Array, "u"],
) -> Complex[Array, "u"]:
    return lct(f, x, u, rotation_matrix(-jnp.pi / 2))

def laplace_transform(
    f: Complex[Array, "x"],
    x: Complex[Array, "x"],
    u: Complex[Array, "u"],
) -> Complex[Array, "u"]:
    return fractional_laplace_transform(f, x, u, jnp.pi / 2)
# %%
