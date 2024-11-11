import jax.numpy as jnp
from jaxtyping import Array, Float, Complex, ArrayLike
from typing import Any
from jax import random


# Global tolerance for floating point comparisons
TOLERANCE = 1e-5


def is_power_of_2(n):
    """Check if number is power of 2 using bit count."""
    return (n & (n - 1) == 0) and n > 0


def centered_fft(x_centered):
    """
    Compute centered FFT to match DAFT convention.
    """
    N = x_centered.shape[0]
    assert is_power_of_2(N), f"Input length {N} must be power of 2"

    x_zeroed = jnp.fft.ifftshift(x_centered)
    # Match DAFT normalization: 1/sqrt(N)
    X_zeroed = jnp.fft.fft(x_zeroed) / jnp.sqrt(N)
    X_centered = jnp.fft.fftshift(X_zeroed)
    return X_centered


def daft_from_lct(x_centered, a, b, c, d):
    """
    DAFT implementation using LCT matrix parameters (a,b,c,d).
    """
    N = x_centered.shape[0]
    assert is_power_of_2(N), f"Input length {N} must be power of 2"
    print(f"input: {x_centered[:5] = }")
    # Handle b ≈ 0 case
    if jnp.abs(b) < TOLERANCE:
        n_centered = jnp.arange(-N // 2, N // 2)
        scaling = jnp.sqrt(d) * jnp.exp(1j * c * d / 2 * (n_centered / N) ** 2)
        return scaling * x_centered / jnp.sqrt(N)  # Changed normalization

    assert jnp.isclose(
        a * d - b * c, 1, atol=TOLERANCE
    ), f"LCT matrix must satisfy ad-bc=1, got {jnp.abs(a*d - b*c - 1)}"

    dt = 2 * jnp.pi / N
    p = (a / b) * dt**2
    q = (a / b) * dt**2
    s = jnp.sign(b)

    assert jnp.abs(s) == 1, f"For power-of-2 length, sign(b) must be ±1, got {s}"

    n_centered = jnp.arange(-N // 2, N // 2)
    m_centered = jnp.arange(-N // 2, N // 2)  # Using N instead of M

    # Pre-chirp
    chirp1 = jnp.exp(1j * q * (n_centered / N) ** 2)
    print(f"{chirp1[:5] = }")
    print(f"before chirp: {x_centered[:5] = }")
    x_chirped = x_centered * chirp1
    print(f"after chirp: {x_chirped[:5] = }")
    # DFT term
    dft_term = jnp.exp(-2j * jnp.pi * s * jnp.outer(m_centered, n_centered) / N)
    print(f"{dft_term[:5, :5] = }")
    # Post-chirp
    chirp2 = jnp.exp(1j * p * (m_centered / N) ** 2)
    print(f"{chirp2[:5] = }")
    # Changed normalization to 1/sqrt(N)
    prefactor = 1.0 / jnp.sqrt(N)
    X_centered = prefactor * chirp2 * (dft_term @ x_chirped)
    print(f"after chirp2: {X_centered[:5] = }")
    return X_centered


def daft_inverse_from_lct(X_centered, a, b, c, d):
    """
    Inverse DAFT computed by inverting the LCT matrix.
    """
    N = X_centered.shape[0]
    assert is_power_of_2(N), f"Input length {N} must be power of 2"
    assert (
        jnp.abs(a * d - b * c - 1) < TOLERANCE
    ), f"LCT matrix must satisfy ad-bc=1, got {a*d - b*c}"
    return daft_from_lct(X_centered, d, -b, -c, a)


def test_daft_special_cases():
    """Test DAFT against known special cases."""
    N = 64
    n_centered = jnp.arange(-N // 2, N // 2)

    signals = {
        "Gaussian": jnp.exp(-(n_centered**2) / (N / 8) ** 2),
        "Impulse": jnp.zeros(N).at[N // 4].set(1.0),
        "Two impulses": jnp.zeros(N).at[N // 4].set(1.0).at[3 * N // 4].set(1.0),
    }

    cases = [
        ("DFT", (0, 1, -1, 0)),
        ("IDFT", (0, -1, 1, 0)),
        ("Identity", (1, 0, 0, 1)),
        ("Scale", (2, 0, 0, 0.5)),
        (
            "Fractional",
            (
                jnp.cos(jnp.pi / 4),
                jnp.sin(jnp.pi / 4),
                -jnp.sin(jnp.pi / 4),
                jnp.cos(jnp.pi / 4),
            ),
        ),
    ]

    print("\nSpecial Cases Test:")
    for sig_name, x_centered in signals.items():
        print(f"\nTesting with {sig_name}:")
        for name, (a, b, c, d) in cases:
            if abs(b) < TOLERANCE:
                print(f"\n{name}: b≈0 case")
                continue

            X = daft_from_lct(x_centered, a, b, c, d)

            # Check energy conservation
            energy_in = jnp.sum(jnp.abs(x_centered) ** 2)
            energy_out = jnp.sum(jnp.abs(X) ** 2)
            error = jnp.abs(energy_out - energy_in)

            print(f"\n{name}:")
            print(f"Energy in:  {energy_in:.8f}")
            print(f"Energy out: {energy_out:.8f}")
            print(f"Error:      {error:.8f}")

            # For DFT case, compare with FFT
            if name == "DFT":
                X_fft = centered_fft(x_centered)
                max_diff = jnp.max(jnp.abs(X - X_fft))
                print(f"Max FFT difference: {max_diff:.8f}")

                if max_diff > TOLERANCE:
                    print("WARNING: FFT difference exceeds tolerance!")
                    print("First few values comparison:")
                    for i in range(min(5, N)):
                        print(
                            f"{i}: DAFT={X[i]:.6f}, FFT={X_fft[i]:.6f}, "
                            f"diff={abs(X[i]-X_fft[i]):.6f}"
                        )


from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np


def generate_random_sl2c(
    key: Any,
) -> tuple[
    Complex[Array, ""], Complex[Array, ""], Complex[Array, ""], Complex[Array, ""]
]:
    """
    Generate random SL(2,C) matrices by sampling a,b,c and computing d numerically stably.
    Returns a,b,c,d components ensuring ad-bc=1.

    Uses Hypothesis for property-based testing of the SL(2,C) constraints.
    Computes d = (1 + bc)/a in a numerically stable way.
    """
    # Generate random complex numbers for a,b,c
    key1, key2, key3 = random.split(key, 3)
    a = random.uniform(key1, dtype=jnp.float64) + 1j * random.uniform(
        key1, dtype=jnp.float64
    )
    b = random.normal(key2, dtype=jnp.float64) + 1j * random.normal(
        key2, dtype=jnp.float64
    )
    c = random.normal(key3, dtype=jnp.float64) + 1j * random.normal(
        key3, dtype=jnp.float64
    )

    # Ensure a is not too close to zero for numerical stability
    a = jnp.where(jnp.abs(a) < TOLERANCE, 1.0 + 0j, a)

    # Compute d numerically stably by avoiding division by small numbers
    # Use the fact that ad - bc = 1
    d = (1 + b * c) / a
    print((a * d - b * c).shape)
    # Verify SL(2,C) constraint numerically
    assert jnp.isclose(
        a * d - b * c, 1, atol=TOLERANCE
    ), f"Matrix not in SL(2,C): ad-bc={a*d - b*c}"
    print(f"{a = }\n{b = }\n{c = }\n{d = }")
    return a, b, c, d


def test_composition_property():
    """
    Test how well the composition property holds for discrete LCT.
    LCT(A)·LCT(B) should approximately equal LCT(A@B).
    """
    N = 64
    key = random.PRNGKey(0)
    for i in range(5):
        # Generate test signal
        n_centered = jnp.arange(-N // 2, N // 2)
        x_centered = jnp.exp(-(n_centered**2) / (N / 8) ** 2)

        # Generate random SL(2,C) matrices
        a1, b1, c1, d1 = generate_random_sl2c(key)
        key = random.split(key)[1]
        a2, b2, c2, d2 = generate_random_sl2c(key)

        print("\nComposition Property Test:")

        # First matrix
        A = jnp.array([[a1, b1], [c1, d1]])

        # Second matrix
        B = jnp.array([[a2, b2], [c2, d2]])
        print(f"{A = }\n{B = }")
        # Matrix product
        C = A @ B

        # Apply transforms separately
        X1 = daft_from_lct(x_centered, A[0, 0], A[0, 1], A[1, 0], A[1, 1])
        X2 = daft_from_lct(X1, B[0, 0], B[0, 1], B[1, 0], B[1, 1])
        print(X1[:5], X2[:5])
        # Apply combined transform
        X_combined = daft_from_lct(x_centered, C[0, 0], C[0, 1], C[1, 0], C[1, 1])

        # Measure difference
        error = jnp.max(jnp.abs(X2 - X_combined))
        relative_error = error / jnp.max(jnp.abs(X_combined))

        print(f"\nTest {i+1}:")
        print(f"Max absolute error: {error:.8f}")
        print(f"Max relative error: {relative_error:.8f}")

        # Energy conservation check
        energy_in = jnp.sum(jnp.abs(x_centered) ** 2)
        energy_out1 = jnp.sum(jnp.abs(X2) ** 2)
        energy_out2 = jnp.sum(jnp.abs(X_combined) ** 2)

        print(f"Energy conservation errors:")
        print(f"Separate transforms: {abs(energy_out1 - energy_in):.8f}")
        print(f"Combined transform: {abs(energy_out2 - energy_in):.8f}")


def test_special_matrices():
    """Test standard special cases: Fourier, fractional Fourier."""
    N = 64
    n_centered = jnp.arange(-N // 2, N // 2)
    x_centered = jnp.exp(-(n_centered**2) / (N / 8) ** 2)
    print(f"orig input: {x_centered[:5] = }")
    # Test matrices
    theta = jnp.pi / 4  # 45 degrees
    matrices = {
        "Fourier": (0, 1, -1, 0),
        "Fractional Fourier": (
            jnp.cos(theta),
            jnp.sin(theta),
            -jnp.sin(theta),
            jnp.cos(theta),
        ),
        "Inverse Fractional Fourier": (
            jnp.cos(-theta),
            jnp.sin(-theta),
            -jnp.sin(-theta),
            jnp.cos(-theta),
        ),
    }

    print("\nSpecial Matrices Test:")
    for name, (a, b, c, d) in matrices.items():
        print(f"\nTesting {name} transform:")

        # Forward transform
        X = daft_from_lct(x_centered, a, b, c, d)
        print(f"X: {X[:5] = }")
        # For Fourier case, compare with FFT
        if name == "Fourier":
            X_fft = centered_fft(x_centered)
            fft_error = jnp.max(jnp.abs(X - X_fft))
            print(f"Max difference from FFT: {fft_error:.8f}")

        # Test inverse
        x_rec = daft_inverse_from_lct(X, a, b, c, d)
        rec_error = jnp.max(jnp.abs(x_centered - x_rec))
        print(f"Max reconstruction error: {rec_error:.8f}")

        # Energy conservation
        energy_in = jnp.sum(jnp.abs(x_centered) ** 2)
        energy_out = jnp.sum(jnp.abs(X) ** 2)
        energy_error = jnp.abs(energy_out - energy_in)
        print(f"Energy conservation error: {energy_error:.8f}")


if __name__ == "__main__":
    test_daft_special_cases()
    test_special_matrices()
    test_composition_property()
