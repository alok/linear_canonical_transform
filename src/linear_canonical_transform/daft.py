# from typing import Any
# import jax.numpy as jnp
# from jaxtyping import Array, Float, Complex, ArrayLike

# # Constants
# TOLERANCE: float = 1e-6


# def is_coprime(a: ArrayLike, b: ArrayLike) -> bool:
#     """Check if two numbers are coprime."""
#     return jnp.gcd(a, b) == 1


# def is_power_of_2(n: int) -> bool:
#     """Check if number is power of 2 using bit count."""
#     return (n & (n - 1) == 0) and n > 0


# def centered_fft(x_centered: Complex[Array, "N"]) -> Complex[Array, "N"]:
#     """
#     Compute centered FFT to match DAFT convention.

#     Args:
#         x_centered: Input signal indexed [-N/2, N/2-1], N must be power of 2
#     Returns:
#         X_centered: Transform indexed [-N/2, N/2-1]
#     """
#     N = x_centered.shape[0]
#     assert is_power_of_2(N), f"Input length {N} must be power of 2"

#     x_zeroed = jnp.fft.ifftshift(x_centered)
#     X_zeroed = jnp.fft.fft(x_zeroed, norm="ortho")
#     X_centered = jnp.fft.fftshift(X_zeroed)
#     return X_centered


# def daft_from_lct(
#     x_centered: Complex[Array, "N"],
#     a: Float[Array, ""],
#     b: Float[Array, ""],
#     c: Float[Array, ""],
#     d: Float[Array, ""],
# ) -> Complex[Array, "N"]:
#     """
#     DAFT implementation using LCT matrix parameters (a,b,c,d).
#     The LCT parameters are converted to DAFT type 2 parameters as:
#     p = a/b·Δt², q = a/b·Δt², s = sgn(b)

#     Args:
#         x_centered: Input signal array indexed [-N/2, N/2-1], N must be power of 2
#         a,b,c,d: LCT matrix parameters satisfying ad-bc=1

#     Returns:
#         X_centered: DAFT of input signal, indexed [-N/2, N/2-1]
#     """
#     N = x_centered.shape[0]
#     assert is_power_of_2(N), f"Input length {N} must be power of 2"

#     M = N  # Using M=N since N is power of 2

#     # Handle b ≈ 0 case
#     if jnp.abs(b) < TOLERANCE:
#         # When b=0, the LCT is just a scaling and chirp multiplication
#         n_centered = jnp.arange(-N // 2, N // 2)
#         scaling = jnp.sqrt(d) * jnp.exp(1j * c * d / 2 * (n_centered / N) ** 2)
#         return scaling * x_centered / jnp.sqrt(2 * M + 1)

#     # Verify LCT matrix determinant
#     assert jnp.abs(a * d - b * c - 1) < TOLERANCE, f"LCT matrix must satisfy ad-bc=1"

#     # Convert LCT parameters to DAFT type 2 parameters
#     dt = 2 * jnp.pi / N  # Sampling interval
#     p = (a / b) * dt**2
#     q = (a / b) * dt**2
#     s = jnp.sign(b)  # Must be ±1 since N is power of 2

#     # For power of 2 lengths, s must be ±1 to be coprime
#     assert jnp.abs(s) == 1, f"For power-of-2 length, sign(b) must be ±1, got {s}"

#     # Create centered grids
#     n_centered = jnp.arange(-N // 2, N // 2)  # Input indices
#     m_centered = jnp.arange(-M // 2, M // 2)  # Output indices

#     # Pre-chirp multiplication (operates on centered input)
#     chirp1 = jnp.exp(1j * q * (n_centered / N) ** 2)
#     x_chirped = x_centered * chirp1

#     # DFT term with parameter s (note: indices remain centered)
#     dft_term = jnp.exp(
#         -2j * jnp.pi * s * jnp.outer(m_centered, n_centered) / (2 * M + 1)
#     )

#     # Post-chirp multiplication (operates on centered output)
#     chirp2 = jnp.exp(1j * p * (m_centered / M) ** 2)

#     # Combine all terms with normalization
#     prefactor = 1.0 / jnp.sqrt(2 * M + 1)
#     X_centered = prefactor * chirp2 * (dft_term @ x_chirped)

#     return X_centered


# def daft_inverse_from_lct(
#     X_centered: Complex[Array, "N"],
#     a: Float[Array, ""],
#     b: Float[Array, ""],
#     c: Float[Array, ""],
#     d: Float[Array, ""],
# ) -> Complex[Array, "N"]:
#     """
#     Inverse DAFT computed by inverting the LCT matrix.

#     Args:
#         X_centered: Input transform indexed [-N/2, N/2-1], N must be power of 2
#         a,b,c,d: LCT matrix parameters satisfying ad-bc=1
#     Returns:
#         x_centered: Inverse transform indexed [-N/2, N/2-1]
#     """
#     N = X_centered.shape[0]
#     assert is_power_of_2(N), f"Input length {N} must be power of 2"

#     # Construct LCT matrix and invert it
#     E = jnp.array([[a, b], [c, d]])
#     E_ = jnp.linalg.inv(E)

#     # Extract inverse matrix elements
#     a_, b_ = E_[0, 0], E_[0, 1]
#     c_, d_ = E_[1, 0], E_[1, 1]

#     return daft_from_lct(X_centered, a_, b_, c_, d_)


# # Example usage:
# if __name__ == "__main__":
#     # Create test signal (centered) with power-of-2 length
#     N = 64  # Must be power of 2
#     assert is_power_of_2(N), f"Example length {N} must be power of 2"

#     n_centered = jnp.arange(-N // 2, N // 2)
#     x_centered = jnp.exp(-(n_centered**2) / (N / 8) ** 2) + 0.5 * jnp.sin(2 * jnp.pi * n_centered / (N/4))  # Gaussian + sinusoid

#     # Example LCT parameters (a,b,c,d) satisfying ad-bc=1
#     alpha = jnp.pi / 4  # 45 degree rotation
#     a = d = jnp.cos(alpha)
#     b = jnp.sin(alpha)  # Will give s = sign(b) = ±1
#     c = -jnp.sin(alpha)
#     assert jnp.abs(a * d - b * c - 1) < 1e-7, "Example LCT matrix invalid"

#     # Forward transform
#     X_centered = daft_from_lct(x_centered, a, b, c, d)

#     # Inverse transform
#     x_rec_centered = daft_inverse_from_lct(X_centered, a, b, c, d)

#     # Verify reconstruction
#     print("Max reconstruction error:", jnp.max(jnp.abs(x_centered - x_rec_centered)))

#     # Verify energy conservation
#     print(
#         "Energy conservation error:",
#         jnp.abs(jnp.sum(jnp.abs(X_centered) ** 2) - jnp.sum(jnp.abs(x_centered) ** 2)),
#     )

#     # Compare with centered FFT for the case of regular DFT
#     X_daft = daft_from_lct(x_centered, 0, 1, -1, 0)  # Should match FFT
#     X_fft = centered_fft(x_centered)
#     print("x_centered:", x_centered[:5], x_centered[-5:])
#     print("X_daft:", X_daft[:5], X_daft[-5:])
#     print("X_fft:", X_fft[:5], X_fft[-5:])
#     print("Max difference from centered FFT:", jnp.max(jnp.abs(X_daft - X_fft)))
# import jax.numpy as jnp

# # Global tolerance for floating point comparisons
# TOLERANCE = 1e-6

# def is_power_of_2(n):
#     """Check if number is power of 2 using bit count."""
#     return (n & (n - 1) == 0) and n > 0

# def centered_fft(x_centered):
#     """
#     Compute centered FFT to match DAFT convention when checking DFT case.
#     Uses same normalization as DAFT for comparison purposes.
#     """
#     N = x_centered.shape[0]
#     assert is_power_of_2(N), f"Input length {N} must be power of 2"
    
#     x_zeroed = jnp.fft.ifftshift(x_centered)
#     # Use same normalization as DAFT for DFT case
#     X_zeroed = jnp.fft.fft(x_zeroed) / jnp.sqrt(2*N + 1)  
#     X_centered = jnp.fft.fftshift(X_zeroed)
#     return X_centered

# def daft_from_lct(x_centered, a, b, c, d):
#     """
#     DAFT implementation using LCT matrix parameters (a,b,c,d).
#     The LCT parameters are converted to DAFT type 2 parameters as:
#     p = a/b·Δt², q = a/b·Δt², s = sgn(b)
    
#     Args:
#         x_centered: Input signal array indexed [-N/2, N/2-1], N must be power of 2
#         a,b,c,d: LCT matrix parameters satisfying ad-bc=1
        
#     Returns:
#         X_centered: DAFT of input signal, indexed [-N/2, N/2-1]
#     """
#     N = x_centered.shape[0]
#     assert is_power_of_2(N), f"Input length {N} must be power of 2"
    
#     M = N  # Using M=N since N is power of 2

#     # Handle b ≈ 0 case
#     if jnp.abs(b) < TOLERANCE:
#         n_centered = jnp.arange(-N//2, N//2)
#         scaling = jnp.sqrt(d) * jnp.exp(1j * c * d/2 * (n_centered/N)**2)
#         return scaling * x_centered / jnp.sqrt(2*M + 1)

#     assert jnp.abs(a*d - b*c - 1) < TOLERANCE, f"LCT matrix must satisfy ad-bc=1"

#     dt = 2*jnp.pi/N
#     p = (a/b) * dt**2  
#     q = (a/b) * dt**2
#     s = jnp.sign(b)
    
#     assert jnp.abs(s) == 1, f"For power-of-2 length, sign(b) must be ±1, got {s}"

#     n_centered = jnp.arange(-N//2, N//2)
#     m_centered = jnp.arange(-M//2, M//2)
    
#     # Pre-chirp
#     chirp1 = jnp.exp(1j * q * (n_centered/N)**2)
#     x_chirped = x_centered * chirp1
    
#     # DFT term
#     dft_term = jnp.exp(-2j * jnp.pi * s * jnp.outer(m_centered, n_centered) / (2*M + 1))
    
#     # Post-chirp
#     chirp2 = jnp.exp(1j * p * (m_centered/M)**2)
    
#     # Combine with normalization as per paper equation (63)
#     prefactor = 1.0 / jnp.sqrt(2*M + 1)
#     X_centered = prefactor * chirp2 * (dft_term @ x_chirped)
    
#     return X_centered

# def daft_inverse_from_lct(X_centered, a, b, c, d):
#     """
#     Inverse DAFT computed by inverting the LCT matrix.
    
#     Args:
#         X_centered: Input transform indexed [-N/2, N/2-1], N must be power of 2
#         a,b,c,d: LCT matrix parameters satisfying ad-bc=1
#     Returns:
#         x_centered: Inverse transform indexed [-N/2, N/2-1]
#     """
#     N = X_centered.shape[0]
#     assert is_power_of_2(N), f"Input length {N} must be power of 2"
#     assert jnp.abs(a*d - b*c - 1) < TOLERANCE, f"LCT matrix must satisfy ad-bc=1"
#     return daft_from_lct(X_centered, d, -b, -c, a)

# def test_daft_special_cases():
#     """Test DAFT against known special cases."""
#     N = 64
#     n_centered = jnp.arange(-N//2, N//2)
    
#     # Test cases with different input signals
#     signals = {
#         "Gaussian": jnp.exp(-(n_centered**2)/(N/8)**2),
#         "Impulse": jnp.zeros(N).at[N//4].set(1.0),
#         "Two impulses": jnp.zeros(N).at[N//4].set(1.0).at[3*N//4].set(1.0)
#     }
    
#     # LCT parameter test cases
#     cases = [
#         ("DFT", (0, 1, -1, 0)),  # Regular DFT
#         ("IDFT", (0, -1, 1, 0)),  # Inverse DFT
#         ("Identity", (1, 0, 0, 1)),  # Identity transform
#         ("Scale", (2, 0, 0, 0.5)),  # Scaling by 2
#         ("Fractional", (jnp.cos(jnp.pi/4), jnp.sin(jnp.pi/4), 
#                        -jnp.sin(jnp.pi/4), jnp.cos(jnp.pi/4)))  # 45° rotation
#     ]
    
#     print("\nSpecial Cases Test:")
#     for sig_name, x_centered in signals.items():
#         print(f"\nTesting with {sig_name}:")
#         for name, (a, b, c, d) in cases:
#             if abs(b) < TOLERANCE:
#                 print(f"\n{name}: b≈0 case")
#                 continue
                
#             X = daft_from_lct(x_centered, a, b, c, d)
            
#             # Check energy conservation
#             energy_in = jnp.sum(jnp.abs(x_centered)**2)
#             energy_out = jnp.sum(jnp.abs(X)**2)
#             error = jnp.abs(energy_out - energy_in)
            
#             print(f"\n{name}:")
#             print(f"Energy in:  {energy_in:.8f}")
#             print(f"Energy out: {energy_out:.8f}")
#             print(f"Error:      {error:.8f}")
            
#             # For DFT case, compare with FFT
#             if name == "DFT":
#                 X_fft = centered_fft(x_centered)
#                 max_diff = jnp.max(jnp.abs(X - X_fft))
#                 print(f"Max FFT difference: {max_diff:.8f}")
                
#                 if max_diff > TOLERANCE:
#                     print("WARNING: FFT difference exceeds tolerance!")
#                     # Print some diagnostic info
#                     print("First few values comparison:")
#                     for i in range(min(5, N)):
#                         print(f"{i}: DAFT={X[i]:.6f}, FFT={X_fft[i]:.6f}, "
#                               f"diff={abs(X[i]-X_fft[i]):.6f}")

# if __name__ == "__main__":
#     test_daft_special_cases()

import jax.numpy as jnp

# Global tolerance for floating point comparisons
TOLERANCE = 1e-6

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
    
    # Handle b ≈ 0 case
    if jnp.abs(b) < TOLERANCE:
        n_centered = jnp.arange(-N//2, N//2)
        scaling = jnp.sqrt(d) * jnp.exp(1j * c * d/2 * (n_centered/N)**2)
        return scaling * x_centered / jnp.sqrt(N)  # Changed normalization

    assert jnp.abs(a*d - b*c - 1) < TOLERANCE, f"LCT matrix must satisfy ad-bc=1"

    dt = 2*jnp.pi/N
    p = (a/b) * dt**2  
    q = (a/b) * dt**2
    s = jnp.sign(b)
    
    assert jnp.abs(s) == 1, f"For power-of-2 length, sign(b) must be ±1, got {s}"

    n_centered = jnp.arange(-N//2, N//2)
    m_centered = jnp.arange(-N//2, N//2)  # Using N instead of M
    
    # Pre-chirp
    chirp1 = jnp.exp(1j * q * (n_centered/N)**2)
    x_chirped = x_centered * chirp1
    
    # DFT term
    dft_term = jnp.exp(-2j * jnp.pi * s * jnp.outer(m_centered, n_centered) / N)
    
    # Post-chirp
    chirp2 = jnp.exp(1j * p * (m_centered/N)**2)
    
    # Changed normalization to 1/sqrt(N)
    prefactor = 1.0 / jnp.sqrt(N)
    X_centered = prefactor * chirp2 * (dft_term @ x_chirped)
    
    return X_centered

def daft_inverse_from_lct(X_centered, a, b, c, d):
    """
    Inverse DAFT computed by inverting the LCT matrix.
    """
    N = X_centered.shape[0]
    assert is_power_of_2(N), f"Input length {N} must be power of 2"
    assert jnp.abs(a*d - b*c - 1) < TOLERANCE, f"LCT matrix must satisfy ad-bc=1"
    return daft_from_lct(X_centered, d, -b, -c, a)

def test_daft_special_cases():
    """Test DAFT against known special cases."""
    N = 64
    n_centered = jnp.arange(-N//2, N//2)
    
    signals = {
        "Gaussian": jnp.exp(-(n_centered**2)/(N/8)**2),
        "Impulse": jnp.zeros(N).at[N//4].set(1.0),
        "Two impulses": jnp.zeros(N).at[N//4].set(1.0).at[3*N//4].set(1.0)
    }
    
    cases = [
        ("DFT", (0, 1, -1, 0)),
        ("IDFT", (0, -1, 1, 0)),
        ("Identity", (1, 0, 0, 1)),
        ("Scale", (2, 0, 0, 0.5)),
        ("Fractional", (jnp.cos(jnp.pi/4), jnp.sin(jnp.pi/4), 
                       -jnp.sin(jnp.pi/4), jnp.cos(jnp.pi/4)))
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
            energy_in = jnp.sum(jnp.abs(x_centered)**2)
            energy_out = jnp.sum(jnp.abs(X)**2)
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
                        print(f"{i}: DAFT={X[i]:.6f}, FFT={X_fft[i]:.6f}, "
                              f"diff={abs(X[i]-X_fft[i]):.6f}")

if __name__ == "__main__":
    test_daft_special_cases()