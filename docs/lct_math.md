# LCT Math Notes

The Linear Canonical Transform is parameterized by a symplectic matrix

$$
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$

with the constraint `ad - bc = 1`.

For `b != 0`, the continuum transform can be written as

$$
\mathcal{L}_{a,b,c,d}[f](y)
=
\frac{1}{\sqrt{2 \pi i b}}
\int_{-\infty}^{\infty}
\exp \left(
\frac{i}{2 b}(a x^2 - 2 x y + d y^2)
\right)
f(x) \, dx.
$$

Useful special cases:

- `(a, b, c, d) = (0, 1, 0, 0)` gives the Fourier transform
- `(a, b, c, d) = (\cos \theta, \sin \theta, -\sin \theta, \cos \theta)` gives
  the fractional Fourier transform
- `(a, b, c, d) = (1, \lambda z, 0, 1)` gives a Fresnel-style propagation setup

This repository uses a discrete approximation with two paths:

- A dense reference kernel for small transforms and validation
- A Bluestein / Chirp-Z style path for larger generic transforms

The dense path is intentionally the ground truth for the package. The fast path
is tested against it rather than treated as the specification.
