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

Useful continuum special cases:

- `(a, b, c, d) = (0, 1, -1, 0)` gives the Fourier transform up to the
  continuum phase convention
- `(a, b, c, d) = (\cos \theta, \sin \theta, -\sin \theta, \cos \theta)` gives
  the fractional Fourier transform
- `(a, b, c, d) = (1, \lambda z, 0, 1)` gives a Fresnel-style propagation setup

The package also retains a historical shorthand `(a, b, c) = (0, 1, 0)` for
the exact unitary `torch.fft.fft(..., norm="ortho")` fast path. That shortcut is
useful for neural-network experiments, but it is not a symplectic canonical
matrix because `ad - bc != 1`. Use the `properties` helpers when checking
canonical determinant and composition behavior.

This repository uses a discrete approximation with two paths:

- A dense reference kernel for small transforms and validation
- A Bluestein / Chirp-Z style path for larger generic transforms

The dense path is intentionally the ground truth for the package. The fast path
is tested against it rather than treated as the specification.

## Finite-dimensional tradeoff

In finite dimensions, a discrete LCT cannot generally preserve every continuum
property at once. The practical tradeoff in this repo is between:

- energy preservation / unitarity
- exact composition under matrix multiplication of the canonical parameters

The API exposes this through both normalization and projection choices:

- `normalization="unitary"` favors energy preservation on the discrete grid
- `normalization="compositional"` uses the continuum-style amplitude convention
  and is useful as a separate empirical axis
- `unitary_projection=True` projects dense small-grid kernels to a unitary
  matrix, which strongly improves unitarity but can increase composition error
  for generic fractional transforms

For trainable layers, `unitary` is the default because it is usually the more
stable optimization regime. `compositional` is still useful when you care more
about finite-dimensional matrix behavior than strict energy preservation.

You can inspect the finite-grid behavior directly:

```python
from lct_activation import property_report

report = property_report(
    16,
    (0.8660254, 0.5, -0.5),
    (0.8660254, -0.5, 0.5),
    normalization="unitary",
    unitary_projection=True,
)

print(report.first_unitarity_error)
print(report.composition_error)
```
