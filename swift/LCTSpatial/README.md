# LCTSpatial

`LCTSpatial` is a Swift/Metal toolkit for spatial intuition about finite-grid
linear canonical transforms.

The package deliberately exposes two different operations:

- `DiscreteLCT` / `MetalLCTTransformer` transform a sampled complex field. This
  is the Atlas-style signal transform.
- `CanonicalPairTransform` applies the `2×2` canonical map to paired vectors or
  mesh attributes. It is useful as a zero-latency preview, but is not presented
  as the integral transform of a mesh.

The scalar canonical block is modeled over `SL(2,C)`. A `ComplexField` may have
any rank; the same block is applied separably along every axis. The visionOS
sample uses rank three.

```swift
import LCTSpatial

var lock = SL2CConstraint(matrix: .identity, dependent: .d)
let matrix = try lock.set(.b, to: Complex32(real: 0.8, imaginary: 0.15))

let field = try ComplexField(
    shape: [16, 16, 16],
    values: Array(repeating: .zero, count: 16 * 16 * 16)
)
let transformed = try DiscreteLCT.transform(field, matrix: matrix)
```

`SL2CConstraint` keeps `ad - bc = 1` during interactive scrubbing. It prefers
to solve for `d`, then automatically changes the dependent coefficient if the
current denominator approaches a singular pivot.

## Mathematical boundary

Classical unitary LCTs use real symplectic parameters. Complex matrices are a
useful extension, but determinant one alone does not guarantee a bounded or
unitary transform. The CPU and Metal kernels clamp the real part of the complex
exponent for interactive safety. Apps should expose that stability limit rather
than treating every point of `SL(2,C)` as physically interchangeable.
