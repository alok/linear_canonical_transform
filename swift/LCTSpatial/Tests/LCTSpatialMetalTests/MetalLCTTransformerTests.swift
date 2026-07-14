import LCTSpatial
import Metal
import Testing

@testable import LCTSpatialMetal

@Test func metalMatchesCPUReferenceAcrossThreeAxes() throws {
  guard let device = MTLCreateSystemDefaultDevice() else { return }
  let values = (0..<27).map { index in
    Complex32(real: Float(index % 5) / 5, imaginary: Float(index % 3) / 7)
  }
  let field = try ComplexField(shape: [3, 3, 3], values: values)
  let matrix = SL2CMatrix.fractionalFourier(angle: 0.63)
  let reference = try DiscreteLCT.transform(field, matrix: matrix)
  let actual = try MetalLCTTransformer(device: device).transform(field, matrix: matrix)

  #expect(actual.shape == reference.shape)
  for (actualValue, expectedValue) in zip(actual.values, reference.values) {
    #expect((actualValue - expectedValue).magnitude < 2e-4)
  }
}

@Test func metalMatchesCPUSingularBranch() throws {
  guard let device = MTLCreateSystemDefaultDevice() else { return }
  let values = (0..<25).map { index in
    Complex32(real: Float(index % 7) / 7, imaginary: Float(index % 4) / 9)
  }
  let field = try ComplexField(shape: [5, 5], values: values)
  let lens = SL2CMatrix(
    a: .init(real: 1.25),
    b: .zero,
    c: .init(real: -0.28, imaginary: 0.05),
    d: .init(real: 0.8)
  )
  let reference = try DiscreteLCT.transform(field, matrix: lens)
  let actual = try MetalLCTTransformer(device: device).transform(field, matrix: lens)

  for (actualValue, expectedValue) in zip(actual.values, reference.values) {
    #expect((actualValue - expectedValue).magnitude < 3e-4)
  }
}
