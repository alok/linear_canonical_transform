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
