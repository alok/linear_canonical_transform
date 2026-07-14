import Foundation
import Testing

@testable import LCTSpatial

@Test func complexArithmetic() {
  let lhs = Complex32(real: 2, imaginary: 3)
  let rhs = Complex32(real: -1, imaginary: 4)
  #expect(lhs * rhs == Complex32(real: -14, imaginary: 5))
  let quotient = (lhs * rhs) / rhs
  #expect(abs(quotient.real - lhs.real) < 1e-5)
  #expect(abs(quotient.imaginary - lhs.imaginary) < 1e-5)
}

@Test func determinantLockUpdatesD() throws {
  var lock = SL2CConstraint()
  try lock.set(.b, to: .init(real: 0.4, imaginary: 0.2))
  try lock.set(.c, to: .init(real: -0.3, imaginary: 0.1))
  #expect(lock.dependent == .d)
  #expect(lock.matrix.isUnimodular(tolerance: 1e-5))
}

@Test func determinantLockPivotsNearZeroA() throws {
  var lock = SL2CConstraint(matrix: .fourier, dependent: .d)
  try lock.set(.a, to: .zero)
  #expect(lock.dependent != .d)
  #expect(lock.matrix.isUnimodular(tolerance: 1e-5))
}

@Test func canonicalPairIsDimensionAgnostic() {
  let q = [Complex32(real: 1), .init(real: 2), .init(real: 3), .init(real: 4)]
  let p = [Complex32(real: -1), .init(real: -2), .init(real: -3), .init(real: -4)]
  let transformed = CanonicalPairTransform.apply(.fourier, q: q, p: p)
  #expect(transformed.q == p)
  #expect(transformed.p == q.map(-))
}

@Test func fourierOfCenteredImpulseHasFlatMagnitude() throws {
  var values = [Complex32](repeating: .zero, count: 5)
  values[2] = .one
  let field = try ComplexField(shape: [5], values: values)
  let transformed = try DiscreteLCT.transform(field, matrix: .fourier)
  let expected = 1 / sqrt(Float(5))
  for value in transformed.values {
    #expect(abs(value.magnitude - expected) < 1e-5)
  }
}

@Test func multidimensionalIdentityPreservesField() throws {
  let values = (0..<24).map { Complex32(real: Float($0), imaginary: Float(-$0)) }
  let field = try ComplexField(shape: [2, 3, 4], values: values)
  #expect(try DiscreteLCT.transform(field, matrix: .identity) == field)
}

@Test func objImportAndVoxelization() throws {
  let source = """
    v -1 -1 0
    v 1 -1 0
    v 0 1 0
    f 1 2 3
    """
  let mesh = try OBJMeshLoader.parse(source)
  #expect(mesh.positions.count == 3)
  #expect(mesh.indices == [0, 1, 2])
  let field = try MeshVoxelizer.surfaceField(from: mesh, resolution: 8)
  #expect(field.shape == [8, 8, 8])
  #expect(field.maxMagnitude > 0)
}
