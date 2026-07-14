import Foundation

public enum ComplexFieldError: Error, Equatable, Sendable {
  case emptyShape
  case nonPositiveDimension(Int)
  case valueCount(expected: Int, actual: Int)
  case rankMismatch(expected: Int, actual: Int)
  case indexOutOfBounds(axis: Int, value: Int)
}

/// A row-major complex field with arbitrary rank.
public struct ComplexField: Sendable, Equatable {
  public let shape: [Int]
  public var values: [Complex32]

  public init(shape: [Int], values: [Complex32]) throws {
    guard !shape.isEmpty else { throw ComplexFieldError.emptyShape }
    for dimension in shape where dimension <= 0 {
      throw ComplexFieldError.nonPositiveDimension(dimension)
    }
    let expected = shape.reduce(1, *)
    guard values.count == expected else {
      throw ComplexFieldError.valueCount(expected: expected, actual: values.count)
    }
    self.shape = shape
    self.values = values
  }

  public init(shape: [Int], repeating value: Complex32 = .zero) throws {
    try self.init(shape: shape, values: .init(repeating: value, count: shape.reduce(1, *)))
  }

  public var rank: Int { shape.count }
  public var count: Int { values.count }
  public var maxMagnitude: Float { values.lazy.map(\.magnitude).max() ?? 0 }

  public func flatIndex(_ indices: [Int]) throws -> Int {
    guard indices.count == rank else {
      throw ComplexFieldError.rankMismatch(expected: rank, actual: indices.count)
    }
    var result = 0
    for (axis, index) in indices.enumerated() {
      guard (0..<shape[axis]).contains(index) else {
        throw ComplexFieldError.indexOutOfBounds(axis: axis, value: index)
      }
      result = result * shape[axis] + index
    }
    return result
  }

  public func normalizedMagnitudes(gamma: Float = 0.55) -> [Float] {
    let maximum = max(maxMagnitude, .leastNonzeroMagnitude)
    return values.map { pow($0.magnitude / maximum, gamma) }
  }
}

public enum DiscreteLCTError: Error, Equatable, Sendable {
  case singularB
}

/// A small, deterministic CPU reference for the separable finite-grid LCT.
///
/// The scalar `SL(2,C)` block is applied independently along each field axis.
/// This implementation is intentionally O(rank × n^(rank+1)) and is meant for
/// tests, tiny fields, and the fallback path; `LCTSpatialMetal` owns live grids.
public enum DiscreteLCT {
  public static func transform(
    _ field: ComplexField,
    matrix: SL2CMatrix,
    maximumKernelExponent: Float = 12
  ) throws -> ComplexField {
    if matrix == .identity { return field }
    guard matrix.b.magnitude > 1e-6 else { throw DiscreteLCTError.singularB }

    var values = field.values
    for axis in field.shape.indices {
      values = transformAxis(
        values,
        shape: field.shape,
        axis: axis,
        matrix: matrix,
        maximumKernelExponent: maximumKernelExponent
      )
    }
    return try ComplexField(shape: field.shape, values: values)
  }

  private static func transformAxis(
    _ input: [Complex32],
    shape: [Int],
    axis: Int,
    matrix: SL2CMatrix,
    maximumKernelExponent: Float
  ) -> [Complex32] {
    let length = shape[axis]
    let stride = shape.dropFirst(axis + 1).reduce(1, *)
    let block = length * stride
    let outerCount = input.count / block
    let normalization = 1 / sqrt(Float(length))
    let piI = Complex32(real: 0, imaginary: .pi)
    var output = [Complex32](repeating: .zero, count: input.count)

    for outer in 0..<outerCount {
      let blockStart = outer * block
      for inner in 0..<stride {
        for outputIndex in 0..<length {
          let y = Float(outputIndex) - Float(length - 1) / 2
          var sum = Complex32.zero
          for inputIndex in 0..<length {
            let x = Float(inputIndex) - Float(length - 1) / 2
            let numerator =
              matrix.a * (x * x)
              - Complex32(real: 2 * x * y)
              + matrix.d * (y * y)
            let phase = piI * (numerator / (matrix.b * Float(length)))
            let kernel = phase.exponential(maximumReal: maximumKernelExponent)
            sum += input[blockStart + inputIndex * stride + inner] * kernel
          }
          output[blockStart + outputIndex * stride + inner] = sum * normalization
        }
      }
    }
    return output
  }
}
