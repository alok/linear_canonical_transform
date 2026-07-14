import Foundation

public enum SL2CCoefficient: String, CaseIterable, Codable, Hashable, Sendable {
  case a, b, c, d
}

/// A 2×2 complex matrix constrained by `ad - bc = 1` when used as an LCT control.
@frozen
public struct SL2CMatrix: Codable, Hashable, Sendable {
  public var a: Complex32
  public var b: Complex32
  public var c: Complex32
  public var d: Complex32

  @inlinable
  public init(a: Complex32, b: Complex32, c: Complex32, d: Complex32) {
    self.a = a
    self.b = b
    self.c = c
    self.d = d
  }

  public static let identity = Self(a: .one, b: .zero, c: .zero, d: .one)
  public static let fourier = Self(a: .zero, b: .one, c: -.one, d: .zero)

  public static func fractionalFourier(angle: Float) -> Self {
    let cosine = Complex32(real: cos(angle))
    let sine = Complex32(real: sin(angle))
    return .init(a: cosine, b: sine, c: -sine, d: cosine)
  }

  public static func fresnel(distance: Complex32) -> Self {
    .init(a: .one, b: distance, c: .zero, d: .one)
  }

  @inlinable
  public var determinant: Complex32 {
    a * d - b * c
  }

  @inlinable
  public func isUnimodular(tolerance: Float = 1e-4) -> Bool {
    (determinant - .one).magnitude <= tolerance
  }

  public subscript(coefficient: SL2CCoefficient) -> Complex32 {
    get {
      switch coefficient {
      case .a: a
      case .b: b
      case .c: c
      case .d: d
      }
    }
    set {
      switch coefficient {
      case .a: a = newValue
      case .b: b = newValue
      case .c: c = newValue
      case .d: d = newValue
      }
    }
  }
}

public enum SL2CConstraintError: Error, Equatable, Sendable {
  case noStablePivot
}

/// Maintains determinant one while a person continuously scrubs any matrix entry.
///
/// The controller prefers the selected dependent coefficient (normally `d`). If
/// that solve becomes ill-conditioned, it pivots to the best available entry.
public struct SL2CConstraint: Sendable {
  public private(set) var matrix: SL2CMatrix
  public private(set) var dependent: SL2CCoefficient
  public var lockDeterminant: Bool
  public var pivotEpsilon: Float

  public init(
    matrix: SL2CMatrix = .identity,
    dependent: SL2CCoefficient = .d,
    lockDeterminant: Bool = true,
    pivotEpsilon: Float = 1e-4
  ) {
    self.matrix = matrix
    self.dependent = dependent
    self.lockDeterminant = lockDeterminant
    self.pivotEpsilon = pivotEpsilon
  }

  @discardableResult
  public mutating func set(
    _ coefficient: SL2CCoefficient,
    to value: Complex32
  ) throws -> SL2CMatrix {
    matrix[coefficient] = value
    guard lockDeterminant else { return matrix }

    let preferred = coefficient == dependent ? nil : dependent
    let selected =
      try preferred.flatMap { candidate in
        pivotMagnitude(for: candidate, in: matrix) >= pivotEpsilon ? candidate : nil
      } ?? bestPivot(excluding: coefficient)

    matrix[selected] = solvedValue(for: selected, in: matrix)
    dependent = selected
    return matrix
  }

  public mutating func replace(with matrix: SL2CMatrix, dependent: SL2CCoefficient = .d) {
    self.matrix = matrix
    self.dependent = dependent
  }

  private func bestPivot(excluding edited: SL2CCoefficient) throws -> SL2CCoefficient {
    guard
      let best = SL2CCoefficient.allCases
        .filter({ $0 != edited })
        .map({ ($0, pivotMagnitude(for: $0, in: matrix)) })
        .filter({ $0.1 >= pivotEpsilon })
        .max(by: { $0.1 < $1.1 })?.0
    else {
      throw SL2CConstraintError.noStablePivot
    }
    return best
  }

  private func pivotMagnitude(for coefficient: SL2CCoefficient, in matrix: SL2CMatrix) -> Float {
    switch coefficient {
    case .d: matrix.a.magnitude
    case .a: matrix.d.magnitude
    case .c: matrix.b.magnitude
    case .b: matrix.c.magnitude
    }
  }

  private func solvedValue(for coefficient: SL2CCoefficient, in matrix: SL2CMatrix) -> Complex32 {
    switch coefficient {
    case .d:
      (.one + matrix.b * matrix.c) / matrix.a
    case .a:
      (.one + matrix.b * matrix.c) / matrix.d
    case .c:
      (matrix.a * matrix.d - .one) / matrix.b
    case .b:
      (matrix.a * matrix.d - .one) / matrix.c
    }
  }
}

public enum CanonicalPairTransform {
  /// Applies the canonical matrix to paired fields of any dimensionality.
  public static func apply(
    _ matrix: SL2CMatrix,
    q: [Complex32],
    p: [Complex32]
  ) -> (q: [Complex32], p: [Complex32]) {
    precondition(q.count == p.count, "q and p must have the same dimensionality")
    var transformedQ = [Complex32](repeating: .zero, count: q.count)
    var transformedP = [Complex32](repeating: .zero, count: p.count)
    for index in q.indices {
      transformedQ[index] = matrix.a * q[index] + matrix.b * p[index]
      transformedP[index] = matrix.c * q[index] + matrix.d * p[index]
    }
    return (transformedQ, transformedP)
  }
}
