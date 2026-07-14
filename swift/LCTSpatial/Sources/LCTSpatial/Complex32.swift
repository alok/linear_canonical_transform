import Foundation

/// A compact complex scalar with a stable two-float layout for Swift/Metal interop.
@frozen
public struct Complex32: Codable, Hashable, Sendable {
  public var real: Float
  public var imaginary: Float

  @inlinable
  public init(real: Float = 0, imaginary: Float = 0) {
    self.real = real
    self.imaginary = imaginary
  }

  @inlinable public static var zero: Self { .init() }
  @inlinable public static var one: Self { .init(real: 1) }
  @inlinable public static var i: Self { .init(imaginary: 1) }

  @inlinable
  public var magnitudeSquared: Float {
    real * real + imaginary * imaginary
  }

  @inlinable
  public var magnitude: Float {
    sqrt(magnitudeSquared)
  }

  @inlinable
  public var phase: Float {
    atan2(imaginary, real)
  }

  @inlinable
  public var conjugate: Self {
    .init(real: real, imaginary: -imaginary)
  }

  @inlinable
  public func exponential(maximumReal: Float = 20) -> Self {
    let boundedReal = min(max(real, -maximumReal), maximumReal)
    let scale = exp(boundedReal)
    return .init(
      real: scale * cos(imaginary),
      imaginary: scale * sin(imaginary)
    )
  }
}

extension Complex32: CustomStringConvertible {
  public var description: String {
    let sign = imaginary < 0 ? "−" : "+"
    return String(format: "%.3f %@ %.3fi", real, sign, abs(imaginary))
  }
}

@inlinable
public prefix func - (value: Complex32) -> Complex32 {
  .init(real: -value.real, imaginary: -value.imaginary)
}

@inlinable
public func + (lhs: Complex32, rhs: Complex32) -> Complex32 {
  .init(real: lhs.real + rhs.real, imaginary: lhs.imaginary + rhs.imaginary)
}

@inlinable
public func - (lhs: Complex32, rhs: Complex32) -> Complex32 {
  .init(real: lhs.real - rhs.real, imaginary: lhs.imaginary - rhs.imaginary)
}

@inlinable
public func * (lhs: Complex32, rhs: Complex32) -> Complex32 {
  .init(
    real: lhs.real * rhs.real - lhs.imaginary * rhs.imaginary,
    imaginary: lhs.real * rhs.imaginary + lhs.imaginary * rhs.real
  )
}

@inlinable
public func * (lhs: Complex32, rhs: Float) -> Complex32 {
  .init(real: lhs.real * rhs, imaginary: lhs.imaginary * rhs)
}

@inlinable
public func * (lhs: Float, rhs: Complex32) -> Complex32 {
  rhs * lhs
}

@inlinable
public func / (lhs: Complex32, rhs: Float) -> Complex32 {
  .init(real: lhs.real / rhs, imaginary: lhs.imaginary / rhs)
}

@inlinable
public func / (lhs: Complex32, rhs: Complex32) -> Complex32 {
  let denominator = rhs.magnitudeSquared
  return .init(
    real: (lhs.real * rhs.real + lhs.imaginary * rhs.imaginary) / denominator,
    imaginary: (lhs.imaginary * rhs.real - lhs.real * rhs.imaginary) / denominator
  )
}

@inlinable
public func += (lhs: inout Complex32, rhs: Complex32) {
  lhs = lhs + rhs
}
