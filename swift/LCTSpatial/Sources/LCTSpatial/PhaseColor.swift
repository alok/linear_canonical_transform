import Foundation
import simd

public enum PhaseColor {
  /// Maps phase around a perceptual color wheel and magnitude to luminance.
  public static func rgba(
    value: Complex32,
    maximumMagnitude: Float,
    gamma: Float = 0.55,
    alpha: Float = 1
  ) -> SIMD4<Float> {
    let magnitude = pow(
      min(value.magnitude / max(maximumMagnitude, .leastNonzeroMagnitude), 1), gamma)
    let hue = (value.phase + .pi) / (2 * .pi)
    let rgb = hsvToRGB(hue: hue, saturation: 0.82, value: 0.18 + 0.82 * magnitude)
    return SIMD4<Float>(rgb.x, rgb.y, rgb.z, alpha * magnitude)
  }

  private static func hsvToRGB(hue: Float, saturation: Float, value: Float) -> SIMD3<Float> {
    let scaled = (hue - floor(hue)) * 6
    let sector = Int(floor(scaled)) % 6
    let fraction = scaled - floor(scaled)
    let p = value * (1 - saturation)
    let q = value * (1 - fraction * saturation)
    let t = value * (1 - (1 - fraction) * saturation)
    switch sector {
    case 0: return .init(value, t, p)
    case 1: return .init(q, value, p)
    case 2: return .init(p, value, t)
    case 3: return .init(p, q, value)
    case 4: return .init(t, p, value)
    default: return .init(value, p, q)
    }
  }
}
