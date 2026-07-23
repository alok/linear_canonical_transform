import LCTSpatial
import RealityKit
import UIKit

@MainActor
final class SpectralCloud {
  let root = Entity()

  private let points: [ModelEntity]
  private let materials: [UnlitMaterial]
  private let maximumPoints: Int
  private var signature: Int = 0

  init(maximumPoints: Int = 520, phaseBins: Int = 18) {
    self.maximumPoints = maximumPoints
    let sphere = MeshResource.generateSphere(radius: 0.008)
    let phaseMaterials = (0..<phaseBins).map { index in
      let phase = 2 * Float.pi * Float(index) / Float(phaseBins) - .pi
      let value = Complex32(real: cos(phase), imaginary: sin(phase))
      let rgba = PhaseColor.rgba(value: value, maximumMagnitude: 1)
      var material = UnlitMaterial()
      material.color = .init(
        tint: UIColor(
          red: CGFloat(rgba.x),
          green: CGFloat(rgba.y),
          blue: CGFloat(rgba.z),
          alpha: 0.96
        )
      )
      return material
    }
    let pointEntities = (0..<maximumPoints).map { _ in
      let point = ModelEntity(mesh: sphere, materials: [phaseMaterials[0]])
      point.isEnabled = false
      return point
    }
    materials = phaseMaterials
    points = pointEntities
    for point in pointEntities {
      root.addChild(point)
    }
    root.name = "True sampled-field LCT"
  }

  func update(field: ComplexField) {
    let nextSignature = field.values.reduce(into: field.count) { partial, value in
      partial = partial &* 31 &+ Int(value.real.bitPattern ^ value.imaginary.bitPattern)
    }
    guard signature != nextSignature else { return }
    signature = nextSignature

    let maximum = max(field.maxMagnitude, .leastNonzeroMagnitude)
    let ranked = field.values.enumerated()
      .filter { $0.element.magnitude > maximum * 0.065 }
      .sorted { $0.element.magnitude > $1.element.magnitude }
      .prefix(maximumPoints)
    let shape = field.shape
    guard shape.count == 3 else { return }

    for (displayIndex, sample) in ranked.enumerated() {
      let point = points[displayIndex]
      let flat = sample.offset
      let yz = shape[1] * shape[2]
      let x = flat / yz
      let y = (flat % yz) / shape[2]
      let z = flat % shape[2]
      let coordinate = SIMD3<Float>(
        normalizedCoordinate(x, length: shape[0]),
        normalizedCoordinate(y, length: shape[1]),
        normalizedCoordinate(z, length: shape[2])
      )
      let magnitude = pow(sample.element.magnitude / maximum, 0.55)
      let phaseUnit = (sample.element.phase + .pi) / (2 * .pi)
      let materialIndex = min(
        materials.count - 1,
        max(0, Int(phaseUnit * Float(materials.count)))
      )
      point.position = coordinate * 0.58
      point.scale = SIMD3<Float>(repeating: 0.55 + 1.55 * magnitude)
      point.model?.materials = [materials[materialIndex]]
      point.isEnabled = true
    }

    for point in points.dropFirst(ranked.count) {
      point.isEnabled = false
    }
  }

  private func normalizedCoordinate(_ index: Int, length: Int) -> Float {
    guard length > 1 else { return 0 }
    return Float(index) / Float(length - 1) - 0.5
  }
}
