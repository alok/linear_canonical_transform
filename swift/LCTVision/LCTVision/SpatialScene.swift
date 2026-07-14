import LCTSpatial
import LCTSpatialMetal
import RealityKit
import UIKit

@MainActor
final class SpatialScene: ObservableObject {
  let root = Entity()

  private var preview: MetalCanonicalPreview?
  private var previewEntity: ModelEntity?
  private var spectralCloud: SpectralCloud?
  private var geometrySignature = 0

  func install(geometry: MeshGeometry, matrix: SL2CMatrix, outputField: ComplexField) throws {
    root.children.removeAll()
    geometrySignature = signature(for: geometry)

    let preview = try MetalCanonicalPreview(geometry: geometry)
    let materials = makePreviewMaterials()
    let previewEntity = ModelEntity(mesh: preview.meshResource, materials: materials)
    previewEntity.position = SIMD3<Float>(-0.31, 0.03, 0)
    previewEntity.scale = SIMD3<Float>(repeating: 0.36)
    previewEntity.name = "Canonical preview"
    root.addChild(previewEntity)

    let cloud = SpectralCloud()
    cloud.root.position = SIMD3<Float>(0.31, 0.03, 0)
    cloud.update(field: outputField)
    root.addChild(cloud.root)

    addAxisFrame(at: SIMD3<Float>(-0.31, 0.03, 0), color: .systemTeal)
    addAxisFrame(at: SIMD3<Float>(0.31, 0.03, 0), color: .systemIndigo)

    self.preview = preview
    self.previewEntity = previewEntity
    spectralCloud = cloud
    try preview.update(matrix: matrix)
  }

  func update(
    geometry: MeshGeometry,
    matrix: SL2CMatrix,
    field: ComplexField,
    showsImaginary: Bool
  ) {
    if signature(for: geometry) != geometrySignature {
      try? install(geometry: geometry, matrix: matrix, outputField: field)
    }
    try? preview?.update(matrix: matrix, showsImaginary: showsImaginary)
    spectralCloud?.update(field: field)
  }

  private func makePreviewMaterials() -> [any Material] {
    var real = SimpleMaterial()
    real.color = .init(tint: UIColor(red: 0.17, green: 0.95, blue: 0.78, alpha: 0.96))
    real.roughness = 0.28
    real.metallic = 0.18

    var imaginary = UnlitMaterial()
    imaginary.color = .init(tint: UIColor(red: 0.38, green: 0.52, blue: 1, alpha: 0.25))
    imaginary.blending = .transparent(opacity: 0.28)
    return [real, imaginary]
  }

  private func addAxisFrame(at origin: SIMD3<Float>, color: UIColor) {
    var material = UnlitMaterial()
    material.color = .init(tint: color.withAlphaComponent(0.38))
    let dimensions: [(SIMD3<Float>, SIMD3<Float>)] = [
      (SIMD3<Float>(0.52, 0.0015, 0.0015), SIMD3<Float>(0, -0.29, 0)),
      (SIMD3<Float>(0.0015, 0.52, 0.0015), SIMD3<Float>(-0.29, 0, 0)),
      (SIMD3<Float>(0.0015, 0.0015, 0.52), SIMD3<Float>(-0.29, -0.29, 0)),
    ]
    for (size, offset) in dimensions {
      let axis = ModelEntity(mesh: .generateBox(size: size), materials: [material])
      axis.position = origin + offset
      root.addChild(axis)
    }
  }

  private func signature(for geometry: MeshGeometry) -> Int {
    var hasher = Hasher()
    hasher.combine(geometry.positions.count)
    hasher.combine(geometry.indices.count)
    if let first = geometry.positions.first {
      hasher.combine(first.x)
      hasher.combine(first.y)
      hasher.combine(first.z)
    }
    return hasher.finalize()
  }
}
