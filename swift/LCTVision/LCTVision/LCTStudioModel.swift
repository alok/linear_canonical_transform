import Combine
import Foundation
import LCTSpatial
import LCTSpatialMetal

@MainActor
final class LCTStudioModel: ObservableObject {
  @Published private(set) var matrix = SL2CMatrix.identity
  @Published private(set) var dependent: SL2CCoefficient = .d
  @Published private(set) var outputField: ComplexField
  @Published private(set) var sourceGeometry: MeshGeometry
  @Published private(set) var sourceName = "Torus study"
  @Published private(set) var isTransforming = false
  @Published private(set) var status = "Ready · sampled-field LCT"
  @Published var showsImaginary = true
  @Published var isImporting = false
  @Published var resolution = 14

  private var constraint = SL2CConstraint()
  private var sourceField: ComplexField
  private let transformer: MetalLCTTransformer?
  private var transformTask: Task<Void, Never>?
  private var generation = 0

  init() {
    let geometry = try! DemoGeometry.torus()
    let field = try! MeshVoxelizer.surfaceField(from: geometry, resolution: 14)
    sourceGeometry = geometry
    sourceField = field
    outputField = field
    transformer = try? MetalLCTTransformer()
  }

  var determinant: Complex32 { matrix.determinant }
  var hasMetal: Bool { transformer != nil }

  func value(for coefficient: SL2CCoefficient) -> Complex32 {
    matrix[coefficient]
  }

  func set(_ coefficient: SL2CCoefficient, to value: Complex32) {
    do {
      matrix = try constraint.set(coefficient, to: value)
      dependent = constraint.dependent
      status = "Scrubbing \(coefficient.rawValue) · det locked"
      scheduleTransform()
    } catch {
      status = "No stable unimodular pivot"
    }
  }

  func apply(_ preset: LCTPreset) {
    let preferredDependent: SL2CCoefficient = preset == .fourier ? .c : .d
    constraint.replace(with: preset.matrix, dependent: preferredDependent)
    matrix = preset.matrix
    dependent = preferredDependent
    status = "\(preset.title) preset"
    scheduleTransform(immediate: true)
  }

  func rebuildResolution(_ newResolution: Int) {
    resolution = newResolution
    status = "Sampling \(newResolution)³ field…"
    let geometry = sourceGeometry
    Task {
      do {
        let field = try await Task.detached {
          try MeshVoxelizer.surfaceField(from: geometry, resolution: newResolution)
        }.value
        sourceField = field
        scheduleTransform(immediate: true)
      } catch {
        status = "Sampling failed: \(error.localizedDescription)"
      }
    }
  }

  func importOBJ(from url: URL) {
    isImporting = true
    status = "Importing \(url.lastPathComponent)…"
    let resolution = resolution
    Task {
      let scoped = url.startAccessingSecurityScopedResource()
      defer {
        if scoped { url.stopAccessingSecurityScopedResource() }
      }
      do {
        let result = try await Task.detached { () -> (MeshGeometry, ComplexField) in
          let geometry = try OBJMeshLoader.load(contentsOf: url)
          let field = try MeshVoxelizer.surfaceField(from: geometry, resolution: resolution)
          return (geometry, field)
        }.value
        sourceGeometry = result.0
        sourceField = result.1
        sourceName = url.deletingPathExtension().lastPathComponent
        isImporting = false
        status = "Imported \(url.lastPathComponent)"
        scheduleTransform(immediate: true)
      } catch {
        isImporting = false
        status = "Import failed: \(error.localizedDescription)"
      }
    }
  }

  private func scheduleTransform(immediate: Bool = false) {
    generation += 1
    let requestedGeneration = generation
    transformTask?.cancel()
    let field = sourceField
    let matrix = matrix
    let transformer = transformer
    isTransforming = true

    transformTask = Task {
      if !immediate {
        try? await Task.sleep(for: .milliseconds(42))
      }
      guard !Task.isCancelled else { return }
      do {
        let transformed = try await Task.detached {
          if let transformer {
            return try transformer.transform(field, matrix: matrix)
          }
          return try DiscreteLCT.transform(field, matrix: matrix)
        }.value
        guard !Task.isCancelled, requestedGeneration == generation else { return }
        outputField = transformed
        isTransforming = false
        status = transformer == nil ? "CPU reference · live" : "Metal field · live"
      } catch {
        guard requestedGeneration == generation else { return }
        isTransforming = false
        status = "Transform unavailable: \(error.localizedDescription)"
      }
    }
  }
}
