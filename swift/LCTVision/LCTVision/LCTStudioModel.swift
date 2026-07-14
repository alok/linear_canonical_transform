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
  @Published private(set) var selectedPreset: LCTPreset? = .identity
  @Published private(set) var isPlaying = false
  @Published var showsImaginary = true
  @Published var presentsImporter = false
  @Published private(set) var isImporting = false
  @Published var resolution = 14

  private var constraint = SL2CConstraint()
  private var sourceField: ComplexField
  private let transformer: MetalLCTTransformer?
  private var transformTask: Task<Void, Never>?
  private var playbackTask: Task<Void, Never>?
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
    stopPlayback()
    do {
      matrix = try constraint.set(coefficient, to: value)
      dependent = constraint.dependent
      selectedPreset = nil
      status = "Scrubbing \(coefficient.rawValue) · det locked"
      scheduleTransform()
    } catch {
      status = "No stable unimodular pivot"
    }
  }

  func apply(_ preset: LCTPreset) {
    stopPlayback()
    let preferredDependent: SL2CCoefficient = preset == .fourier ? .c : .d
    constraint.replace(with: preset.matrix, dependent: preferredDependent)
    matrix = preset.matrix
    dependent = preferredDependent
    selectedPreset = preset
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

  func togglePlayback() {
    if isPlaying {
      stopPlayback()
      status = "Scrub animation paused"
      scheduleTransform(immediate: true)
      return
    }

    isPlaying = true
    selectedPreset = .fractional
    status = "FrFT orbit · live"
    playbackTask = Task {
      var frame = 0
      while !Task.isCancelled {
        let angle = 2 * Float.pi * Float(frame % 96) / 96
        let next = SL2CMatrix.fractionalFourier(angle: angle)
        constraint.replace(with: next, dependent: .d)
        matrix = next
        dependent = .d
        scheduleTransform(immediate: true)
        frame += 1
        try? await Task.sleep(for: .milliseconds(90))
      }
    }
  }

  func importOBJ(from url: URL) {
    stopPlayback()
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

  private func stopPlayback() {
    playbackTask?.cancel()
    playbackTask = nil
    isPlaying = false
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
