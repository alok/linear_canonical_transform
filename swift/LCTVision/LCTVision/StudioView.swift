import RealityKit
import SwiftUI
import UniformTypeIdentifiers

struct StudioView: View {
  @EnvironmentObject private var model: LCTStudioModel
  @StateObject private var scene = SpatialScene()

  var body: some View {
    ZStack {
      RealityView { content in
        try? scene.install(
          geometry: model.sourceGeometry,
          matrix: model.matrix,
          outputField: model.outputField
        )
        content.add(scene.root)
      } update: { _ in
        scene.update(
          geometry: model.sourceGeometry,
          matrix: model.matrix,
          field: model.outputField,
          showsImaginary: model.showsImaginary
        )
      }

      VStack(spacing: 0) {
        header
        Spacer()
        Text("Control deck loading…")
          .font(.caption.monospaced())
          .foregroundStyle(.secondary)
          .padding(10)
          .glassBackgroundEffect()
      }
      .padding(24)
    }
    .fileImporter(
      isPresented: $model.isImporting,
      allowedContentTypes: [UTType(filenameExtension: "obj")!],
      allowsMultipleSelection: false
    ) { result in
      guard case .success(let urls) = result, let url = urls.first else { return }
      model.importOBJ(from: url)
    }
  }

  private var header: some View {
    HStack(alignment: .top) {
      VStack(alignment: .leading, spacing: 4) {
        Text("LCT // SPATIAL ATLAS")
          .font(.headline.monospaced().weight(.semibold))
        Text("canonical preview")
          .foregroundStyle(.mint)
          + Text("   ·   ")
          .foregroundStyle(.secondary)
          + Text("sampled-field transform")
          .foregroundStyle(.indigo)
      }
      Spacer()
      VStack(alignment: .trailing, spacing: 4) {
        Text(model.sourceName)
          .font(.subheadline.weight(.medium))
        Text(model.status)
          .font(.caption.monospaced())
          .foregroundStyle(model.isTransforming ? .yellow : .secondary)
      }
    }
    .padding(14)
    .glassBackgroundEffect()
  }
}
