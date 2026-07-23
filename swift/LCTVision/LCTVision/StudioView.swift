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
        HStack {
          viewLabel("CANONICAL MAP", detail: "instant preview", color: .mint)
          Spacer()
          viewLabel("SAMPLED FIELD", detail: "amplitude × phase", color: .indigo)
        }
        .padding(.horizontal, 34)
        .padding(.top, 12)
        Spacer()
        ControlDeck()
          .padding(.bottom, 30)
      }
      .padding(20)
    }
    .fileImporter(
      isPresented: $model.presentsImporter,
      allowedContentTypes: [UTType(filenameExtension: "obj")!],
      allowsMultipleSelection: false
    ) { result in
      guard case .success(let urls) = result, let url = urls.first else { return }
      model.importOBJ(from: url)
    }
  }

  private func viewLabel(_ title: String, detail: String, color: Color) -> some View {
    HStack(spacing: 7) {
      RoundedRectangle(cornerRadius: 1)
        .fill(color)
        .frame(width: 3, height: 25)
      VStack(alignment: .leading, spacing: 1) {
        Text(title)
          .font(.system(size: 9, weight: .bold, design: .monospaced))
          .foregroundStyle(color)
        Text(detail)
          .font(.system(size: 8, design: .monospaced))
          .foregroundStyle(.secondary)
      }
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
