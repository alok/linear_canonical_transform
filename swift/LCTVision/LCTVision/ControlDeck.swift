import LCTSpatial
import SwiftUI

struct ControlDeck: View {
  @EnvironmentObject private var model: LCTStudioModel

  private let accents: [SL2CCoefficient: Color] = [
    .a: .mint,
    .b: .cyan,
    .c: .purple,
    .d: .orange,
  ]

  var body: some View {
    VStack(spacing: 10) {
      HStack(spacing: 12) {
        VStack(alignment: .leading, spacing: 2) {
          Text("SL(2,C) SCRUB DECK")
            .font(.caption.monospaced().weight(.bold))
            .foregroundStyle(.white)
          HStack(spacing: 6) {
            Circle()
              .fill(model.hasMetal ? Color.mint : .yellow)
              .frame(width: 6, height: 6)
            Text(model.hasMetal ? "METAL FIELD ONLINE" : "CPU REFERENCE")
              .font(.system(size: 9, weight: .medium, design: .monospaced))
              .foregroundStyle(.secondary)
          }
        }
        Spacer()
        determinantReadout
      }

      HStack(spacing: 9) {
        ForEach(SL2CCoefficient.allCases, id: \.self) { coefficient in
          ArgandPad(
            coefficient: coefficient,
            value: model.value(for: coefficient),
            isDependent: model.dependent == coefficient,
            range: 2,
            accent: accents[coefficient] ?? .white
          ) { value in
            model.set(coefficient, to: value)
          }
        }
      }

      HStack(spacing: 8) {
        Button {
          model.togglePlayback()
        } label: {
          Label(
            model.isPlaying ? "Pause" : "Orbit",
            systemImage: model.isPlaying ? "pause.fill" : "play.fill")
        }
        .tint(model.isPlaying ? .orange : .mint)

        ForEach(LCTPreset.allCases) { preset in
          Button(preset.title) { model.apply(preset) }
            .tint(model.selectedPreset == preset ? .cyan : .gray)
        }

        Divider().frame(height: 22)

        Button {
          model.presentsImporter = true
        } label: {
          Label(model.isImporting ? "Sampling…" : "Import OBJ", systemImage: "cube.transparent")
        }
        .disabled(model.isImporting)

        Menu("\(model.resolution)³") {
          ForEach([10, 14, 18, 22], id: \.self) { resolution in
            Button("\(resolution)³ field") { model.rebuildResolution(resolution) }
          }
        }

        Toggle("Imaginary ghost", isOn: $model.showsImaginary)
          .toggleStyle(.button)
          .tint(.indigo)
      }
      .font(.caption.weight(.medium))
      .buttonStyle(.bordered)
      .controlSize(.small)
    }
    .padding(13)
    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
    .overlay {
      RoundedRectangle(cornerRadius: 22, style: .continuous)
        .stroke(.white.opacity(0.14), lineWidth: 0.7)
    }
    .frame(maxWidth: 650)
  }

  private var determinantReadout: some View {
    HStack(spacing: 8) {
      VStack(alignment: .trailing, spacing: 1) {
        Text("UNIMODULAR LOCK")
          .font(.system(size: 8, weight: .bold, design: .monospaced))
          .foregroundStyle(.mint)
        Text("det = \(model.determinant.description)")
          .font(.system(size: 10, weight: .medium, design: .monospaced))
          .foregroundStyle(.white.opacity(0.84))
      }
      Image(systemName: "lock.fill")
        .font(.caption)
        .foregroundStyle(.mint)
    }
    .padding(.horizontal, 10)
    .padding(.vertical, 6)
    .background(.mint.opacity(0.08), in: Capsule())
  }
}
