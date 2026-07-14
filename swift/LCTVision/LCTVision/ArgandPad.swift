import LCTSpatial
import SwiftUI

struct ArgandPad: View {
  let coefficient: SL2CCoefficient
  let value: Complex32
  let isDependent: Bool
  let range: Float
  let accent: Color
  let onChange: (Complex32) -> Void

  @State private var isDragging = false

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack(spacing: 6) {
        Text(coefficient.rawValue.uppercased())
          .font(.caption.monospaced().weight(.bold))
          .foregroundStyle(accent)
        if isDependent {
          Text("AUTO")
            .font(.system(size: 8, weight: .bold, design: .monospaced))
            .foregroundStyle(.black)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(accent, in: Capsule())
        }
        Spacer()
        Text(shortValue)
          .font(.system(size: 9, weight: .medium, design: .monospaced))
          .foregroundStyle(.secondary)
      }

      TimelineView(.animation(minimumInterval: 1 / 24)) { timeline in
        Canvas { context, size in
          drawGrid(context: &context, size: size, date: timeline.date)
        }
      }
      .frame(width: 120, height: 94)
      .contentShape(Rectangle())
      .gesture(
        DragGesture(minimumDistance: 0)
          .onChanged { gesture in
            isDragging = true
            onChange(value(at: gesture.location, size: CGSize(width: 120, height: 94)))
          }
          .onEnded { _ in isDragging = false }
      )
      .hoverEffect(.highlight)
      .accessibilityLabel("Complex coefficient \(coefficient.rawValue)")
      .accessibilityValue(value.description)
    }
    .padding(9)
    .background(
      RoundedRectangle(cornerRadius: 15, style: .continuous)
        .fill(.black.opacity(isDragging ? 0.42 : 0.28))
        .stroke(accent.opacity(isDependent ? 0.72 : 0.28), lineWidth: isDependent ? 1.2 : 0.7)
    )
    .animation(.easeOut(duration: 0.12), value: isDragging)
  }

  private var shortValue: String {
    String(format: "%+.2f %+.2fi", value.real, value.imaginary)
  }

  private func value(at location: CGPoint, size: CGSize) -> Complex32 {
    let normalizedX = Float(location.x / size.width * 2 - 1)
    let normalizedY = Float(1 - location.y / size.height * 2)
    return .init(
      real: min(max(normalizedX, -1), 1) * range,
      imaginary: min(max(normalizedY, -1), 1) * range
    )
  }

  private func point(for value: Complex32, size: CGSize) -> CGPoint {
    let x = CGFloat(min(max(value.real / range, -1), 1))
    let y = CGFloat(min(max(value.imaginary / range, -1), 1))
    return CGPoint(x: (x + 1) * size.width / 2, y: (1 - y) * size.height / 2)
  }

  private func drawGrid(context: inout GraphicsContext, size: CGSize, date: Date) {
    let bounds = CGRect(origin: .zero, size: size)
    context.fill(
      Path(roundedRect: bounds, cornerRadius: 10),
      with: .linearGradient(
        Gradient(colors: [.black.opacity(0.78), accent.opacity(0.11)]),
        startPoint: .zero,
        endPoint: CGPoint(x: size.width, y: size.height)
      )
    )

    for index in 1..<6 {
      let x = size.width * CGFloat(index) / 6
      let y = size.height * CGFloat(index) / 6
      var vertical = Path()
      vertical.move(to: CGPoint(x: x, y: 0))
      vertical.addLine(to: CGPoint(x: x, y: size.height))
      context.stroke(vertical, with: .color(accent.opacity(0.10)), lineWidth: 0.5)
      var horizontal = Path()
      horizontal.move(to: CGPoint(x: 0, y: y))
      horizontal.addLine(to: CGPoint(x: size.width, y: y))
      context.stroke(horizontal, with: .color(accent.opacity(0.10)), lineWidth: 0.5)
    }

    var axes = Path()
    axes.move(to: CGPoint(x: size.width / 2, y: 0))
    axes.addLine(to: CGPoint(x: size.width / 2, y: size.height))
    axes.move(to: CGPoint(x: 0, y: size.height / 2))
    axes.addLine(to: CGPoint(x: size.width, y: size.height / 2))
    context.stroke(axes, with: .color(accent.opacity(0.48)), lineWidth: 0.8)

    let scanPhase = date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 2.2) / 2.2
    let scanY = size.height * scanPhase
    var scan = Path()
    scan.move(to: CGPoint(x: 0, y: scanY))
    scan.addLine(to: CGPoint(x: size.width, y: scanY))
    context.stroke(scan, with: .color(accent.opacity(0.17)), lineWidth: 1)

    let location = point(for: value, size: size)
    context.fill(
      Path(ellipseIn: CGRect(x: location.x - 11, y: location.y - 11, width: 22, height: 22)),
      with: .radialGradient(
        Gradient(colors: [accent.opacity(0.48), .clear]),
        center: location,
        startRadius: 0,
        endRadius: 11
      )
    )
    context.fill(
      Path(ellipseIn: CGRect(x: location.x - 3.5, y: location.y - 3.5, width: 7, height: 7)),
      with: .color(.white)
    )
    context.stroke(
      Path(ellipseIn: CGRect(x: location.x - 5.5, y: location.y - 5.5, width: 11, height: 11)),
      with: .color(accent),
      lineWidth: 1.5
    )
  }
}
