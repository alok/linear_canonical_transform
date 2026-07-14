import LCTSpatial

enum LCTPreset: String, CaseIterable, Identifiable {
  case identity
  case fourier
  case fractional
  case fresnel
  case complexLens

  var id: Self { self }

  var title: String {
    switch self {
    case .identity: "Identity"
    case .fourier: "Fourier"
    case .fractional: "FrFT"
    case .fresnel: "Fresnel"
    case .complexLens: "Complex lens"
    }
  }

  var matrix: SL2CMatrix {
    switch self {
    case .identity:
      .identity
    case .fourier:
      .fourier
    case .fractional:
      .fractionalFourier(angle: .pi / 4)
    case .fresnel:
      .fresnel(distance: .init(real: 0.72))
    case .complexLens:
      .fresnel(distance: .init(real: 0.72, imaginary: 0.16))
    }
  }
}
