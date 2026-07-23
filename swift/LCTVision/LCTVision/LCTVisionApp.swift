import SwiftUI

@main
struct LCTVisionApp: App {
  @StateObject private var model = LCTStudioModel()

  var body: some Scene {
    WindowGroup {
      StudioView()
        .environmentObject(model)
    }
    .windowStyle(.volumetric)
    .defaultSize(width: 1.32, height: 0.86, depth: 0.72, in: .meters)
  }
}
