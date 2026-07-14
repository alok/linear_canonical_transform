// swift-tools-version: 6.0

import PackageDescription

// The root manifest makes the Swift library consumable directly from this Git
// repository. The nested manifest remains convenient for focused development.
let package = Package(
  name: "LCTSpatial",
  platforms: [
    .macOS(.v15),
    .visionOS(.v2),
  ],
  products: [
    .library(name: "LCTSpatial", targets: ["LCTSpatial"]),
    .library(name: "LCTSpatialMetal", targets: ["LCTSpatialMetal"]),
  ],
  targets: [
    .target(
      name: "LCTSpatial",
      path: "swift/LCTSpatial/Sources/LCTSpatial"
    ),
    .target(
      name: "LCTSpatialMetal",
      dependencies: ["LCTSpatial"],
      path: "swift/LCTSpatial/Sources/LCTSpatialMetal",
      resources: [.process("Shaders")]
    ),
    .testTarget(
      name: "LCTSpatialTests",
      dependencies: ["LCTSpatial"],
      path: "swift/LCTSpatial/Tests/LCTSpatialTests"
    ),
    .testTarget(
      name: "LCTSpatialMetalTests",
      dependencies: ["LCTSpatial", "LCTSpatialMetal"],
      path: "swift/LCTSpatial/Tests/LCTSpatialMetalTests"
    ),
  ]
)
