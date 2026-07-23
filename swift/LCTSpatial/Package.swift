// swift-tools-version: 6.0

import PackageDescription

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
    .target(name: "LCTSpatial"),
    .target(
      name: "LCTSpatialMetal",
      dependencies: ["LCTSpatial"],
      resources: [.process("Shaders")]
    ),
    .testTarget(
      name: "LCTSpatialTests",
      dependencies: ["LCTSpatial"]
    ),
    .testTarget(
      name: "LCTSpatialMetalTests",
      dependencies: ["LCTSpatial", "LCTSpatialMetal"]
    ),
  ]
)
