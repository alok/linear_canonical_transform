# LCT Vision

A volumetric visionOS instrument for scrubbing an `SL(2,C)` LCT while comparing
two deliberately distinct views:

- the left mesh is a zero-latency canonical-map preview;
- the right phase-colored cloud is the actual separable transform of a sampled
  three-dimensional complex field.

Generate the Xcode project and build for the installed simulator:

```sh
xcodegen generate
xcodebuild \
  -project LCTVision.xcodeproj \
  -scheme LCTVision \
  -sdk xrsimulator \
  -destination 'platform=visionOS Simulator,name=Apple Vision Pro,OS=26.5' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

The app imports Wavefront OBJ meshes, normalizes them into its study volume,
samples their triangle surfaces, and sends the field through `LCTSpatialMetal`.
