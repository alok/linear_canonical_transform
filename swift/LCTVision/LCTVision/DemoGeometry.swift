import LCTSpatial
import simd

enum DemoGeometry {
  static func torus(
    majorSegments: Int = 54,
    minorSegments: Int = 22,
    majorRadius: Float = 0.62,
    minorRadius: Float = 0.24
  ) throws -> MeshGeometry {
    var positions: [SIMD3<Float>] = []
    var normals: [SIMD3<Float>] = []
    var indices: [UInt32] = []
    positions.reserveCapacity(majorSegments * minorSegments)
    normals.reserveCapacity(positions.capacity)

    for majorIndex in 0..<majorSegments {
      let u = 2 * Float.pi * Float(majorIndex) / Float(majorSegments)
      // Face the demo torus toward the viewer. Imported meshes retain their
      // authored orientation.
      let center = SIMD3<Float>(majorRadius * cos(u), majorRadius * sin(u), 0)
      for minorIndex in 0..<minorSegments {
        let v = 2 * Float.pi * Float(minorIndex) / Float(minorSegments)
        let normal = simd_normalize(
          SIMD3<Float>(cos(u) * cos(v), sin(u) * cos(v), sin(v))
        )
        positions.append(center + minorRadius * normal)
        normals.append(normal)
      }
    }

    for majorIndex in 0..<majorSegments {
      let nextMajor = (majorIndex + 1) % majorSegments
      for minorIndex in 0..<minorSegments {
        let nextMinor = (minorIndex + 1) % minorSegments
        let i00 = UInt32(majorIndex * minorSegments + minorIndex)
        let i10 = UInt32(nextMajor * minorSegments + minorIndex)
        let i01 = UInt32(majorIndex * minorSegments + nextMinor)
        let i11 = UInt32(nextMajor * minorSegments + nextMinor)
        indices.append(contentsOf: [i00, i10, i11, i00, i11, i01])
      }
    }
    return try MeshGeometry(positions: positions, normals: normals, indices: indices)
  }
}
