import Foundation
import simd

public enum MeshGeometryError: Error, Equatable, Sendable {
  case emptyVertices
  case indexOutOfBounds(UInt32)
  case indexCountNotTriangular
}

public struct MeshGeometry: Sendable, Equatable {
  public var positions: [SIMD3<Float>]
  public var normals: [SIMD3<Float>]
  public var indices: [UInt32]

  public init(
    positions: [SIMD3<Float>],
    normals: [SIMD3<Float>] = [],
    indices: [UInt32]
  ) throws {
    guard !positions.isEmpty else { throw MeshGeometryError.emptyVertices }
    guard indices.count.isMultiple(of: 3) else {
      throw MeshGeometryError.indexCountNotTriangular
    }
    if let invalid = indices.first(where: { Int($0) >= positions.count }) {
      throw MeshGeometryError.indexOutOfBounds(invalid)
    }
    self.positions = positions
    self.indices = indices
    self.normals =
      normals.count == positions.count
      ? normals
      : Self.generatedNormals(positions: positions, indices: indices)
  }

  public func normalized(extent targetExtent: Float = 1.7) throws -> Self {
    var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
    var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
    for position in positions {
      minimum = simd.min(minimum, position)
      maximum = simd.max(maximum, position)
    }
    let center = (minimum + maximum) / 2
    let sourceExtent = max(maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z)
    let scale = sourceExtent > .leastNonzeroMagnitude ? targetExtent / sourceExtent : 1
    return try Self(
      positions: positions.map { ($0 - center) * scale },
      normals: normals,
      indices: indices
    )
  }

  private static func generatedNormals(
    positions: [SIMD3<Float>],
    indices: [UInt32]
  ) -> [SIMD3<Float>] {
    var normals = [SIMD3<Float>](repeating: .zero, count: positions.count)
    for triangle in stride(from: 0, to: indices.count, by: 3) {
      let i0 = Int(indices[triangle])
      let i1 = Int(indices[triangle + 1])
      let i2 = Int(indices[triangle + 2])
      let face = simd_cross(positions[i1] - positions[i0], positions[i2] - positions[i0])
      normals[i0] += face
      normals[i1] += face
      normals[i2] += face
    }
    return normals.map { normal in
      let length = simd_length(normal)
      return length > .leastNonzeroMagnitude ? normal / length : SIMD3<Float>(0, 1, 0)
    }
  }
}

public enum OBJMeshError: Error, Equatable, Sendable {
  case malformedVertex(line: Int)
  case malformedFace(line: Int)
  case noFaces
}

/// A dependency-free Wavefront OBJ loader suitable for document-picker imports.
public enum OBJMeshLoader {
  public static func parse(_ source: String) throws -> MeshGeometry {
    var positions: [SIMD3<Float>] = []
    var indices: [UInt32] = []

    for (offset, rawLine) in source.split(whereSeparator: \.isNewline).enumerated() {
      let lineNumber = offset + 1
      let parts = rawLine.split(whereSeparator: \.isWhitespace)
      guard let command = parts.first else { continue }
      if command == "v" {
        guard parts.count >= 4,
          let x = Float(parts[1]),
          let y = Float(parts[2]),
          let z = Float(parts[3])
        else { throw OBJMeshError.malformedVertex(line: lineNumber) }
        positions.append(.init(x, y, z))
      } else if command == "f" {
        guard parts.count >= 4 else { throw OBJMeshError.malformedFace(line: lineNumber) }
        let face = try parts.dropFirst().map { token -> UInt32 in
          guard let head = token.split(separator: "/", omittingEmptySubsequences: false).first,
            let rawIndex = Int(head), rawIndex != 0
          else { throw OBJMeshError.malformedFace(line: lineNumber) }
          let resolved = rawIndex > 0 ? rawIndex - 1 : positions.count + rawIndex
          guard positions.indices.contains(resolved) else {
            throw OBJMeshError.malformedFace(line: lineNumber)
          }
          return UInt32(resolved)
        }
        for corner in 1..<(face.count - 1) {
          indices.append(contentsOf: [face[0], face[corner], face[corner + 1]])
        }
      }
    }
    guard !indices.isEmpty else { throw OBJMeshError.noFaces }
    return try MeshGeometry(positions: positions, indices: indices)
  }

  public static func load(contentsOf url: URL) throws -> MeshGeometry {
    try parse(String(contentsOf: url, encoding: .utf8))
  }
}

public enum MeshVoxelizer {
  /// Samples triangle surfaces into a cubic complex field for the true LCT path.
  public static func surfaceField(
    from geometry: MeshGeometry,
    resolution: Int
  ) throws -> ComplexField {
    precondition(resolution >= 4)
    let mesh = try geometry.normalized()
    var density = [Float](repeating: 0, count: resolution * resolution * resolution)

    func splat(_ point: SIMD3<Float>) {
      let unit = simd_clamp((point + 1) / 2, .zero, SIMD3<Float>(repeating: 1))
      let scaled = unit * Float(resolution - 1)
      let coordinate = SIMD3<Float>(round(scaled.x), round(scaled.y), round(scaled.z))
      let x = Int(coordinate.x)
      let y = Int(coordinate.y)
      let z = Int(coordinate.z)
      let index = (x * resolution + y) * resolution + z
      density[index] += 1
    }

    for triangle in stride(from: 0, to: mesh.indices.count, by: 3) {
      let p0 = mesh.positions[Int(mesh.indices[triangle])]
      let p1 = mesh.positions[Int(mesh.indices[triangle + 1])]
      let p2 = mesh.positions[Int(mesh.indices[triangle + 2])]
      let longest = max(simd_length(p1 - p0), simd_length(p2 - p1), simd_length(p0 - p2))
      let steps = max(2, min(64, Int(ceil(longest * Float(resolution)))))
      for i in 0...steps {
        for j in 0...(steps - i) {
          let u = Float(i) / Float(steps)
          let v = Float(j) / Float(steps)
          splat(p0 + u * (p1 - p0) + v * (p2 - p0))
        }
      }
    }

    let maximum = max(density.max() ?? 0, 1)
    let values = density.map { Complex32(real: min($0 / maximum, 1)) }
    return try ComplexField(shape: [resolution, resolution, resolution], values: values)
  }
}
