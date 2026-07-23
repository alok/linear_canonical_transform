import Foundation
import LCTSpatial
@preconcurrency import Metal
@preconcurrency import RealityKit

private struct PreviewSourceVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
}

private struct PreviewVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
  var color: SIMD4<Float>
}

/// A RealityKit mesh whose real and imaginary canonical previews are updated
/// directly by a Metal compute kernel.
///
/// The preview treats mesh position as `q` and its scaled normal as the paired
/// field `p`, then displays `q′ = aq + bp`. It is a responsive canonical-map
/// aid; use `MetalLCTTransformer` for the actual sampled-field transform.
@MainActor
public final class MetalCanonicalPreview {
  public let lowLevelMesh: LowLevelMesh
  public let meshResource: MeshResource

  private let device: any MTLDevice
  private let commandQueue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState
  private let sourceBuffer: any MTLBuffer
  private let vertexCount: Int

  public init(
    geometry: MeshGeometry,
    device: (any MTLDevice)? = nil,
    boundsExtent: Float = 8
  ) throws {
    guard let resolvedDevice = device ?? MTLCreateSystemDefaultDevice() else {
      throw MetalLCTError.metalUnavailable
    }
    guard let commandQueue = resolvedDevice.makeCommandQueue() else {
      throw MetalLCTError.commandQueueUnavailable
    }
    let library = try MetalLCTTransformer.makeLibrary(device: resolvedDevice)
    guard let function = library.makeFunction(name: "canonical_preview_mesh") else {
      throw MetalLCTError.commandEncodingFailed
    }
    let pipeline = try resolvedDevice.makeComputePipelineState(function: function)

    let sourceVertices = zip(geometry.positions, geometry.normals).map {
      PreviewSourceVertex(position: $0.0, normal: $0.1)
    }
    let sourceByteCount = sourceVertices.count * MemoryLayout<PreviewSourceVertex>.stride
    guard
      let sourceBuffer = sourceVertices.withUnsafeBytes({ bytes in
        resolvedDevice.makeBuffer(
          bytes: bytes.baseAddress!,
          length: sourceByteCount,
          options: .storageModeShared
        )
      })
    else {
      throw MetalLCTError.bufferAllocationFailed
    }

    let attributes = [
      LowLevelMesh.Attribute(
        semantic: .position,
        format: .float3,
        offset: MemoryLayout<PreviewVertex>.offset(of: \.position)!
      ),
      LowLevelMesh.Attribute(
        semantic: .normal,
        format: .float3,
        offset: MemoryLayout<PreviewVertex>.offset(of: \.normal)!
      ),
      LowLevelMesh.Attribute(
        semantic: .color,
        format: .float4,
        offset: MemoryLayout<PreviewVertex>.offset(of: \.color)!
      ),
    ]
    let layouts = [
      LowLevelMesh.Layout(bufferIndex: 0, bufferStride: MemoryLayout<PreviewVertex>.stride)
    ]
    let descriptor = LowLevelMesh.Descriptor(
      vertexCapacity: sourceVertices.count * 2,
      vertexAttributes: attributes,
      vertexLayouts: layouts,
      indexCapacity: geometry.indices.count * 2,
      indexType: .uint32
    )
    let lowLevelMesh = try LowLevelMesh(descriptor: descriptor)
    lowLevelMesh.withUnsafeMutableIndices { rawIndices in
      let indices = rawIndices.bindMemory(to: UInt32.self)
      for index in geometry.indices.indices {
        indices[index] = geometry.indices[index]
        indices[index + geometry.indices.count] =
          geometry.indices[index] + UInt32(sourceVertices.count)
      }
    }
    let bounds = BoundingBox(
      min: SIMD3<Float>(repeating: -boundsExtent),
      max: SIMD3<Float>(repeating: boundsExtent)
    )
    lowLevelMesh.parts.append(
      .init(
        indexOffset: 0,
        indexCount: geometry.indices.count,
        topology: .triangle,
        materialIndex: 0,
        bounds: bounds
      )
    )
    lowLevelMesh.parts.append(
      .init(
        indexOffset: geometry.indices.count,
        indexCount: geometry.indices.count,
        topology: .triangle,
        materialIndex: 1,
        bounds: bounds
      )
    )

    self.device = resolvedDevice
    self.commandQueue = commandQueue
    self.pipeline = pipeline
    self.sourceBuffer = sourceBuffer
    vertexCount = sourceVertices.count
    self.lowLevelMesh = lowLevelMesh
    meshResource = try MeshResource(from: lowLevelMesh)

    try update(matrix: .identity)
  }

  public func update(
    matrix: SL2CMatrix,
    companionScale: Float = 0.18,
    showsImaginary: Bool = true
  ) throws {
    guard
      let commandBuffer = commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
      throw MetalLCTError.commandEncodingFailed
    }

    let outputBuffer = lowLevelMesh.replace(bufferIndex: 0, using: commandBuffer)
    var a = SIMD2<Float>(matrix.a.real, matrix.a.imaginary)
    var b = SIMD2<Float>(matrix.b.real, matrix.b.imaginary)
    var parameters = SIMD4<Float>(companionScale, showsImaginary ? 1 : 0, 0, 0)
    var count = UInt32(vertexCount)

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(sourceBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)
    encoder.setBytes(&a, length: MemoryLayout<SIMD2<Float>>.stride, index: 2)
    encoder.setBytes(&b, length: MemoryLayout<SIMD2<Float>>.stride, index: 3)
    encoder.setBytes(&parameters, length: MemoryLayout<SIMD4<Float>>.stride, index: 4)
    encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 5)

    let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
    encoder.dispatchThreads(
      MTLSize(width: vertexCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
  }
}
