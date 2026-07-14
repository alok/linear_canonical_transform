import Foundation
@_exported import LCTSpatial
@preconcurrency import Metal

public enum MetalLCTError: Error, Sendable {
  case metalUnavailable
  case shaderResourceMissing
  case commandQueueUnavailable
  case bufferAllocationFailed
  case commandEncodingFailed
  case commandFailed(String)
  case singularB
}

/// A Metal implementation of the dimension-agnostic separable LCT kernel.
///
/// One compute pass is dispatched per field axis, keeping a cubic field at
/// O(rank × n^(rank+1)) instead of evaluating every input/output voxel pair.
public final class MetalLCTTransformer: @unchecked Sendable {
  public let device: any MTLDevice

  private let commandQueue: any MTLCommandQueue
  private let axisPipeline: any MTLComputePipelineState

  public convenience init() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw MetalLCTError.metalUnavailable
    }
    try self.init(device: device)
  }

  public init(device: any MTLDevice) throws {
    self.device = device
    guard let commandQueue = device.makeCommandQueue() else {
      throw MetalLCTError.commandQueueUnavailable
    }
    self.commandQueue = commandQueue

    let library = try Self.makeLibrary(device: device)
    guard let function = library.makeFunction(name: "lct_axis_pass") else {
      throw MetalLCTError.commandEncodingFailed
    }
    axisPipeline = try device.makeComputePipelineState(function: function)
  }

  public func transform(
    _ field: ComplexField,
    matrix: SL2CMatrix,
    maximumKernelExponent: Float = 12
  ) throws -> ComplexField {
    if matrix == .identity { return field }
    guard matrix.b.magnitude > 1e-6 else { throw MetalLCTError.singularB }
    precondition(MemoryLayout<Complex32>.stride == MemoryLayout<SIMD2<Float>>.stride)

    let byteCount = field.count * MemoryLayout<Complex32>.stride
    guard
      let firstBuffer = field.values.withUnsafeBytes({ bytes in
        device.makeBuffer(
          bytes: bytes.baseAddress!,
          length: byteCount,
          options: .storageModeShared
        )
      }),
      let secondBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared),
      let commandBuffer = commandQueue.makeCommandBuffer()
    else {
      throw MetalLCTError.bufferAllocationFailed
    }

    var input = firstBuffer
    var output = secondBuffer
    var a = SIMD2<Float>(matrix.a.real, matrix.a.imaginary)
    var b = SIMD2<Float>(matrix.b.real, matrix.b.imaginary)
    var d = SIMD2<Float>(matrix.d.real, matrix.d.imaginary)
    var maximumExponent = maximumKernelExponent

    for axis in field.shape.indices {
      guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw MetalLCTError.commandEncodingFailed
      }
      let length = field.shape[axis]
      let elementStride = field.shape.dropFirst(axis + 1).reduce(1, *)
      let block = length * elementStride
      var dimensions = SIMD4<UInt32>(
        UInt32(length),
        UInt32(elementStride),
        UInt32(block),
        UInt32(field.count)
      )

      encoder.setComputePipelineState(axisPipeline)
      encoder.setBuffer(input, offset: 0, index: 0)
      encoder.setBuffer(output, offset: 0, index: 1)
      encoder.setBytes(&a, length: MemoryLayout<SIMD2<Float>>.stride, index: 2)
      encoder.setBytes(&b, length: MemoryLayout<SIMD2<Float>>.stride, index: 3)
      encoder.setBytes(&d, length: MemoryLayout<SIMD2<Float>>.stride, index: 4)
      encoder.setBytes(&dimensions, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 5)
      encoder.setBytes(&maximumExponent, length: MemoryLayout<Float>.stride, index: 6)

      let width = min(axisPipeline.maxTotalThreadsPerThreadgroup, 256)
      encoder.dispatchThreads(
        MTLSize(width: field.count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
      )
      encoder.endEncoding()
      swap(&input, &output)
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    guard commandBuffer.status == .completed else {
      throw MetalLCTError.commandFailed(
        commandBuffer.error?.localizedDescription ?? "unknown Metal error")
    }

    let pointer = input.contents().bindMemory(to: Complex32.self, capacity: field.count)
    let values = Array(UnsafeBufferPointer(start: pointer, count: field.count))
    return try ComplexField(shape: field.shape, values: values)
  }

  static func makeLibrary(device: any MTLDevice) throws -> any MTLLibrary {
    // Xcode compiles package Metal resources into the target's default
    // metallib. SwiftPM's command-line test runner instead leaves the source
    // available as a resource, so keep source compilation as a portable
    // fallback for library consumers and tests.
    if let library = try? device.makeDefaultLibrary(bundle: Bundle.module) {
      return library
    }

    let candidates = [
      Bundle.module.url(forResource: "LCTKernels", withExtension: "metal"),
      Bundle.module.url(
        forResource: "LCTKernels",
        withExtension: "metal",
        subdirectory: "Shaders"
      ),
    ]
    guard let url = candidates.compactMap({ $0 }).first else {
      throw MetalLCTError.shaderResourceMissing
    }
    let source = try String(contentsOf: url, encoding: .utf8)
    return try device.makeLibrary(source: source, options: nil)
  }
}
