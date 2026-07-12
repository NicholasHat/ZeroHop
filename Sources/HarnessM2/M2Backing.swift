import CoreML
import CoreVideo
import Foundation
import IOSurface
import Metal
import HarnessCore
import HarnessM1

/// One draft-logits output backing plus its zero-copy GPU view (experiment E1).
/// Two implementations — the A/B under test:
///   Option A: IOSurface-backed CVPixelBuffer → MLMultiArray(pixelBuffer:) on
///             the CoreML side, MTLTexture(iosurface:) on the Metal side.
///   Option B: page-aligned malloc → MLMultiArray(dataPointer:) on the CoreML
///             side, makeBuffer(bytesNoCopy:) on the Metal side.
public protocol M2Backing: AnyObject {
    var array: MLMultiArray { get }
    var kindName: String { get }
    /// Bind the GPU-side view of this backing (texture(0) or buffer(0)).
    func bind(to encoder: MTLComputeCommandEncoder)
    /// E2: was this backing honored, or did CoreML substitute an internal
    /// buffer (silent copy path)?
    func isHonored(by output: MLMultiArray) -> Bool
    /// E7: attempt to mlock the CPU-visible mapping; returns a result note.
    func attemptMlock() -> String
}

/// Option B — spec §5.
public final class BytesNoCopyBacking: M2Backing {
    private let aligned: AlignedBacking
    public let buffer: MTLBuffer
    public var array: MLMultiArray { aligned.array }
    public let kindName = "B-bytesNoCopy"

    public init(shape: [Int], device: MTLDevice) throws {
        aligned = try AlignedBacking(shape: shape)
        guard let buf = device.makeBuffer(
            bytesNoCopy: aligned.pointer, length: aligned.byteCount,
            options: .storageModeShared, deallocator: nil) else {
            throw HarnessError("makeBuffer(bytesNoCopy:) rejected the backing memory")
        }
        buffer = buf
    }

    public func bind(to encoder: MTLComputeCommandEncoder) {
        encoder.setBuffer(buffer, offset: 0, index: 0)
    }

    public func isHonored(by output: MLMultiArray) -> Bool {
        aligned.isHonored(by: output)
    }

    public func attemptMlock() -> String {
        let rc = mlock(aligned.pointer, aligned.byteCount)
        return rc == 0 ? "mlock ok (\(aligned.byteCount) bytes)" : "mlock failed errno=\(errno)"
    }
}

/// Option A — spec §5. Shape must be [..., rows, width] with width equal to
/// the pixel-buffer width (Float16 ⇒ kCVPixelFormatType_OneComponent16Half).
public final class IOSurfaceBacking: M2Backing {
    public let array: MLMultiArray
    public let texture: MTLTexture
    private let pixelBuffer: CVPixelBuffer
    private let surfaceID: IOSurfaceID
    public let kindName = "A-iosurface"

    public init(shape: [Int], device: MTLDevice) throws {
        let width = shape.last!
        let height = shape.dropLast().reduce(1, *)
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary]
        let cvRet = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                        kCVPixelFormatType_OneComponent16Half,
                                        attrs as CFDictionary, &pb)
        guard cvRet == kCVReturnSuccess, let pb else {
            throw HarnessError("CVPixelBufferCreate failed: \(cvRet)")
        }
        guard let unmanagedSurface = CVPixelBufferGetIOSurface(pb) else {
            throw HarnessError("pixel buffer is not IOSurface-backed")
        }
        let surface = unmanagedSurface.takeUnretainedValue()
        surfaceID = IOSurfaceGetID(surface)
        pixelBuffer = pb
        array = MLMultiArray(pixelBuffer: pb, shape: shape.map { NSNumber(value: $0) })

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float, width: width, height: height, mipmapped: false)
        desc.usage = .shaderRead
        desc.storageMode = .shared
        guard let tex = device.makeTexture(descriptor: desc, iosurface: surface, plane: 0) else {
            throw HarnessError("makeTexture(descriptor:iosurface:plane:) failed for \(width)x\(height)")
        }
        texture = tex
    }

    public func bind(to encoder: MTLComputeCommandEncoder) {
        encoder.setTexture(texture, index: 0)
    }

    public func isHonored(by output: MLMultiArray) -> Bool {
        guard let outPB = output.pixelBuffer,
              let outSurface = CVPixelBufferGetIOSurface(outPB) else { return false }
        return IOSurfaceGetID(outSurface.takeUnretainedValue()) == surfaceID
    }

    public func attemptMlock() -> String {
        // Spec §9 item 5: expected redundant/ineffective for IOSurface memory;
        // attempted once and the outcome recorded, never relied on.
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return "mlock n/a: no base address"
        }
        let len = CVPixelBufferGetDataSize(pixelBuffer)
        let rc = mlock(base, len)
        return rc == 0 ? "mlock ok on IOSurface mapping (\(len) bytes)" : "mlock failed errno=\(errno)"
    }
}
