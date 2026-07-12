import Foundation
import Metal
import HarnessCore

/// GPU lane (spec §4.3): a ring of pre-encoded, pre-committed command buffers,
/// each parked on `encodeWait(sharedEvent, value: n)`. The handoff thread's
/// only GPU-facing action is writing `signaledValue = n`; the released command
/// buffer's kernel-start timestamp is t6.
public final class GPULane {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let event: MTLSharedEvent
    private let pipeline: MTLComputePipelineState
    private let srcBuffer: MTLBuffer
    private let sinkBuffer: MTLBuffer
    private let counterSet: MTLCounterSet?

    public struct Completed {
        public let gpuStartTimeNS: UInt64      // MTLCommandBuffer.gpuStartTime (host-time domain)
        public let counterStartGPU: UInt64?    // stage-boundary counter sample (GPU clock domain)
    }

    private struct Slot {
        var commandBuffer: MTLCommandBuffer
        var sampleBuffer: MTLCounterSampleBuffer?
        var value: UInt64
    }

    private var ring: [Slot] = []
    private let ringDepth: Int

    static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;
    // No-op verify stand-in: reads a word so the event wait cannot be elided,
    // writes a sink so the dispatch is observable. Replaced by the real
    // reader/canary kernel at M2.
    kernel void verify_noop(device const uint* src [[buffer(0)]],
                            device atomic_uint* sink [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
        if (tid == 0) {
            atomic_fetch_add_explicit(sink, src[0] + 1u, memory_order_relaxed);
        }
    }
    """

    public init(ringDepth: Int = 8) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw HarnessError("no Metal device")
        }
        device = dev
        guard let q = dev.makeCommandQueue() else { throw HarnessError("no command queue") }
        queue = q
        guard let ev = dev.makeSharedEvent() else { throw HarnessError("no shared event") }
        event = ev

        // Compiled at startup from source — off the critical path, no build step.
        let library = try dev.makeLibrary(source: Self.kernelSource, options: nil)
        pipeline = try dev.makeComputePipelineState(function: library.makeFunction(name: "verify_noop")!)
        srcBuffer = dev.makeBuffer(length: 16, options: .storageModeShared)!
        sinkBuffer = dev.makeBuffer(length: 16, options: .storageModeShared)!

        // Apple GPUs expose stage-boundary sampling only; assert and degrade
        // to gpuStartTime-based t6 (recorded in metadata) if absent.
        if dev.supportsCounterSampling(.atStageBoundary),
           let set = (dev.counterSets ?? []).first(where: { $0.name.lowercased().contains("timestamp") }) {
            counterSet = set
        } else {
            counterSet = nil
        }
        self.ringDepth = ringDepth
    }

    public var hasCounterSampling: Bool { counterSet != nil }

    /// Pre-commit command buffers waiting on values startValue..<startValue+depth.
    public func primeRing(startValue: UInt64) throws {
        ring.removeAll()
        for i in 0..<UInt64(ringDepth) {
            ring.append(try makeSlot(value: startValue + i))
        }
    }

    private func makeSlot(value: UInt64) throws -> Slot {
        var sampleBuffer: MTLCounterSampleBuffer?
        let passDesc = MTLComputePassDescriptor()
        if let counterSet {
            let desc = MTLCounterSampleBufferDescriptor()
            desc.counterSet = counterSet
            desc.storageMode = .shared
            desc.sampleCount = 2
            sampleBuffer = try device.makeCounterSampleBuffer(descriptor: desc)
            let att = passDesc.sampleBufferAttachments[0]!
            att.sampleBuffer = sampleBuffer
            att.startOfEncoderSampleIndex = 0
            att.endOfEncoderSampleIndex = 1
        }

        guard let cb = queue.makeCommandBuffer() else { throw HarnessError("command buffer") }
        cb.encodeWaitForEvent(event, value: value)
        guard let enc = cb.makeComputeCommandEncoder(descriptor: passDesc) else {
            throw HarnessError("compute encoder")
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(srcBuffer, offset: 0, index: 0)
        enc.setBuffer(sinkBuffer, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        return Slot(commandBuffer: cb, sampleBuffer: sampleBuffer, value: value)
    }

    /// GPU keep-warm (hypothesis from the M1 sweep: t5→t6 release latency of
    /// ~650 µs p50 is the GPU waking from idle, since one tiny kernel every
    /// 50 ms lets it power-gate — the GPU-side analog of E5). A background
    /// thread trickles trivial dispatches on a SEPARATE queue so the measured
    /// queue's pre-committed buffers are untouched.
    /// periodNS nil = saturated (back-to-back).
    public func startKeepWarm(periodNS: UInt64?) throws -> () -> Void {
        guard let warmQueue = device.makeCommandQueue() else {
            throw HarnessError("no keep-warm command queue")
        }
        let stopFlag = UnsafeMutablePointer<Bool>.allocate(capacity: 1)
        stopFlag.initialize(to: false)
        let pipeline = self.pipeline
        let src = self.srcBuffer
        let sink = self.sinkBuffer
        let thread = DedicatedThread(policy: .default) {
            while !stopFlag.pointee {
                guard let cb = warmQueue.makeCommandBuffer(),
                      let enc = cb.makeComputeCommandEncoder() else { break }
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(src, offset: 0, index: 0)
                enc.setBuffer(sink, offset: 0, index: 1)
                enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                if let periodNS {
                    mach_wait_until(MachClock.now() + MachClock.fromNanos(periodNS))
                }
            }
        }
        return {
            stopFlag.pointee = true
            thread.join()
            stopFlag.deallocate()
        }
    }

    /// Block until the command buffer for `value` completes, harvest its GPU
    /// timestamps, and refill the ring (both off the measured segment, which
    /// ends at kernel start).
    public func harvest(value: UInt64) throws -> Completed {
        guard let idx = ring.firstIndex(where: { $0.value == value }) else {
            throw HarnessError("ring underrun: no slot for value \(value)")
        }
        let slot = ring[idx]
        slot.commandBuffer.waitUntilCompleted()
        if slot.commandBuffer.status == .error {
            throw HarnessError("command buffer error: \(String(describing: slot.commandBuffer.error))")
        }

        var counterStart: UInt64?
        if let sb = slot.sampleBuffer,
           let data = try? sb.resolveCounterRange(0..<2) {
            data.withUnsafeBytes { raw in
                let stamps = raw.bindMemory(to: MTLCounterResultTimestamp.self)
                if stamps.count >= 1, stamps[0].timestamp != 0, stamps[0].timestamp != .max {
                    counterStart = stamps[0].timestamp
                }
            }
        }
        let completed = Completed(
            gpuStartTimeNS: UInt64(slot.commandBuffer.gpuStartTime * 1e9),
            counterStartGPU: counterStart
        )
        ring[idx] = try makeSlot(value: value + UInt64(ringDepth))
        return completed
    }
}
