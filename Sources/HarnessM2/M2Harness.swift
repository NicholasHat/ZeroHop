import CoreML
import Darwin
import Foundation
import Metal
import HarnessCore
import HarnessM1

public struct M2Config {
    public enum ReadPath: String { case A, B }          // E1
    public enum Granularity: String { case multi, seq } // E8

    public var modelPathK8: String
    public var modelPathK1: String
    public var readPath: ReadPath
    public var granularity: Granularity
    public var mlockAttempt: Bool   // E7 axis
    public var pressure: Bool       // E7 axis
    public var gpuWarm: String
    public var warmup: Int
    public var iterations: Int
    public var paceNS: UInt64 = 50_000_000

    public init(modelPathK8: String, modelPathK1: String, readPath: ReadPath,
                granularity: Granularity, mlockAttempt: Bool, pressure: Bool,
                gpuWarm: String, warmup: Int, iterations: Int) {
        self.modelPathK8 = modelPathK8
        self.modelPathK1 = modelPathK1
        self.readPath = readPath
        self.granularity = granularity
        self.mlockAttempt = mlockAttempt
        self.pressure = pressure
        self.gpuWarm = gpuWarm
        self.warmup = warmup
        self.iterations = iterations
    }
}

/// M2 — real buffer shapes + backing A/B (spec §10 M2, experiments E1/E2/E6/
/// E7/E8). Sync completion on an RT thread (the winning M1 arm); what varies
/// here is the transport: which zero-copy path is real, and is the ANE's DMA
/// write actually visible when the GPU kernel reads.
public enum M2Harness {

    // Mirrors the MSL structs in kernelSource.
    struct VerifyParams {
        var expectedBase: Float32  // < 0 disables the canary check
        var canaryOffset: UInt32
        var canaryRow: UInt32
        var rowStride: UInt32
        var rows: UInt32
    }
    struct VerifyResult {
        var canaryOK: UInt32
        var checksum: Float32
    }

    static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;
    struct VerifyParams {
        float expectedBase;
        uint  canaryOffset;
        uint  canaryRow;
        uint  rowStride;
        uint  rows;
    };
    struct VerifyResult { uint canaryOK; float checksum; };

    // E6 discipline: validate the canary tail BEFORE touching the payload,
    // then do a strided read across the whole tensor (the verify stand-in).
    kernel void verify_buf(device const half* logits [[buffer(0)]],
                           constant VerifyParams& p [[buffer(1)]],
                           device VerifyResult* res [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
        if (tid != 0) return;
        uint ok = 1;
        if (p.expectedBase >= 0.0f) {
            for (uint j = 0; j < 8; ++j) {
                float v = float(logits[p.canaryRow * p.rowStride + p.canaryOffset + j]);
                if (v != p.expectedBase + float(j)) ok = 0;
            }
        }
        float sum = 0.0f;
        for (uint r = 0; r < p.rows; ++r)
            for (uint c = 0; c < 16; ++c)
                sum += float(logits[r * p.rowStride + c * (p.rowStride / 16)]);
        res->canaryOK = ok;
        res->checksum = sum;
    }

    kernel void verify_tex(texture2d<half, access::read> logits [[texture(0)]],
                           constant VerifyParams& p [[buffer(1)]],
                           device VerifyResult* res [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
        if (tid != 0) return;
        uint ok = 1;
        if (p.expectedBase >= 0.0f) {
            for (uint j = 0; j < 8; ++j) {
                float v = float(logits.read(uint2(p.canaryOffset + j, p.canaryRow)).r);
                if (v != p.expectedBase + float(j)) ok = 0;
            }
        }
        float sum = 0.0f;
        for (uint r = 0; r < p.rows; ++r)
            for (uint c = 0; c < 16; ++c)
                sum += float(logits.read(uint2(c * (p.rowStride / 16), r)).r);
        res->canaryOK = ok;
        res->checksum = sum;
    }
    """

    static let ringDepth = 8
    static let canaryWidth = 8   // matches C in make_models.py
    static let featureDim = 512  // matches D

    @inline(__always)
    static func expectedBase(forValue v: UInt64) -> Float32 { Float32(v % 1024) }

    public static func run(resultsRoot: URL, config: M2Config) throws {
        let env = EnvInfo.capture()
        let cellName = "e1-\(config.readPath.rawValue)_e8-\(config.granularity.rawValue)"
            + "_mlock-\(config.mlockAttempt ? "on" : "off")_press-\(config.pressure ? "on" : "off")"
            + "_gw-\(config.gpuWarm)"
        print("M2 transport — \(env.chip), macOS \(env.macOSBuild), power: \(env.powerSource)")
        print("cell: \(cellName)  (warmup \(config.warmup), measured \(config.iterations))")

        // --- ANE lane ---------------------------------------------------------
        let seq = config.granularity == .seq
        let modelPath = seq ? config.modelPathK1 : config.modelPathK8
        let model = try M1Model.loadWithPlacementAssert(
            compiledModelURL: URL(fileURLWithPath: modelPath))
        guard let inputName = model.modelDescription.inputDescriptionsByName.keys.first,
              let outputName = model.modelDescription.outputDescriptionsByName.keys.first,
              let inShape = model.modelDescription.inputDescriptionsByName[inputName]?
                .multiArrayConstraint?.shape.map(\.intValue),
              let outShape = model.modelDescription.outputDescriptionsByName[outputName]?
                .multiArrayConstraint?.shape.map(\.intValue) else {
            throw HarnessError("model must have one multiarray input and one output")
        }
        let k = outShape.dropLast().reduce(1, *)   // rows
        let V = outShape.last!                     // row stride
        let callsPerCycle = seq ? 8 : 1

        let input = try MLMultiArray(shape: inShape.map { NSNumber(value: $0) }, dataType: .float16)
        input.withUnsafeMutableBytes { raw, _ in
            raw.bindMemory(to: Float16.self).update(repeating: 0.5)
        }
        let features = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(multiArray: input)])

        // --- E1 backings (ping-pong pair, spec §4.1) ---------------------------
        guard let device = MTLCreateSystemDefaultDevice() else { throw HarnessError("no Metal device") }
        let backings: [M2Backing]
        switch config.readPath {
        case .A: backings = [try IOSurfaceBacking(shape: outShape, device: device),
                             try IOSurfaceBacking(shape: outShape, device: device)]
        case .B: backings = [try BytesNoCopyBacking(shape: outShape, device: device),
                             try BytesNoCopyBacking(shape: outShape, device: device)]
        }
        let optionsPair = backings.map { backing -> MLPredictionOptions in
            let o = MLPredictionOptions()
            o.outputBackings = [outputName: backing.array]
            return o
        }

        var mlockNotes: [String] = []
        if config.mlockAttempt {
            mlockNotes = backings.map { "\($0.kindName): \($0.attemptMlock())" }
            mlockNotes.forEach { print("  E7 \($0)") }
        }

        // --- GPU lane with the real reader/canary kernel ------------------------
        let paramsBuffers = (0..<ringDepth).map { _ in
            device.makeBuffer(length: MemoryLayout<VerifyParams>.stride, options: .storageModeShared)!
        }
        let resultBuffers = (0..<ringDepth).map { _ in
            device.makeBuffer(length: MemoryLayout<VerifyResult>.stride, options: .storageModeShared)!
        }
        let gpu = try GPULane(
            ringDepth: ringDepth,
            kernelSource: kernelSource,
            kernelName: config.readPath == .A ? "verify_tex" : "verify_buf"
        ) { encoder, value, slotIndex in
            let params = paramsBuffers[slotIndex].contents()
                .bindMemory(to: VerifyParams.self, capacity: 1)
            params.pointee = VerifyParams(
                expectedBase: expectedBase(forValue: value),
                canaryOffset: UInt32(V - canaryWidth),
                canaryRow: UInt32(k - 1),
                rowStride: UInt32(V),
                rows: UInt32(k))
            backings[Int((value - 1) % 2)].bind(to: encoder)
            encoder.setBuffer(paramsBuffers[slotIndex], offset: 0, index: 1)
            encoder.setBuffer(resultBuffers[slotIndex], offset: 0, index: 2)
            encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        }
        try gpu.primeRing(startValue: 1)

        var stopGPUWarm: (() -> Void)?
        if config.gpuWarm != "none" {
            let periodNS: UInt64? = config.gpuWarm == "saturated" ? nil
                : UInt64(config.gpuWarm).map { $0 * 1_000_000 }
            stopGPUWarm = try gpu.startKeepWarm(periodNS: periodNS)
        }

        // --- E7 pressure helper -------------------------------------------------
        var pressureProc: Process?
        if config.pressure {
            pressureProc = try PressureHelper.spawn(megabytes: 6144)
            print("  E7 pressure helper pid \(pressureProc!.processIdentifier) (6 GiB touch loop)")
        }
        defer { pressureProc?.terminate() }

        // --- measured loop (sync arm on RT thread) ------------------------------
        let watcher = ThermalWatcher()
        watcher.start()
        let n = config.iterations
        let raw = Raw(count: n)
        var correlation = GPUClockCorrelation(device: device)

        let total = config.warmup + n
        var thrownError: Error?
        let paceTicks = MachClock.fromNanos(config.paceNS)
        let thread = DedicatedThread(policy: .timeConstraint) {
            var next = MachClock.now() + paceTicks
            for iter in 0..<total {
                mach_wait_until(next)
                let value = UInt64(iter + 1)
                let backingIdx = Int((value - 1) % 2)
                do {
                    // Write the canary pattern for this iteration into the
                    // input tail; the ANE matmul's identity block carries it
                    // into the logits tail (E6).
                    let base = expectedBase(forValue: value)
                    input.withUnsafeMutableBytes { rawBuf, _ in
                        let p = rawBuf.bindMemory(to: Float16.self)
                        let rowLen = featureDim + canaryWidth
                        for r in 0..<inShape.dropLast().reduce(1, *) {
                            for j in 0..<canaryWidth {
                                p[r * rowLen + featureDim + j] = Float16(base + Float32(j))
                            }
                        }
                    }

                    let t0 = MachClock.now()
                    var lastOut: MLFeatureProvider?
                    for _ in 0..<callsPerCycle {
                        lastOut = try model.prediction(from: features, options: optionsPair[backingIdx])
                    }
                    let t2 = MachClock.now()
                    gpu.event.signaledValue = value
                    let t5 = MachClock.now()

                    let done = try gpu.harvest(value: value)
                    let i = iter - config.warmup
                    if i >= 0 {
                        raw.t0[i] = t0; raw.t2[i] = t2; raw.t5[i] = t5
                        raw.gpuStartNS[i] = done.gpuStartTimeNS
                        raw.counterGPU[i] = done.counterStartGPU ?? 0
                        raw.thermal[i] = watcher.isNominal ? 1 : 0
                        let result = resultBuffers[gpu.slotIndex(for: value)].contents()
                            .load(as: VerifyResult.self)
                        if result.canaryOK != 1 { raw.canaryFailures += 1 }
                        if let arr = lastOut?.featureValue(for: outputName)?.multiArrayValue,
                           !backings[backingIdx].isHonored(by: arr) {
                            raw.backingViolations += 1
                        }
                    }
                } catch {
                    thrownError = error
                    return
                }
                next = MachClock.now() + paceTicks
            }
        }
        thread.join()
        correlation.finish(device: device)
        stopGPUWarm?()
        if let thrownError { throw thrownError }

        try report(config: config, cellName: cellName, env: env, raw: raw,
                   correlation: correlation, watcher: watcher, resultsRoot: resultsRoot,
                   mlockNotes: mlockNotes, callsPerCycle: callsPerCycle)
    }

    final class Raw {
        var t0, t2, t5, gpuStartNS, counterGPU: [UInt64]
        var thermal: [UInt8]
        var canaryFailures = 0
        var backingViolations = 0
        init(count: Int) {
            t0 = .init(repeating: 0, count: count)
            t2 = .init(repeating: 0, count: count)
            t5 = .init(repeating: 0, count: count)
            gpuStartNS = .init(repeating: 0, count: count)
            counterGPU = .init(repeating: 0, count: count)
            thermal = .init(repeating: 1, count: count)
        }
    }

    static func report(config: M2Config, cellName: String, env: EnvInfo, raw: Raw,
                       correlation: GPUClockCorrelation, watcher: ThermalWatcher,
                       resultsRoot: URL, mlockNotes: [String], callsPerCycle: Int) throws {
        let n = config.iterations
        let dispatch = SampleRecorder(name: "t0_t2_dispatch", capacity: n)
        let signal = SampleRecorder(name: "t2_t5_signal", capacity: n)
        let release = SampleRecorder(name: "t5_t6_release", capacity: n)
        let handoff = SampleRecorder(name: "t2_t6_handoff", capacity: n)

        var negativeRelease = 0
        for i in 0..<n {
            let ok = raw.thermal[i] == 1
            let t2ns = MachClock.toNanos(raw.t2[i])
            let t5ns = MachClock.toNanos(raw.t5[i])
            let t6ns = raw.counterGPU[i] != 0
                ? correlation.gpuToCPUNanos(raw.counterGPU[i]) : raw.gpuStartNS[i]
            dispatch.record(MachClock.toNanos(raw.t2[i] &- raw.t0[i]), thermalOK: ok)
            signal.record(MachClock.toNanos(raw.t5[i] &- raw.t2[i]), thermalOK: ok)
            if t6ns >= t5ns { release.record(t6ns - t5ns, thermalOK: ok) }
            else { negativeRelease += 1; release.record(0, thermalOK: false) }
            handoff.record(t6ns >= t2ns ? t6ns - t2ns : 0, thermalOK: ok)
        }

        let recorders = [dispatch, signal, release, handoff]
        let sink = try ResultSink(resultsRoot: resultsRoot, milestone: "m2", cellName: cellName)
        var meta = ResultSink.Meta(
            milestone: "m2",
            cell: ["E1.readPath": config.readPath.rawValue,
                   "E8.granularity": config.granularity.rawValue,
                   "E7.mlock": config.mlockAttempt ? "on" : "off",
                   "E7.pressure": config.pressure ? "on" : "off",
                   "gpu_warm": config.gpuWarm,
                   "calls_per_cycle": String(callsPerCycle),
                   "pace_ns": String(config.paceNS)],
            env: env, warmupIterations: config.warmup, measuredIterations: n)
        meta.thermalTimeline = watcher.stop()
        meta.clockCorrelation = [
            "before": ["cpu": correlation.before.cpu, "gpu": correlation.before.gpu],
            "after": ["cpu": correlation.after?.cpu ?? 0, "gpu": correlation.after?.gpu ?? 0],
        ]
        meta.notes.append(raw.canaryFailures == 0
            ? "E6: canary valid on all \(n) iterations — no stale read observed"
            : "E6 STALE READ: canary mismatch on \(raw.canaryFailures)/\(n) iterations")
        meta.notes.append(raw.backingViolations == 0
            ? "E2: outputBacking (\(config.readPath.rawValue)) honored on all iterations"
            : "E2 WARNING: backing NOT honored on \(raw.backingViolations)/\(n) iterations — hidden copy path")
        meta.notes.append(contentsOf: mlockNotes.map { "E7 \($0)" })
        if negativeRelease > 0 {
            meta.notes.append("negative t5->t6 after clock mapping on \(negativeRelease) iterations")
        }
        try sink.writeMeta(meta)
        try sink.writeSamplesCSV(iterationsOf: recorders)
        try sink.writeSummaries(recorders.map { $0.summarize() })

        print("")
        printSummaryTable(recorders.map { $0.summarize() })
        for note in meta.notes { print("  note: \(note)") }
        print("\nhandoff (t2'→t6) distribution:")
        print(handoff.bucketRender())
        print("-> \(sink.dir.path)")
    }
}

/// E7 helper: child process that allocates and continuously touches memory to
/// generate sustained pressure. Runs our own binary with the hidden
/// `pressure-helper` subcommand; terminated by the parent.
public enum PressureHelper {
    public static func spawn(megabytes: Int) throws -> Process {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
        p.arguments = ["pressure-helper", "--mb", String(megabytes)]
        try p.run()
        return p
    }

    public static func runLoop(megabytes: Int) -> Never {
        let chunkMB = 256
        var chunks: [UnsafeMutableRawPointer] = []
        for _ in 0..<(megabytes / chunkMB) {
            guard let c = malloc(chunkMB * 1_048_576) else { break }
            memset(c, 0xA5, chunkMB * 1_048_576)
            chunks.append(c)
        }
        // Keep the pages hot so the pager can't quietly reclaim them.
        var pass: UInt8 = 0
        while true {
            for c in chunks { memset(c, Int32(pass), 4096) ; memset(c + chunkMB * 524_288, Int32(pass), 4096) }
            pass &+= 1
            usleep(50_000)
        }
    }
}
