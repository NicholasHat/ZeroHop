import CoreML
import Darwin
import Foundation
import Metal
import HarnessCore
import os

public struct M1Config {
    public enum CompletionStyle: String {
        case sync    // blocking prediction() on the pinned thread (E3 hypothesis: wins)
        case async   // completion handler does only a semaphore signal to the dedicated handoff thread
        case naive   // completion handler signals the MTLSharedEvent itself, on CoreML's callback queue
    }

    public var modelPath: String
    public var completionStyle: CompletionStyle
    public var policy: ThreadPolicy
    public var warmth: String        // E5: none | 500 | 100 | 50 | 10 | saturated (ms period)
    public var gpuWarm: String = "none" // GPU-side keep-warm: none | <ms period> | saturated
    public var warmup: Int
    public var iterations: Int
    public var paceNS: UInt64 = 50_000_000  // 20 Hz measured dispatch rate (PLAN §4)

    public init(modelPath: String, completionStyle: CompletionStyle, policy: ThreadPolicy,
                warmth: String, warmup: Int, iterations: Int) {
        self.modelPath = modelPath
        self.completionStyle = completionStyle
        self.policy = policy
        self.warmth = warmth
        self.warmup = warmup
        self.iterations = iterations
    }
}

/// M1 — empty-model round-trip harness (spec §10 M1): trivial ANE-resident
/// model + no-op GPU kernel behind an MTLSharedEvent. Measures t0→t2′ (the
/// dispatch figure to re-derive) and t2′→t6 (the handoff) with sub-segments.
public enum M1Harness {

    // Per-iteration raw values; converted/correlated after the run so the
    // measured loop only does indexed stores.
    final class RawStore {
        var t0, t2, t4, t5: [UInt64]        // mach ticks
        var gpuStartNS, counterGPU: [UInt64] // host-ns / GPU-clock (0 = absent)
        var thermal: [UInt8]
        var backingViolations = 0
        init(count: Int) {
            t0 = .init(repeating: 0, count: count)
            t2 = .init(repeating: 0, count: count)
            t4 = .init(repeating: 0, count: count)
            t5 = .init(repeating: 0, count: count)
            gpuStartNS = .init(repeating: 0, count: count)
            counterGPU = .init(repeating: 0, count: count)
            thermal = .init(repeating: 1, count: count)
        }
    }

    public static func run(resultsRoot: URL, config: M1Config) throws {
        let env = EnvInfo.capture()
        let cellName = "e3-\(config.completionStyle.rawValue)_e4-\(config.policy.rawValue)_e5-\(config.warmth)_gw-\(config.gpuWarm)"
        print("M1 round-trip — \(env.chip), macOS \(env.macOSBuild), power: \(env.powerSource)")
        print("cell: \(cellName)  (warmup \(config.warmup), measured \(config.iterations))")

        // --- ANE lane -------------------------------------------------------
        let modelURL = try resolveCompiledModel(path: config.modelPath)
        let model = try M1Model.loadWithPlacementAssert(compiledModelURL: modelURL)

        guard let (inputName, inputShape) = model.modelDescription.inputDescriptionsByName
                .first.map({ ($0.key, $0.value.multiArrayConstraint!.shape.map(\.intValue)) }),
              let outputName = model.modelDescription.outputDescriptionsByName.keys.first else {
            throw HarnessError("model must have one multiarray input and one output")
        }

        let input = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float16)
        input.withUnsafeMutableBytes { raw, _ in
            raw.bindMemory(to: Float16.self).update(repeating: 0.5)
        }
        let features = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(multiArray: input)])

        let outputShape = model.modelDescription.outputDescriptionsByName[outputName]?
            .multiArrayConstraint?.shape.map(\.intValue) ?? inputShape
        let backing = try AlignedBacking(shape: outputShape)
        let options = MLPredictionOptions()
        options.outputBackings = [outputName: backing.array]

        // --- GPU lane -------------------------------------------------------
        let gpu = try GPULane(ringDepth: 8)
        try gpu.primeRing(startValue: 1)
        print("counter sampling (stage boundary): \(gpu.hasCounterSampling ? "available" : "ABSENT — t6 falls back to gpuStartTime")")

        // --- E5 warmth heartbeat ---------------------------------------------
        let heartbeatStop = UnsafeMutablePointer<Bool>.allocate(capacity: 1)
        heartbeatStop.initialize(to: false)
        defer { heartbeatStop.deallocate() }
        var heartbeat: DedicatedThread?
        if config.warmth != "none" {
            let periodNS: UInt64? = config.warmth == "saturated" ? nil
                : UInt64(config.warmth).map { $0 * 1_000_000 }
            let hbInput = try MLMultiArray(shape: inputShape.map { NSNumber(value: $0) }, dataType: .float16)
            let hbFeatures = try MLDictionaryFeatureProvider(
                dictionary: [inputName: MLFeatureValue(multiArray: hbInput)])
            heartbeat = DedicatedThread(policy: .default) {
                while !heartbeatStop.pointee {
                    _ = try? model.prediction(from: hbFeatures)
                    if let periodNS {
                        mach_wait_until(MachClock.now() + MachClock.fromNanos(periodNS))
                    }
                }
            }
        }

        // --- GPU keep-warm ----------------------------------------------------
        var stopGPUWarm: (() -> Void)?
        if config.gpuWarm != "none" {
            let periodNS: UInt64? = config.gpuWarm == "saturated" ? nil
                : UInt64(config.gpuWarm).map { $0 * 1_000_000 }
            stopGPUWarm = try gpu.startKeepWarm(periodNS: periodNS)
        }

        // --- measured run -----------------------------------------------------
        let watcher = ThermalWatcher()
        watcher.start()
        let raw = RawStore(count: config.iterations)
        var correlation = GPUClockCorrelation(device: gpu.device)

        switch config.completionStyle {
        case .sync:
            try runSync(config: config, model: model, features: features, options: options,
                        outputName: outputName, backing: backing, gpu: gpu, raw: raw, watcher: watcher)
        case .async, .naive:
            try runAsync(config: config, model: model, features: features, options: options,
                         outputName: outputName, backing: backing, gpu: gpu, raw: raw, watcher: watcher)
        }

        correlation.finish(device: gpu.device)
        stopGPUWarm?()
        heartbeatStop.pointee = true
        heartbeat?.join()

        try report(config: config, cellName: cellName, env: env, raw: raw,
                   correlation: correlation, watcher: watcher, resultsRoot: resultsRoot,
                   counterAvailable: gpu.hasCounterSampling)
    }

    // MARK: - E3 sync arm: everything on the one pinned thread

    private static func runSync(config: M1Config, model: MLModel, features: MLFeatureProvider,
                                options: MLPredictionOptions, outputName: String,
                                backing: AlignedBacking, gpu: GPULane, raw: RawStore,
                                watcher: ThermalWatcher) throws {
        let total = config.warmup + config.iterations
        var thrownError: Error?
        let paceTicks = MachClock.fromNanos(config.paceNS)
        let spid = OSSignpostID(log: Signposts.log)

        let thread = DedicatedThread(policy: config.policy) {
            var next = MachClock.now() + paceTicks
            for iter in 0..<total {
                mach_wait_until(next)
                let value = UInt64(iter + 1)
                do {
                    os_signpost(.begin, log: Signposts.log, name: "ane_dispatch", signpostID: spid)
                    let t0 = MachClock.now()
                    let out = try model.prediction(from: features, options: options)
                    let t2 = MachClock.now() // sync: completion observed == thread already awake
                    os_signpost(.end, log: Signposts.log, name: "ane_dispatch", signpostID: spid)
                    os_signpost(.begin, log: Signposts.log, name: "handoff", signpostID: spid)
                    gpu.event.signaledValue = value
                    let t5 = MachClock.now()
                    os_signpost(.end, log: Signposts.log, name: "handoff", signpostID: spid)

                    let done = try gpu.harvest(value: value)
                    let i = iter - config.warmup
                    if i >= 0 {
                        raw.t0[i] = t0; raw.t2[i] = t2; raw.t4[i] = t2; raw.t5[i] = t5
                        raw.gpuStartNS[i] = done.gpuStartTimeNS
                        raw.counterGPU[i] = done.counterStartGPU ?? 0
                        raw.thermal[i] = watcher.isNominal ? 1 : 0
                        if let arr = out.featureValue(for: outputName)?.multiArrayValue,
                           !backing.isHonored(by: arr) {
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
        if let thrownError { throw thrownError }
    }

    // MARK: - E3 async + naive arms

    private static func runAsync(config: M1Config, model: MLModel, features: MLFeatureProvider,
                                 options: MLPredictionOptions, outputName: String,
                                 backing: AlignedBacking, gpu: GPULane, raw: RawStore,
                                 watcher: ThermalWatcher) throws {
        let total = config.warmup + config.iterations
        let naive = config.completionStyle == .naive
        let paceTicks = MachClock.fromNanos(config.paceNS)

        // Mailbox between completion handler / handoff thread / dispatcher.
        // Semaphores provide the ordering edges; iterations are lock-step.
        final class Mailbox {
            var t2: UInt64 = 0
            var t4: UInt64 = 0
            var t5: UInt64 = 0
            var value: UInt64 = 0
            var backingOK = true
            let wake = MachSemaphore() // handler -> handoff thread
            let done = MachSemaphore() // handoff/handler -> dispatcher
            var stop = false
        }
        let box = Mailbox()

        // Dedicated handoff pthread (spec §4.2) — only in the non-naive arm.
        var handoffThread: DedicatedThread?
        if !naive {
            handoffThread = DedicatedThread(policy: config.policy) {
                while true {
                    box.wake.wait()
                    if box.stop { box.done.signal(); return }
                    box.t4 = MachClock.now()
                    gpu.event.signaledValue = box.value
                    box.t5 = MachClock.now()
                    box.done.signal()
                }
            }
        }

        var thrownError: Error?
        let dispatcher = DedicatedThread(policy: naive ? config.policy : .default) {
            var next = MachClock.now() + paceTicks
            for iter in 0..<total {
                mach_wait_until(next)
                let value = UInt64(iter + 1)
                box.value = value
                let t0 = MachClock.now()
                // NS_REFINED_FOR_SWIFT: the completion-handler form is only
                // reachable via the __-prefixed name; the Swift-refined async
                // variant would put concurrency-executor hops on the very path
                // E3 measures.
                model.__prediction(fromFeatures: features, options: options) { out, err in
                    // Runs on CoreML's completion queue (the libdispatch hop
                    // under test). Naive: do the handoff here. Async: wake the
                    // dedicated thread and nothing else.
                    box.t2 = MachClock.now()
                    if err != nil { box.stop = true }
                    if let out, let arr = out.featureValue(for: outputName)?.multiArrayValue {
                        box.backingOK = backing.isHonored(by: arr)
                    }
                    if naive {
                        box.t4 = box.t2
                        gpu.event.signaledValue = value
                        box.t5 = MachClock.now()
                        box.done.signal()
                    } else {
                        box.wake.signal()
                    }
                }
                box.done.wait()
                if box.stop {
                    thrownError = HarnessError("async prediction failed at iteration \(iter)")
                    if !naive { box.wake.signal() }
                    return
                }
                do {
                    let doneInfo = try gpu.harvest(value: value)
                    let i = iter - config.warmup
                    if i >= 0 {
                        raw.t0[i] = t0; raw.t2[i] = box.t2; raw.t4[i] = box.t4; raw.t5[i] = box.t5
                        raw.gpuStartNS[i] = doneInfo.gpuStartTimeNS
                        raw.counterGPU[i] = doneInfo.counterStartGPU ?? 0
                        raw.thermal[i] = watcher.isNominal ? 1 : 0
                        if !box.backingOK { raw.backingViolations += 1 }
                    }
                } catch {
                    thrownError = error
                    if !naive { box.stop = true; box.wake.signal() }
                    return
                }
                next = MachClock.now() + paceTicks
            }
            if !naive { box.stop = true; box.wake.signal() }
        }

        dispatcher.join()
        handoffThread?.join()
        if let thrownError { throw thrownError }
    }

    // MARK: - post-run conversion + report

    private static func report(config: M1Config, cellName: String, env: EnvInfo, raw: RawStore,
                               correlation: GPUClockCorrelation, watcher: ThermalWatcher,
                               resultsRoot: URL, counterAvailable: Bool) throws {
        let n = config.iterations
        let dispatch = SampleRecorder(name: "t0_t2_dispatch", capacity: n)
        let wake = SampleRecorder(name: "t2_t4_wake", capacity: n)
        let signal = SampleRecorder(name: "t4_t5_signal", capacity: n)
        let release = SampleRecorder(name: "t5_t6_release", capacity: n)
        let handoff = SampleRecorder(name: "t2_t6_handoff", capacity: n)

        if ProcessInfo.processInfo.environment["ZEROHOP_DEBUG_CLOCKS"] != nil, n > 0 {
            print("DEBUG clocks: corr.before=(cpu:\(correlation.before.cpu), gpu:\(correlation.before.gpu)) "
                + "corr.after=(cpu:\(correlation.after?.cpu ?? 0), gpu:\(correlation.after?.gpu ?? 0))")
            print("DEBUG iter0: t5_ticks=\(raw.t5[0]) t5_ns=\(MachClock.toNanos(raw.t5[0])) "
                + "counterGPU=\(raw.counterGPU[0]) gpuStartNS=\(raw.gpuStartNS[0])")
        }
        var negativeRelease = 0
        var counterUsed = 0
        for i in 0..<n {
            let ok = raw.thermal[i] == 1
            let t2ns = MachClock.toNanos(raw.t2[i])
            let t5ns = MachClock.toNanos(raw.t5[i])
            let t6ns: UInt64
            if raw.counterGPU[i] != 0 {
                t6ns = correlation.gpuToCPUNanos(raw.counterGPU[i])
                counterUsed += 1
            } else {
                t6ns = raw.gpuStartNS[i]
            }
            dispatch.record(MachClock.toNanos(raw.t2[i] &- raw.t0[i]), thermalOK: ok)
            wake.record(MachClock.toNanos(raw.t4[i] &- raw.t2[i]), thermalOK: ok)
            signal.record(MachClock.toNanos(raw.t5[i] &- raw.t4[i]), thermalOK: ok)
            if t6ns >= t5ns {
                release.record(t6ns - t5ns, thermalOK: ok)
            } else {
                negativeRelease += 1
                release.record(0, thermalOK: false)
            }
            handoff.record(t6ns >= t2ns ? t6ns - t2ns : 0, thermalOK: ok)
        }

        let recorders = [dispatch, wake, signal, release, handoff]
        let sink = try ResultSink(resultsRoot: resultsRoot, milestone: "m1", cellName: cellName)
        var meta = ResultSink.Meta(
            milestone: "m1",
            cell: ["E3.completion": config.completionStyle.rawValue,
                   "E4.policy": config.policy.rawValue,
                   "E5.warmth": config.warmth,
                   "gpu_warm": config.gpuWarm,
                   "pace_ns": String(config.paceNS)],
            env: env, warmupIterations: config.warmup, measuredIterations: n)
        meta.thermalTimeline = watcher.stop()
        meta.clockCorrelation = [
            "before": ["cpu": correlation.before.cpu, "gpu": correlation.before.gpu],
            "after": ["cpu": correlation.after?.cpu ?? 0, "gpu": correlation.after?.gpu ?? 0],
        ]
        meta.notes.append("t6 source: counter for \(counterUsed)/\(n), gpuStartTime otherwise (counter sampling \(counterAvailable ? "available" : "absent"))")
        if negativeRelease > 0 {
            meta.notes.append("negative t5->t6 after clock mapping on \(negativeRelease) iterations (excluded from release histogram)")
        }
        if raw.backingViolations > 0 {
            meta.notes.append("E2 WARNING: outputBacking NOT honored on \(raw.backingViolations)/\(n) iterations — CoreML used the hidden-copy path")
        } else {
            meta.notes.append("E2: outputBacking pointer identity held on all measured iterations")
        }
        try sink.writeMeta(meta)
        try sink.writeSamplesCSV(iterationsOf: recorders)
        try sink.writeSummaries(recorders.map { $0.summarize() })

        print("")
        printSummaryTable(recorders.map { $0.summarize() })
        for note in meta.notes { print("  note: \(note)") }
        print("\nhandoff (t2'→t6) distribution:")
        print(handoff.bucketRender())

        let p99 = handoff.summarize().p99
        print(p99 > 2_000_000
            ? "KILL CRITERION CHECK: p99 handoff \(SampleRecorder.fmt(p99)) EXCEEDS ~2 ms for this cell (spec §10 M1) — compare across mitigation cells before verdict"
            : "kill criterion check: p99 handoff \(SampleRecorder.fmt(p99)) is within the ~2 ms budget for this cell")
        print("-> \(sink.dir.path)")
    }

    private static func resolveCompiledModel(path: String) throws -> URL {
        let url = URL(fileURLWithPath: path)
        if url.pathExtension == "mlmodelc" {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw HarnessError("compiled model not found at \(url.path) — run Tools/make_models.py then `xcrun coremlcompiler compile`")
            }
            return url
        }
        if url.pathExtension == "mlpackage" {
            print("compiling \(url.lastPathComponent)…")
            return try MLModel.compileModel(at: url)
        }
        throw HarnessError("expected .mlmodelc or .mlpackage, got \(path)")
    }
}
