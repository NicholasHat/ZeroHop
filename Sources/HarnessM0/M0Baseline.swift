import Darwin
import Foundation
import HarnessCore

/// M0 — baseline process-jitter noise floor (spec §8, §10 M0).
///
/// Two sub-benchmarks, each swept over the E4 thread policies:
///  - timer: pinned thread wakes from mach_wait_until and reads the clock;
///    sample = actual - target (pure scheduler wake jitter).
///  - xwake: a signaler thread stores a timestamp then signals a raw mach
///    semaphore; the measured thread parked in semaphore_wait wakes and reads
///    the clock; sample = wake - signal. Direct analog of stage 4 of the
///    critical path ("ANE completion notification -> handoff thread awake").
///
/// Every M1+ handoff number is interpreted as a delta over these histograms.
public enum M0Baseline {

    public static func run(resultsRoot: URL, warmup: Int, iterations: Int) throws {
        let env = EnvInfo.capture()
        print("M0 baseline noise floor — \(env.chip), macOS \(env.macOSBuild), power: \(env.powerSource)")
        if env.powerSource != "AC" {
            print("  WARNING: not on AC power; recorded in metadata (spec protocol expects plugged in)")
        }

        for policy in ThreadPolicy.allCases {
            try runTimerCell(policy: policy, resultsRoot: resultsRoot, env: env,
                             warmup: warmup, iterations: iterations)
            try runCrossWakeCell(policy: policy, resultsRoot: resultsRoot, env: env,
                                 warmup: warmup, iterations: iterations)
        }
    }

    // MARK: timer wake jitter

    private static func runTimerCell(policy: ThreadPolicy, resultsRoot: URL, env: EnvInfo,
                                     warmup: Int, iterations: Int) throws {
        let recorder = SampleRecorder(name: "timer_wake_jitter", capacity: iterations)
        let watcher = ThermalWatcher()
        watcher.start()

        let periodTicks = MachClock.fromNanos(1_000_000)
        let thread = DedicatedThread(policy: policy) {
            var target = MachClock.now() + periodTicks
            for i in 0..<(warmup + iterations) {
                mach_wait_until(target)
                let actual = MachClock.now()
                let jitter = actual >= target ? MachClock.toNanos(actual - target) : 0
                if i >= warmup {
                    recorder.record(jitter, thermalOK: watcher.isNominal)
                }
                target = actual + periodTicks
            }
        }
        thread.join()

        try finishCell(milestone: "m0", cellName: "timer-\(policy.rawValue)",
                       cell: ["benchmark": "timer", "E4.policy": policy.rawValue],
                       recorders: [recorder], env: env, watcher: watcher,
                       warmup: warmup, iterations: iterations, resultsRoot: resultsRoot)
    }

    // MARK: cross-thread wake latency

    private static func runCrossWakeCell(policy: ThreadPolicy, resultsRoot: URL, env: EnvInfo,
                                         warmup: Int, iterations: Int) throws {
        let recorder = SampleRecorder(name: "xwake_latency", capacity: iterations)
        let watcher = ThermalWatcher()
        watcher.start()

        let wakeSem = MachSemaphore()
        let ackSem = MachSemaphore()
        // Timestamp mailbox: written before signal, read after wake. The
        // semaphore is the synchronization edge; page is pre-touched.
        let signalTS = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
        signalTS.initialize(to: 0)
        defer { signalTS.deallocate() }

        let total = warmup + iterations
        let periodTicks = MachClock.fromNanos(1_000_000)

        let waiter = DedicatedThread(policy: policy) {
            for i in 0..<total {
                wakeSem.wait()
                let woke = MachClock.now()
                let sent = signalTS.pointee
                if i >= warmup {
                    let ns = woke >= sent ? MachClock.toNanos(woke - sent) : 0
                    recorder.record(ns, thermalOK: watcher.isNominal)
                }
                ackSem.signal()
            }
        }

        let signaler = DedicatedThread(policy: .default) {
            var next = MachClock.now() + periodTicks
            for _ in 0..<total {
                mach_wait_until(next)
                signalTS.pointee = MachClock.now()
                wakeSem.signal()
                ackSem.wait() // lock-step so the mailbox is never overwritten early
                next = MachClock.now() + periodTicks
            }
        }

        waiter.join()
        signaler.join()

        try finishCell(milestone: "m0", cellName: "xwake-\(policy.rawValue)",
                       cell: ["benchmark": "xwake", "E4.policy": policy.rawValue],
                       recorders: [recorder], env: env, watcher: watcher,
                       warmup: warmup, iterations: iterations, resultsRoot: resultsRoot)
    }

    // MARK: shared reporting

    private static func finishCell(milestone: String, cellName: String, cell: [String: String],
                                   recorders: [SampleRecorder], env: EnvInfo, watcher: ThermalWatcher,
                                   warmup: Int, iterations: Int, resultsRoot: URL) throws {
        let sink = try ResultSink(resultsRoot: resultsRoot, milestone: milestone, cellName: cellName)
        var meta = ResultSink.Meta(milestone: milestone, cell: cell, env: env,
                                   warmupIterations: warmup, measuredIterations: iterations)
        meta.thermalTimeline = watcher.stop()
        try sink.writeMeta(meta)
        try sink.writeSamplesCSV(iterationsOf: recorders)
        let summaries = recorders.map { $0.summarize() }
        try sink.writeSummaries(summaries)

        print("\ncell \(cellName):")
        printSummaryTable(summaries)
        print(recorders[0].bucketRender())
        print("  -> \(sink.dir.path)")
    }
}
