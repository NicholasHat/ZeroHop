import Foundation
import HarnessCore
import MLX
import MLXLLM
import MLXLMCommon

public struct M3Config {
    public var targetID: String
    public var draftID: String
    public var k: Int
    public var tokens: Int
    public var prompt: String

    public init(targetID: String, draftID: String, k: Int, tokens: Int, prompt: String) {
        self.targetID = targetID
        self.draftID = draftID
        self.k = k
        self.tokens = tokens
        self.prompt = prompt
    }
}

/// M3.1 — speculative decoding with both lanes on MLX (algorithm validation
/// + tokens/s accounting). M3.2 swaps `MLXDraft` for the CoreML/ANE draft.
public enum M3Runner {

    public static func run(resultsRoot: URL, config: M3Config) async throws {
        let env = EnvInfo.capture()
        print("M3.1 speculative decode — \(env.chip), macOS \(env.macOSBuild), power: \(env.powerSource)")
        print("target: \(config.targetID)\ndraft:  \(config.draftID)\nk=\(config.k), n=\(config.tokens)")

        print("loading target…")
        let targetContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(id: config.targetID)) { p in
                report(progress: p, label: "target")
            }
        print("\nloading draft…")
        let draftContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(id: config.draftID)) { p in
                report(progress: p, label: "draft")
            }
        print("")

        let cfg = config
        let root = resultsRoot
        let envCaptured = env
        try await targetContainer.perform { targetCtx in
            try await draftContainer.perform { draftCtx in
                try decodeAndReport(targetCtx: targetCtx, draftCtx: draftCtx,
                                    config: cfg, env: envCaptured, resultsRoot: root)
            }
        }
    }

    static func decodeAndReport(targetCtx: ModelContext, draftCtx: ModelContext,
                                config: M3Config, env: EnvInfo, resultsRoot: URL) throws {
        let promptTokens = targetCtx.tokenizer.encode(text: config.prompt)
        let draftPrompt = draftCtx.tokenizer.encode(text: config.prompt)
        if promptTokens != draftPrompt {
            print("WARNING: draft/target tokenizations differ (\(draftPrompt.count) vs \(promptTokens.count) tokens) — models must share a vocab for speculation to be sound")
        }

        // --- speculative run --------------------------------------------------
        let verifier = TargetVerifier(model: targetCtx.model, prompt: promptTokens)
        let draft = MLXDraft(model: draftCtx.model, prompt: promptTokens)

        let n = config.tokens
        let draftTimes = SampleRecorder(name: "draft_propose", capacity: n)
        let verifyTimes = SampleRecorder(name: "target_verify", capacity: n)

        var produced: [Int] = []
        var acceptedDraftTotal = 0
        var verifyRounds = 0
        var ingest: [Int] = []
        let specStart = MachClock.now()
        while produced.count < n {
            let t0 = MachClock.now()
            let drafts = try draft.propose(k: config.k, ingest: ingest)
            let t1 = MachClock.now()
            let outcome = verifier.verify(drafts: drafts)
            let t2 = MachClock.now()
            draftTimes.record(MachClock.toNanos(t1 - t0))
            verifyTimes.record(MachClock.toNanos(t2 - t1))

            // Reconcile the draft with what was actually accepted.
            if !outcome.allAccepted {
                // Draft cache holds d_1..d_{k-1}; keep the j accepted ones.
                try draft.rollback((config.k - 1) - outcome.acceptedDrafts)
            }
            ingest = [outcome.accepted.last!] // correction or bonus token
            produced.append(contentsOf: outcome.accepted)
            acceptedDraftTotal += outcome.acceptedDrafts
            verifyRounds += 1
        }
        let specNS = MachClock.toNanos(MachClock.now() - specStart)
        let specTokS = Double(produced.count) * 1e9 / Double(specNS)
        let acceptRate = Double(acceptedDraftTotal) / Double(verifyRounds * config.k)

        // --- GPU-only baseline (same target, fresh state) ----------------------
        let baselineVerifier = TargetVerifier(model: targetCtx.model, prompt: promptTokens)
        let baseStart = MachClock.now()
        let baseline = baselineVerifier.generateBaseline(n: produced.count)
        let baseNS = MachClock.toNanos(MachClock.now() - baseStart)
        let baseTokS = Double(baseline.count) * 1e9 / Double(baseNS)

        // --- report -----------------------------------------------------------
        print("speculative: \(produced.count) tokens in \(SampleRecorder.fmt(specNS)) → \(String(format: "%.2f", specTokS)) tok/s")
        print("baseline:    \(baseline.count) tokens in \(SampleRecorder.fmt(baseNS)) → \(String(format: "%.2f", baseTokS)) tok/s")
        print("acceptance:  \(String(format: "%.1f%%", acceptRate * 100)) of drafts (k=\(config.k)), \(String(format: "%.2f", Double(produced.count) / Double(verifyRounds))) tokens/verify-round")
        print("speedup:     \(String(format: "%.2fx", specTokS / baseTokS))")
        printSummaryTable([draftTimes.summarize(), verifyTimes.summarize()])
        let sameOutput = produced == baseline
        print("greedy-equivalence check (spec output == baseline output): \(sameOutput ? "PASS" : "FAIL")")
        print("text: \(targetCtx.tokenizer.decode(tokens: Array(produced.prefix(60))))…")

        let sink = try ResultSink(resultsRoot: resultsRoot, milestone: "m3",
                                  cellName: "mlxdraft_k\(config.k)")
        var meta = ResultSink.Meta(
            milestone: "m3",
            cell: ["target": config.targetID, "draft": config.draftID,
                   "k": String(config.k), "draft_lane": "mlx-gpu"],
            env: env, warmupIterations: 0, measuredIterations: verifyRounds)
        meta.notes.append("speculative \(String(format: "%.2f", specTokS)) tok/s vs baseline \(String(format: "%.2f", baseTokS)) tok/s (\(String(format: "%.2fx", specTokS / baseTokS)))")
        meta.notes.append("acceptance rate \(String(format: "%.3f", acceptRate)), tokens/round \(String(format: "%.2f", Double(produced.count) / Double(verifyRounds)))")
        meta.notes.append("greedy equivalence: \(sameOutput ? "PASS" : "FAIL")")
        try sink.writeMeta(meta)
        try sink.writeSamplesCSV(iterationsOf: [draftTimes, verifyTimes])
        try sink.writeSummaries([draftTimes.summarize(), verifyTimes.summarize()])
        print("-> \(sink.dir.path)")
    }

    static func report(progress: Progress, label: String) {
        let pct = Int(progress.fractionCompleted * 100)
        if pct % 10 == 0 {
            FileHandle.standardOutput.write("\r\(label): \(pct)%".data(using: .utf8)!)
        }
    }
}
