import Foundation
import HarnessCore
import HarnessM1
import MLX
import MLXLLM
import MLXLMCommon

public struct M3Config {
    public var targetID: String
    public var draftID: String
    public var k: Int
    public var tokens: Int
    public var prompt: String
    /// Path to a compiled stateful CoreML draft (.mlmodelc). When set, the
    /// draft runs on the ANE (M3.2 heterogeneous mode) instead of MLX.
    public var coremlDraftPath: String?
    /// M3.3: overlap the draft of batch N+1 with the verify of batch N.
    public var overlap = false
    /// Sampling temperature; 0 = greedy. >0 uses Leviathan rejection
    /// acceptance (serial loop; requires a logits-interface draft).
    public var temperature: Float = 0

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

        // Plain contexts, not ModelContainers: the whole run is one task that
        // owns both models, so there is no concurrent access for a container
        // to serialize (and nesting two `perform` closures would capture a
        // non-Sendable context in a @Sendable closure).
        print("loading target…")
        let targetCtx = try await LLMModelFactory.shared.load(
            configuration: ModelConfiguration(id: config.targetID)) { p in
                report(progress: p, label: "target")
            }
        print("\nloading draft…")
        let draftCtx = try await LLMModelFactory.shared.load(
            configuration: ModelConfiguration(id: config.draftID)) { p in
                report(progress: p, label: "draft")
            }
        print("")

        try decodeAndReport(targetCtx: targetCtx, draftCtx: draftCtx,
                            config: config, env: env, resultsRoot: resultsRoot)
    }

    static func decodeAndReport(targetCtx: ModelContext, draftCtx: ModelContext,
                                config: M3Config, env: EnvInfo, resultsRoot: URL) throws {
        let promptTokens = targetCtx.tokenizer.encode(text: config.prompt)
        let draftPrompt = draftCtx.tokenizer.encode(text: config.prompt)
        if promptTokens != draftPrompt {
            print("WARNING: draft/target tokenizations differ (\(draftPrompt.count) vs \(promptTokens.count) tokens) — models must share a vocab for speculation to be sound")
        }

        if config.temperature > 0 {
            try stochasticRun(targetCtx: targetCtx, draftCtx: draftCtx, config: config,
                              env: env, resultsRoot: resultsRoot, promptTokens: promptTokens)
            return
        }

        // --- speculative run --------------------------------------------------
        let verifier = TargetVerifier(model: targetCtx.model, prompt: promptTokens)
        let draft: DraftTokenSource
        let draftLane: String
        var coremlDraft: CoreMLDraft?
        if let coremlPath = config.coremlDraftPath {
            print("draft lane: CoreML/ANE (\(coremlPath))")
            let ane = try CoreMLDraft(compiledModelURL: URL(fileURLWithPath: coremlPath),
                                      prompt: promptTokens)
            coremlDraft = ane
            draft = ane
            draftLane = "coreml-ane"
        } else {
            draft = MLXDraft(model: draftCtx.model, prompt: promptTokens)
            draftLane = "mlx-gpu"
        }
        let thermal = ThermalWatcher()
        thermal.start()

        let n = config.tokens
        let draftTimes = SampleRecorder(
            name: config.overlap ? "overlap_span" : "draft_propose", capacity: n)
        let verifyTimes = SampleRecorder(name: "target_verify", capacity: n)
        let roundTimes = SampleRecorder(name: "round", capacity: n)

        var produced: [Int] = []
        var acceptedDraftTotal = 0
        var verifyRounds = 0
        var pipelineHits = 0
        let specStart = MachClock.now()

        if config.overlap {
            // M3.3: draft batch N+1 on its own RT thread while the GPU
            // verifies batch N (PipelinedDecoder).
            guard config.coremlDraftPath != nil else {
                throw HarnessError("--overlap on requires --coreml-draft: overlapping an MLX "
                    + "draft with an MLX verify runs concurrent evals on one device "
                    + "(unsupported), and same-device overlap has nothing to gain anyway")
            }
            let stats = try PipelinedDecoder.decode(
                draft: draft, verifier: verifier, k: config.k, tokens: n,
                draftTimes: draftTimes, verifyTimes: verifyTimes, roundTimes: roundTimes)
            produced = stats.produced
            acceptedDraftTotal = stats.acceptedDrafts
            verifyRounds = stats.rounds
            pipelineHits = stats.pipelineHits
        } else {
            var ingest: [Int] = []
            while produced.count < n {
                let t0 = MachClock.now()
                let drafts = try draft.propose(k: config.k, ingest: ingest)
                let t1 = MachClock.now()
                let outcome = verifier.verify(drafts: drafts)
                let t2 = MachClock.now()
                draftTimes.record(MachClock.toNanos(t1 - t0))
                verifyTimes.record(MachClock.toNanos(t2 - t1))
                roundTimes.record(MachClock.toNanos(t2 - t0))

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
        if config.overlap {
            print("pipeline:    \(pipelineHits)/\(verifyRounds) speculative batches landed (\(String(format: "%.1f%%", 100.0 * Double(pipelineHits) / Double(max(verifyRounds, 1)))))")
        }
        printSummaryTable([draftTimes.summarize(), verifyTimes.summarize(), roundTimes.summarize()])
        let sameOutput = produced == baseline
        print("greedy-equivalence check (spec output == baseline output): \(sameOutput ? "PASS" : "FAIL")")
        print("text: \(targetCtx.tokenizer.decode(tokens: Array(produced.prefix(60))))…")

        let sink = try ResultSink(resultsRoot: resultsRoot, milestone: "m3",
                                  cellName: "\(draftLane)_k\(config.k)\(config.overlap ? "_overlap" : "")")
        var cell = cellInfo(config, lane: draftLane, coreml: coremlDraft)
        cell["overlap"] = config.overlap ? "on" : "off"
        var meta = ResultSink.Meta(
            milestone: "m3", cell: cell,
            env: env, warmupIterations: 0, measuredIterations: verifyRounds)
        meta.thermalTimeline = thermal.stop()
        meta.notes.append("speculative \(String(format: "%.2f", specTokS)) tok/s vs baseline \(String(format: "%.2f", baseTokS)) tok/s (\(String(format: "%.2fx", specTokS / baseTokS)))")
        meta.notes.append("acceptance rate \(String(format: "%.3f", acceptRate)), tokens/round \(String(format: "%.2f", Double(produced.count) / Double(verifyRounds)))")
        meta.notes.append("greedy equivalence: \(sameOutput ? "PASS" : "FAIL")")
        if config.overlap {
            meta.notes.append("pipeline hits \(pipelineHits)/\(verifyRounds)")
        }
        try sink.writeMeta(meta)
        try sink.writeSamplesCSV(iterationsOf: [draftTimes, verifyTimes, roundTimes])
        try sink.writeSummaries([draftTimes.summarize(), verifyTimes.summarize(), roundTimes.summarize()])
        print("-> \(sink.dir.path)")
    }

    /// Stochastic (temperature > 0) serial speculation: Leviathan rejection
    /// acceptance. No greedy-equivalence check exists here — matching the
    /// target distribution is guaranteed by the acceptance construction.
    static func stochasticRun(targetCtx: ModelContext, draftCtx: ModelContext,
                              config: M3Config, env: EnvInfo, resultsRoot: URL,
                              promptTokens: [Int]) throws {
        guard !config.overlap else {
            throw HarnessError("stochastic mode is serial-only in this PoC (overlap hit-detection is token-exact; sampled bonus matches are rare)")
        }
        let draft: SamplingDraftSource
        let draftLane: String
        var coremlDraft: CoreMLDraft?
        if let coremlPath = config.coremlDraftPath {
            print("draft lane: CoreML/ANE (\(coremlPath)), temperature \(config.temperature)")
            let ane = try CoreMLDraft(compiledModelURL: URL(fileURLWithPath: coremlPath),
                                      prompt: promptTokens)
            coremlDraft = ane
            draft = ane
            draftLane = "coreml-ane"
        } else {
            draft = MLXDraft(model: draftCtx.model, prompt: promptTokens)
            draftLane = "mlx-gpu"
        }
        let thermal = ThermalWatcher()
        thermal.start()
        let verifier = StochasticVerifier(model: targetCtx.model, prompt: promptTokens,
                                          temperature: config.temperature)
        var rng = SplitMix64(seed: 0xC0FFEE)

        let n = config.tokens
        let draftTimes = SampleRecorder(name: "draft_propose", capacity: n)
        let verifyTimes = SampleRecorder(name: "target_verify", capacity: n)
        var produced: [Int] = []
        var acceptedDraftTotal = 0
        var rounds = 0
        var ingest: [Int] = []
        let specStart = MachClock.now()
        while produced.count < n {
            let t0 = MachClock.now()
            let (drafts, dists) = try draft.proposeSampled(
                k: config.k, ingest: ingest, temperature: config.temperature, rng: &rng)
            let t1 = MachClock.now()
            let outcome = verifier.verify(drafts: drafts, draftDists: dists, rng: &rng)
            let t2 = MachClock.now()
            draftTimes.record(MachClock.toNanos(t1 - t0))
            verifyTimes.record(MachClock.toNanos(t2 - t1))
            if outcome.acceptedDrafts < config.k {
                try draft.rollback((config.k - 1) - outcome.acceptedDrafts)
            }
            ingest = [outcome.accepted.last!]
            produced.append(contentsOf: outcome.accepted)
            acceptedDraftTotal += outcome.acceptedDrafts
            rounds += 1
        }
        let specNS = MachClock.toNanos(MachClock.now() - specStart)
        let specTokS = Double(produced.count) * 1e9 / Double(specNS)

        let baselineVerifier = StochasticVerifier(model: targetCtx.model, prompt: promptTokens,
                                                  temperature: config.temperature)
        var baseRng = SplitMix64(seed: 0xC0FFEE)
        let baseStart = MachClock.now()
        let baseline = baselineVerifier.generateBaseline(n: produced.count, rng: &baseRng)
        let baseNS = MachClock.toNanos(MachClock.now() - baseStart)
        let baseTokS = Double(baseline.count) * 1e9 / Double(baseNS)
        let acceptRate = Double(acceptedDraftTotal) / Double(rounds * config.k)

        print("speculative: \(produced.count) tokens in \(SampleRecorder.fmt(specNS)) → \(String(format: "%.2f", specTokS)) tok/s")
        print("baseline:    \(baseline.count) tokens in \(SampleRecorder.fmt(baseNS)) → \(String(format: "%.2f", baseTokS)) tok/s")
        print("acceptance:  \(String(format: "%.1f%%", acceptRate * 100)) of drafts (k=\(config.k), T=\(config.temperature)), \(String(format: "%.2f", Double(produced.count) / Double(rounds))) tokens/verify-round")
        print("speedup:     \(String(format: "%.2fx", specTokS / baseTokS))")
        printSummaryTable([draftTimes.summarize(), verifyTimes.summarize()])
        print("text: \(targetCtx.tokenizer.decode(tokens: Array(produced.prefix(60))))…")

        let sink = try ResultSink(resultsRoot: resultsRoot, milestone: "m3",
                                  cellName: "\(draftLane)_k\(config.k)_T\(config.temperature)")
        var cell = cellInfo(config, lane: draftLane, coreml: coremlDraft)
        cell["temperature"] = String(config.temperature)
        var meta = ResultSink.Meta(
            milestone: "m3", cell: cell,
            env: env, warmupIterations: 0, measuredIterations: rounds)
        meta.thermalTimeline = thermal.stop()
        meta.notes.append("stochastic \(String(format: "%.2f", specTokS)) vs baseline \(String(format: "%.2f", baseTokS)) tok/s (\(String(format: "%.2fx", specTokS / baseTokS)))")
        meta.notes.append("acceptance \(String(format: "%.3f", acceptRate)) at T=\(config.temperature); target distribution exact by construction (Leviathan rejection acceptance)")
        try sink.writeMeta(meta)
        try sink.writeSamplesCSV(iterationsOf: [draftTimes, verifyTimes])
        try sink.writeSummaries([draftTimes.summarize(), verifyTimes.summarize()])
        print("-> \(sink.dir.path)")
    }

    /// meta.json cell keys common to both modes. On the ANE lane the MLX
    /// `draft` is loaded only for the tokenizer check; the model that actually
    /// drafted is the CoreML file, so record its path and output head — the
    /// 4/6-bit and argmax/logits exports are otherwise indistinguishable
    /// after the fact.
    static func cellInfo(_ config: M3Config, lane: String,
                         coreml: CoreMLDraft?) -> [String: String] {
        var cell = ["target": config.targetID, "draft": config.draftID,
                    "k": String(config.k), "draft_lane": lane]
        if let coreml, let path = config.coremlDraftPath {
            cell["coreml_draft"] = URL(fileURLWithPath: path).standardizedFileURL.path
            cell["coreml_draft_head"] = coreml.head
        }
        return cell
    }

    static func report(progress: Progress, label: String) {
        let pct = Int(progress.fractionCompleted * 100)
        if pct % 10 == 0 {
            FileHandle.standardOutput.write("\r\(label): \(pct)%".data(using: .utf8)!)
        }
    }
}
