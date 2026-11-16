import Foundation
import HarnessCore
import MLX
import MLXLMCommon
import SpeculationCore

// The MLX lanes of greedy speculative decoding. The algorithm itself (the
// acceptance rule, cache discipline and decode loops) is backend-agnostic
// and lives in SpeculationCore.

/// MLX-backed draft (M3.1: validates the algorithm with both lanes on the
/// GPU; the heterogeneous ANE draft is M3.2).
public final class MLXDraft: DraftTokenSource {
    private let model: any LanguageModel
    private var cache: [KVCache]

    public init(model: any LanguageModel, prompt: [Int]) {
        self.model = model
        self.cache = model.newCache(parameters: nil)
        // Prime with the prompt except its last token: propose() always
        // ingests at least one token to get its first logits row.
        if prompt.count > 1 {
            let logits = model(MLXArray(prompt.dropLast()).expandedDimensions(axis: 0), cache: cache)
            eval(logits)
        }
        pendingIngest = [prompt.last!]
    }

    private var pendingIngest: [Int]

    public func propose(k: Int, ingest: [Int]) throws -> [Int] {
        var feed = pendingIngest + ingest
        pendingIngest = []
        var drafts: [Int] = []
        drafts.reserveCapacity(k)
        for _ in 0..<k {
            let logits = model(MLXArray(feed).expandedDimensions(axis: 0), cache: cache)
            let next = argmaxLast(logits)
            drafts.append(next)
            feed = [next]
        }
        // d_k was chosen but not fed; it belongs to the next ingest if accepted.
        pendingIngest = [drafts[k - 1]]
        return drafts
    }

    public func rollback(_ n: Int) throws {
        guard n > 0 else { return }
        for c in cache {
            precondition(c.isTrimmable, "draft cache not trimmable")
            _ = c.trim(n)
        }
        // Whatever was pending sat on top of the trimmed suffix; the decoder
        // supplies the correct continuation via `ingest`.
        pendingIngest = []
    }
}

extension MLXDraft: SamplingDraftSource {
    public func proposeSampled(k: Int, ingest: [Int], temperature: Float,
                               rng: inout SplitMix64) throws -> (tokens: [Int], dists: [[Float]]) {
        var feed = pendingIngest + ingest
        pendingIngest = []
        var tokens: [Int] = []
        var dists: [[Float]] = []
        for _ in 0..<k {
            let logits = model(MLXArray(feed).expandedDimensions(axis: 0), cache: cache)
            let raw = logits[0, -1, 0...].asType(.float32).asArray(Float.self)
            let q = Sampling.tempSoftmax(raw, temperature: temperature)
            let next = Sampling.sample(q, rng: &rng)
            tokens.append(next)
            dists.append(q)
            feed = [next]
        }
        pendingIngest = [tokens[k - 1]]
        return (tokens, dists)
    }
}

/// Target lane: batch-verifies k draft tokens in one forward pass and
/// maintains the "next token" logits between rounds.
public final class TargetVerifier: GreedyVerifier {
    private let model: any LanguageModel
    private var cache: [KVCache]
    private var nextToken: Int

    public init(model: any LanguageModel, prompt: [Int]) {
        self.model = model
        self.cache = model.newCache(parameters: nil)
        let logits = model(MLXArray(prompt).expandedDimensions(axis: 0), cache: cache)
        self.nextToken = argmaxLast(logits)
    }

    /// The token the target itself would emit next (used by the baseline and
    /// as the anchor for the first draft comparison).
    public var anchor: Int { nextToken }

    public func verify(drafts: [Int]) -> VerifyOutcome {
        let k = drafts.count
        // Anchor check is free: nextToken already IS the target's greedy
        // choice for position n. Feed all k drafts in one pass for rows
        // predicting positions n+1..n+k.
        let logits = model(MLXArray(drafts).expandedDimensions(axis: 0), cache: cache)
        let rows = argmaxRows(logits) // rows[i] = target's choice after d_1..d_{i+1}
        let outcome = GreedyAcceptance.resolve(drafts: drafts, anchor: nextToken, rows: rows)

        // Roll back the rejected suffix d_{j+1}..d_k (if any), then feed the
        // emitted correction/bonus token (already computed) so the cache
        // matches the emitted sequence.
        let rejected = k - outcome.acceptedDrafts
        if rejected > 0 {
            for c in cache { _ = c.trim(rejected) }
        }
        let emitted = outcome.accepted.last!
        let logits2 = model(MLXArray([emitted]).expandedDimensions(axis: 0), cache: cache)
        nextToken = argmaxLast(logits2)
        return outcome
    }

    /// GPU-only greedy baseline from the same primed state.
    public func generateBaseline(n: Int) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(n)
        var token = nextToken
        for _ in 0..<n {
            out.append(token)
            let logits = model(MLXArray([token]).expandedDimensions(axis: 0), cache: cache)
            token = argmaxLast(logits)
        }
        nextToken = token
        return out
    }
}

@inline(__always)
func argmaxLast(_ logits: MLXArray) -> Int {
    let last = logits[0, -1, 0...]
    return Int(MLX.argMax(last).item(Int32.self))
}

@inline(__always)
func argmaxRows(_ logits: MLXArray) -> [Int] {
    MLX.argMax(logits[0], axis: -1).asArray(Int32.self).map(Int.init)
}
