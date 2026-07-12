import Foundation
import HarnessCore
import MLX
import MLXLMCommon

/// Greedy (temperature-0) speculative decoding per Leviathan et al., with the
/// draft behind a protocol so M3.2 can swap the MLX draft for the CoreML/ANE
/// draft without touching the algorithm.
///
/// Cache discipline (the rollback the spec's §2 margin pays for):
///   - draft cache after propose(k):   context + d_1..d_{k-1}
///   - target cache after verify(k):   context + d_1..d_k (all fed in one batch)
///   - on acceptance of j<k drafts the target trims k-j entries, then feeds
///     the correction token; the draft trims (k-1)-j and ingests the
///     correction on its next propose. Both costs are measured, not modeled.
public protocol DraftTokenSource {
    /// Ingest newly-accepted tokens, then greedily propose k continuations.
    func propose(k: Int, ingest: [Int]) throws -> [Int]
    /// Drop the last n tokens of internal state (rejected draft suffix).
    func rollback(_ n: Int) throws
}

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

/// Target lane: batch-verifies k draft tokens in one forward pass and
/// maintains the "next token" logits between rounds.
public final class TargetVerifier {
    private let model: any LanguageModel
    private var cache: [KVCache]
    private var nextToken: Int

    public struct VerifyOutcome {
        public let accepted: [Int]   // j drafts + 1 correction/bonus token
        public let acceptedDrafts: Int
        public let allAccepted: Bool
    }

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

        var j = 0
        while j < k {
            let expected = j == 0 ? nextToken : rows[j - 1]
            if drafts[j] == expected { j += 1 } else { break }
        }

        let correction = j == 0 ? nextToken : rows[j - 1]
        if j < k {
            // Roll back the rejected suffix d_{j+1}..d_k, then feed the
            // correction so the cache matches the emitted sequence.
            for c in cache { _ = c.trim(k - j) }
            let logits2 = model(MLXArray([correction]).expandedDimensions(axis: 0), cache: cache)
            nextToken = argmaxLast(logits2)
            return VerifyOutcome(accepted: Array(drafts[0..<j]) + [correction],
                                 acceptedDrafts: j, allAccepted: false)
        } else {
            // All k accepted; the bonus token is the target's choice after
            // d_k, already computed. Feed it to keep the cache aligned.
            let bonus = rows[k - 1]
            let logits2 = model(MLXArray([bonus]).expandedDimensions(axis: 0), cache: cache)
            nextToken = argmaxLast(logits2)
            return VerifyOutcome(accepted: drafts + [bonus],
                                 acceptedDrafts: k, allAccepted: true)
        }
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
