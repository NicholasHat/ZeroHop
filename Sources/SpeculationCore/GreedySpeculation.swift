import HarnessCore

// Greedy (temperature-0) speculative decoding per Leviathan et al. — the
// backend-agnostic half. The draft and the target sit behind protocols so the
// same loops drive the MLX GPU draft, the CoreML/ANE draft and the MLX target
// (HarnessM3), and model-free fakes in the tests.
//
// Cache discipline (the rollback the spec's §2 margin pays for):
//   - draft cache after propose(k):   context + d_1..d_{k-1}
//   - target cache after verify(k):   context + d_1..d_k (all fed in one batch)
//   - on acceptance of j<k drafts the target trims k-j entries, then feeds
//     the correction token; the draft trims (k-1)-j and ingests the
//     correction on its next propose. Both costs are measured, not modeled.

public protocol DraftTokenSource {
    /// Ingest newly-accepted tokens, then greedily propose k continuations.
    func propose(k: Int, ingest: [Int]) throws -> [Int]
    /// Drop the last n tokens of internal state (rejected draft suffix) and
    /// any pending, not-yet-fed token. n may be 0: that still discards the
    /// pending d_k when only d_k was rejected.
    func rollback(_ n: Int) throws
}

public struct VerifyOutcome: Equatable {
    public let accepted: [Int]   // j drafts + 1 correction/bonus token
    public let acceptedDrafts: Int
    public let allAccepted: Bool

    public init(accepted: [Int], acceptedDrafts: Int, allAccepted: Bool) {
        self.accepted = accepted
        self.acceptedDrafts = acceptedDrafts
        self.allAccepted = allAccepted
    }
}

/// Target side: scores k drafts in one batch and returns what to emit.
public protocol GreedyVerifier {
    func verify(drafts: [Int]) -> VerifyOutcome
}

public enum GreedyAcceptance {
    /// The greedy acceptance rule. `anchor` is the target's own choice for the
    /// first draft position; `rows[i]` is its choice after d_1..d_{i+1} (one
    /// batched forward pass). Accept the longest prefix of drafts that matches
    /// the target, then emit the target's token at the first mismatch (the
    /// correction) or after the last draft (the bonus), so the output is
    /// exactly the target's own greedy sequence.
    public static func resolve(drafts: [Int], anchor: Int, rows: [Int]) -> VerifyOutcome {
        precondition(rows.count >= drafts.count, "need one target row per draft")
        func target(after j: Int) -> Int { j == 0 ? anchor : rows[j - 1] }
        var j = 0
        while j < drafts.count, drafts[j] == target(after: j) { j += 1 }
        return VerifyOutcome(accepted: Array(drafts[0..<j]) + [target(after: j)],
                             acceptedDrafts: j, allAccepted: j == drafts.count)
    }
}

public struct DecodeStats {
    public var produced: [Int] = []
    public var rounds = 0
    public var acceptedDrafts = 0
    public var pipelineHits = 0   // pipelined only
}

/// The serial loop: propose, verify, reconcile, repeat.
public enum SerialDecoder {
    public static func decode(draft: DraftTokenSource, verifier: GreedyVerifier,
                              k: Int, tokens: Int,
                              draftTimes: SampleRecorder, verifyTimes: SampleRecorder,
                              roundTimes: SampleRecorder) throws -> DecodeStats {
        var stats = DecodeStats()
        var ingest: [Int] = []
        while stats.produced.count < tokens {
            let t0 = MachClock.now()
            let drafts = try draft.propose(k: k, ingest: ingest)
            let t1 = MachClock.now()
            let outcome = verifier.verify(drafts: drafts)
            let t2 = MachClock.now()
            draftTimes.record(MachClock.toNanos(t1 - t0))
            verifyTimes.record(MachClock.toNanos(t2 - t1))
            roundTimes.record(MachClock.toNanos(t2 - t0))

            // Reconcile the draft with what was actually accepted.
            if !outcome.allAccepted {
                // Draft cache holds d_1..d_{k-1}; keep the j accepted ones.
                try draft.rollback((k - 1) - outcome.acceptedDrafts)
            }
            ingest = [outcome.accepted.last!] // correction or bonus token
            stats.produced.append(contentsOf: outcome.accepted)
            stats.acceptedDrafts += outcome.acceptedDrafts
            stats.rounds += 1
        }
        return stats
    }
}
