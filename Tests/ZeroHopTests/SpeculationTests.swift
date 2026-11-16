import HarnessCore
import SpeculationCore
import Testing

@Suite struct GreedyAcceptanceTests {
    // anchor = target's token for position n; rows[i] = its token after d_1..d_{i+1}.
    @Test func allAcceptedEmitsTheBonusToken() {
        let o = GreedyAcceptance.resolve(drafts: [5, 6, 7], anchor: 5, rows: [6, 7, 8])
        #expect(o == VerifyOutcome(accepted: [5, 6, 7, 8], acceptedDrafts: 3, allAccepted: true))
    }

    @Test func firstMismatchEmitsTheCorrection() {
        let o = GreedyAcceptance.resolve(drafts: [5, 9, 7], anchor: 5, rows: [6, 7, 8])
        #expect(o == VerifyOutcome(accepted: [5, 6], acceptedDrafts: 1, allAccepted: false))
    }

    @Test func rejectedFirstDraftEmitsTheAnchor() {
        let o = GreedyAcceptance.resolve(drafts: [1, 6, 7], anchor: 5, rows: [6, 7, 8])
        #expect(o == VerifyOutcome(accepted: [5], acceptedDrafts: 0, allAccepted: false))
    }
}

/// Model-free end-to-end check of the decode loops' cache bookkeeping.
///
/// The target's greedy output is a fixed sequence `truth`. The fake draft
/// keeps an explicit token context with the same fed/pending semantics as
/// the real drafts, and predicts truth unless told to miss at a position. The
/// verifier corrects every wrong draft, so — exactly like the real ANE cache
/// bug — a bookkeeping error would leave the output correct and only corrupt
/// the draft's context. So the test checks the context directly: every
/// non-speculative propose must start from a context that is exactly the
/// true prefix.
@Suite struct DecodeBookkeepingTests {
    static let promptLength = 5

    static func truth(_ i: Int) -> Int { (i * 7_919) % 1_000 + 1 } // > 0

    final class FakeTarget: GreedyVerifier {
        var next = DecodeBookkeepingTests.promptLength
        func verify(drafts: [Int]) -> VerifyOutcome {
            let rows = (1...drafts.count).map { DecodeBookkeepingTests.truth(next + $0) }
            let o = GreedyAcceptance.resolve(drafts: drafts, anchor: DecodeBookkeepingTests.truth(next),
                                             rows: rows)
            next += o.accepted.count
            return o
        }
    }

    final class FakeDraft: DraftTokenSource {
        private var fed: [Int]
        private var pending: [Int]
        private let missesAt: (Int) -> Bool
        private(set) var misalignedProposes = 0

        init(missesAt: @escaping (Int) -> Bool) {
            let prompt = (0..<DecodeBookkeepingTests.promptLength).map(DecodeBookkeepingTests.truth)
            fed = Array(prompt.dropLast())
            pending = [prompt.last!]
            self.missesAt = missesAt
        }

        private var aligned: Bool {
            fed.enumerated().allSatisfy { $0.element == DecodeBookkeepingTests.truth($0.offset) }
        }

        /// Greedy prediction for the next position given the fed context.
        private func predict() -> Int {
            let p = fed.count
            return aligned && !missesAt(p) ? DecodeBookkeepingTests.truth(p) : -(p + 1)
        }

        func propose(k: Int, ingest: [Int]) throws -> [Int] {
            fed += pending + ingest
            pending = []
            // A propose that ingests verified tokens must sit on the true prefix;
            // only the pipelined lane's speculative proposes (no ingest) may not.
            if !ingest.isEmpty, !aligned { misalignedProposes += 1 }
            var drafts: [Int] = []
            for _ in 0..<k {
                let d = predict()
                drafts.append(d)
                if drafts.count < k { fed.append(d) }
            }
            pending = [drafts[k - 1]]
            return drafts
        }

        func rollback(_ n: Int) throws {
            precondition(n >= 0 && n <= fed.count)
            fed.removeLast(n)
            pending = []
        }
    }

    enum Loop: String, CaseIterable { case serial, pipelined }

    static let missPatterns: [(String, (Int) -> Bool)] = [
        ("never", { _ in false }),
        ("every 3rd", { $0 % 3 == 0 }),
        ("every 5th + every 7th", { $0 % 5 == 0 || $0 % 7 == 0 }),
        ("mostly", { $0 % 4 != 1 }),
    ]

    @Test(arguments: Loop.allCases, 1...6)
    func outputIsTheTargetSequenceAndTheDraftStaysAligned(loop: Loop, k: Int) throws {
        let tokens = 120
        for (name, misses) in Self.missPatterns {
            let draft = FakeDraft(missesAt: misses)
            let target = FakeTarget()
            let recorders = (0..<3).map { SampleRecorder(name: "t\($0)", capacity: tokens) }
            let decode = loop == .serial ? SerialDecoder.decode : PipelinedDecoder.decode
            let stats = try decode(draft, target, k, tokens, recorders[0], recorders[1], recorders[2])

            let p = Self.promptLength
            #expect(stats.produced == (p..<(p + stats.produced.count)).map(Self.truth),
                    "\(loop) k=\(k) misses \(name)")
            #expect(stats.produced.count >= tokens)
            #expect(stats.produced.count == stats.rounds + stats.acceptedDrafts) // j + 1 per round
            #expect(draft.misalignedProposes == 0, "\(loop) k=\(k) misses \(name)")
            if name == "never" {
                #expect(stats.acceptedDrafts == stats.rounds * k)
                if loop == .pipelined { #expect(stats.pipelineHits == stats.rounds) }
            }
        }
    }
}
