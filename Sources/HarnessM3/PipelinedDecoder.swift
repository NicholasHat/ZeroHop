import Darwin
import Foundation
import HarnessCore

/// M3.3 — pipelined heterogeneous speculation (spec §3): while the GPU
/// verifies batch N, the draft lane speculates batch N+1 from the *assumed*
/// continuation — its own greedy path.
///
/// Alignment: the speculative propose asks for k+1 tokens s_1..s_{k+1}. The
/// true post-verify sequence is d_1..d_k, b (bonus). If all k drafts were
/// accepted AND s_1 == b, then s_2..s_{k+1} is a perfectly aligned next
/// batch (pipeline HIT: the whole draft cost hid under the verify). On a
/// MISS the in-flight speculation is the wasted draft slot the spec's §2
/// margin budgets for: roll the draft back 2k−j positions (O(1) with the
/// fixed-window mask) and re-propose serially from the correction token.
///
/// Cache accounting (verified against both DraftTokenSource impls):
///   after propose(chunk N):        fed = ctx + d_1..d_{k-1}, pending [d_k]
///   after speculative propose(k+1): fed = ctx + d_1..d_k + s_1..s_k,
///                                   pending [s_{k+1}]
///   HIT:  exactly the state the next round needs — zero fixup.
///   MISS (j accepted + correction c): rollback(2k − j), propose(k, [c]).
public enum PipelinedDecoder {

    public struct Stats {
        public var produced: [Int] = []
        public var rounds = 0
        public var pipelineHits = 0
        public var acceptedDrafts = 0
        public var draftK = 0
    }

    // Draft-lane thread protocol: all DraftTokenSource calls happen on one
    // dedicated RT pthread (E3/E4 winners: sync prediction, time-constraint
    // policy); the decode loop hands over commands through a mach-semaphore
    // mailbox (M0-measured wake cost: ~2 µs p50).
    private final class DraftLane {
        enum Command {
            case propose(k: Int, ingest: [Int])
            case rollbackAndPropose(rollback: Int, k: Int, ingest: [Int])
            case stop
        }

        private var command: Command = .stop
        private var result: [Int] = []
        private var failure: Error?
        private let work = MachSemaphore()
        private let done = MachSemaphore()
        private var thread: DedicatedThread?

        init(draft: DraftTokenSource) {
            thread = DedicatedThread(policy: .timeConstraint) { [self] in
                while true {
                    work.wait()
                    switch command {
                    case .stop:
                        done.signal()
                        return
                    case .propose(let k, let ingest):
                        do { result = try draft.propose(k: k, ingest: ingest) }
                        catch { failure = error }
                    case .rollbackAndPropose(let n, let k, let ingest):
                        do {
                            try draft.rollback(n)
                            result = try draft.propose(k: k, ingest: ingest)
                        } catch { failure = error }
                    }
                    done.signal()
                }
            }
        }

        /// Fire a command without waiting (the overlap).
        func send(_ cmd: Command) {
            command = cmd
            work.signal()
        }

        /// Collect the result of the last send.
        func collect() throws -> [Int] {
            done.wait()
            if let failure { throw failure }
            return result
        }

        func run(_ cmd: Command) throws -> [Int] {
            send(cmd)
            return try collect()
        }

        func stop() {
            send(.stop)
            done.wait()
            thread?.join()
        }
    }

    public static func decode(draft: DraftTokenSource, verifier: TargetVerifier,
                              k: Int, tokens: Int,
                              draftTimes: SampleRecorder, verifyTimes: SampleRecorder,
                              roundTimes: SampleRecorder) throws -> Stats {
        let lane = DraftLane(draft: draft)
        defer { lane.stop() }

        var stats = Stats()
        // Round 1 is unavoidably serial: nothing to overlap with yet.
        var drafts = try lane.run(.propose(k: k, ingest: []))

        while stats.produced.count < tokens {
            let r0 = MachClock.now()
            // Overlap: speculative k+1 self-continuation on the ANE lane…
            lane.send(.propose(k: k + 1, ingest: []))
            // …while the GPU verifies the current batch.
            let v0 = MachClock.now()
            let outcome = verifier.verify(drafts: drafts)
            verifyTimes.record(MachClock.toNanos(MachClock.now() - v0))
            let spec = try lane.collect()
            draftTimes.record(MachClock.toNanos(MachClock.now() - v0)) // ≥ verify ⇒ draft-bound

            stats.produced.append(contentsOf: outcome.accepted)
            stats.acceptedDrafts += outcome.acceptedDrafts
            stats.draftK += k
            stats.rounds += 1

            if outcome.allAccepted, spec.first == outcome.accepted.last {
                stats.pipelineHits += 1
                drafts = Array(spec.dropFirst())
            } else {
                // Wasted draft slot: discard the speculation, resync.
                let j = outcome.acceptedDrafts
                drafts = try lane.run(.rollbackAndPropose(
                    rollback: 2 * k - j, k: k, ingest: [outcome.accepted.last!]))
            }
            roundTimes.record(MachClock.toNanos(MachClock.now() - r0))
        }
        return stats
    }
}
