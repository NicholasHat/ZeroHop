import Foundation
import HarnessCore
import MLX
import MLXLMCommon

/// Stochastic speculative decoding per Leviathan et al. (2023): accept draft
/// token d_i with probability min(1, p(d_i)/q(d_i)); on rejection, resample
/// the correction from norm(max(0, p − q)). This makes the output distribution
/// exactly the target's — the property greedy equivalence checks in the
/// temperature-0 case and the math guarantees here.
///
/// The draft must expose its full per-position distributions (q), so this
/// path requires a logits-interface draft (the fused-argmax export cannot
/// provide q).

/// Deterministic seeded RNG so runs are reproducible across lanes.
public struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    public mutating func uniform() -> Float {
        Float(next() >> 40) * (1.0 / Float(1 << 24))
    }
}

public enum Sampling {
    /// Temperature softmax in place; returns the normalized distribution.
    public static func tempSoftmax(_ logits: [Float], temperature: Float) -> [Float] {
        var out = logits
        let t = max(temperature, 1e-4)
        var maxV = -Float.infinity
        for v in out where v > maxV { maxV = v }
        var sum: Float = 0
        for i in 0..<out.count {
            let e = expf((out[i] - maxV) / t)
            out[i] = e
            sum += e
        }
        let inv = 1 / sum
        for i in 0..<out.count { out[i] *= inv }
        return out
    }

    public static func sample(_ dist: [Float], rng: inout SplitMix64) -> Int {
        let u = rng.uniform()
        var acc: Float = 0
        for (i, p) in dist.enumerated() {
            acc += p
            if u < acc { return i }
        }
        return dist.count - 1
    }

    /// norm(max(0, p − q)) — the rejection-resampling residual.
    public static func residual(_ p: [Float], _ q: [Float]) -> [Float] {
        var r = [Float](repeating: 0, count: p.count)
        var sum: Float = 0
        for i in 0..<p.count {
            let v = p[i] - q[i]
            if v > 0 { r[i] = v; sum += v }
        }
        if sum <= 0 { return p } // distributions identical: fall back to p
        let inv = 1 / sum
        for i in 0..<r.count { r[i] *= inv }
        return r
    }
}

/// A draft that can propose sampled tokens along with its full q distribution
/// at every proposed position.
public protocol SamplingDraftSource {
    func proposeSampled(k: Int, ingest: [Int], temperature: Float,
                        rng: inout SplitMix64) throws -> (tokens: [Int], dists: [[Float]])
    func rollback(_ n: Int) throws
}

/// Target-side stochastic verification. Mirrors TargetVerifier's greedy
/// batch verify but with distribution-level accept/reject.
public final class StochasticVerifier {
    private let model: any LanguageModel
    private var cache: [KVCache]
    private var nextDist: [Float]
    private let temperature: Float

    public struct Outcome {
        public let accepted: [Int]
        public let acceptedDrafts: Int
    }

    public init(model: any LanguageModel, prompt: [Int], temperature: Float) {
        self.model = model
        self.temperature = temperature
        self.cache = model.newCache(parameters: nil)
        let logits = model(MLXArray(prompt).expandedDimensions(axis: 0), cache: cache)
        self.nextDist = Sampling.tempSoftmax(Self.row(logits, -1), temperature: temperature)
    }

    static func row(_ logits: MLXArray, _ i: Int) -> [Float] {
        logits[0, i, 0...].asType(.float32).asArray(Float.self)
    }

    public func verify(drafts: [Int], draftDists: [[Float]],
                       rng: inout SplitMix64) -> Outcome {
        let k = drafts.count
        let logits = model(MLXArray(drafts).expandedDimensions(axis: 0), cache: cache)
        // p distributions for positions n .. n+k (k+1 of them).
        var pDists: [[Float]] = [nextDist]
        for i in 0..<k {
            pDists.append(Sampling.tempSoftmax(Self.row(logits, i), temperature: temperature))
        }

        var j = 0
        while j < k {
            let p = pDists[j][drafts[j]]
            let q = draftDists[j][drafts[j]]
            if rng.uniform() < min(1, p / max(q, 1e-9)) { j += 1 } else { break }
        }

        if j < k {
            let correction = Sampling.sample(
                Sampling.residual(pDists[j], draftDists[j]), rng: &rng)
            for c in cache { _ = c.trim(k - j) }
            let logits2 = model(MLXArray([correction]).expandedDimensions(axis: 0), cache: cache)
            nextDist = Sampling.tempSoftmax(Self.row(logits2, -1), temperature: temperature)
            return Outcome(accepted: Array(drafts[0..<j]) + [correction], acceptedDrafts: j)
        } else {
            let bonus = Sampling.sample(pDists[k], rng: &rng)
            let logits2 = model(MLXArray([bonus]).expandedDimensions(axis: 0), cache: cache)
            nextDist = Sampling.tempSoftmax(Self.row(logits2, -1), temperature: temperature)
            return Outcome(accepted: drafts + [bonus], acceptedDrafts: k)
        }
    }

    /// Target-only sampled baseline from the same primed state and RNG family.
    public func generateBaseline(n: Int, rng: inout SplitMix64) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(n)
        while out.count < n {
            let token = Sampling.sample(nextDist, rng: &rng)
            out.append(token)
            let logits = model(MLXArray([token]).expandedDimensions(axis: 0), cache: cache)
            nextDist = Sampling.tempSoftmax(Self.row(logits, -1), temperature: temperature)
        }
        return out
    }
}
