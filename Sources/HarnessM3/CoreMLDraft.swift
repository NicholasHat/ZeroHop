import CoreML
import Foundation
import HarnessCore
import HarnessM1

/// The heterogeneous draft lane (M3.2): Llama-3.2-1B on the ANE via a
/// stateful CoreML model with fixed-window attention.
///
/// The model takes (inputIds [1,1] i32, causalMask [1,1,1,CTX] fp16,
/// firstPosition [1] i32) and returns logits [1,1,V] fp16; the KV cache lives
/// in MLState. Because attention always spans the full window and the mask
/// excludes unwritten slots, rollback is O(1): move the position back and
/// re-mask — stale cache entries are invisible and get overwritten.
public final class CoreMLDraft: DraftTokenSource {
    private let model: MLModel
    private let state: MLState
    private let options: MLPredictionOptions
    private let features: MLDictionaryFeatureProvider
    private let logitsBacking: AlignedBacking?
    private let tokenMode: Bool

    private let idsPtr: UnsafeMutablePointer<Int32>
    private let posPtr: UnsafeMutablePointer<Int32>
    private let maskPtr: UnsafeMutablePointer<Float16>
    private let logitsPtr: UnsafePointer<Float16>?
    private let vocab: Int
    private let context: Int

    private var pos = 0
    private var pendingIngest: [Int]
    private static let masked: Float16 = -30_000

    public init(compiledModelURL: URL, prompt: [Int]) throws {
        model = try M1Model.loadWithPlacementAssert(compiledModelURL: compiledModelURL)
        state = model.makeState()

        guard let maskDesc = model.modelDescription.inputDescriptionsByName["causalMask"],
              let maskShape = maskDesc.multiArrayConstraint?.shape.map(\.intValue) else {
            throw HarnessError("draft model missing causalMask description")
        }
        context = maskShape.last!
        // Two model interfaces: newer exports fuse the greedy reduction and
        // return "token" [1,1] i32 (4 bytes/call); older ones return "logits"
        // [1,1,V] fp16 and we argmax on the CPU.
        let outputs = model.modelDescription.outputDescriptionsByName
        tokenMode = outputs["token"] != nil
        if tokenMode {
            vocab = 0
        } else if let outShape = outputs["logits"]?.multiArrayConstraint?.shape.map(\.intValue) {
            vocab = outShape.last!
        } else {
            throw HarnessError("draft model has neither 'token' nor 'logits' output")
        }

        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let first = try MLMultiArray(shape: [1], dataType: .int32)
        let mask = try MLMultiArray(shape: maskShape.map { NSNumber(value: $0) }, dataType: .float16)
        idsPtr = ids.withUnsafeMutableBytes { raw, _ in
            raw.bindMemory(to: Int32.self).baseAddress!
        }
        posPtr = first.withUnsafeMutableBytes { raw, _ in
            raw.bindMemory(to: Int32.self).baseAddress!
        }
        maskPtr = mask.withUnsafeMutableBytes { raw, _ in
            raw.bindMemory(to: Float16.self).baseAddress!
        }
        for i in 0..<context { maskPtr[i] = Self.masked }

        options = MLPredictionOptions()
        if tokenMode {
            logitsBacking = nil
            logitsPtr = nil
        } else {
            let backing = try AlignedBacking(shape: [1, 1, vocab])
            logitsBacking = backing
            logitsPtr = UnsafeRawPointer(backing.pointer).assumingMemoryBound(to: Float16.self)
            options.outputBackings = ["logits": backing.array]
        }
        features = try MLDictionaryFeatureProvider(dictionary: [
            "inputIds": MLFeatureValue(multiArray: ids),
            "causalMask": MLFeatureValue(multiArray: mask),
            "firstPosition": MLFeatureValue(multiArray: first),
        ])

        precondition(prompt.count < context, "prompt exceeds draft context window")
        pendingIngest = []
        // Prefill one token at a time (static [1,1] model); one-time cost.
        for token in prompt.dropLast() { try feed(token) }
        pendingIngest = [prompt.last!]
    }

    /// Feed one token at the current position; returns the model's greedy
    /// next token (from the fused head, or CPU argmax over the logits).
    @discardableResult
    private func feed(_ token: Int) throws -> Int {
        idsPtr[0] = Int32(token)
        posPtr[0] = Int32(pos)
        maskPtr[pos] = 0
        let out = try model.prediction(from: features, using: state, options: options)
        pos += 1
        if tokenMode {
            guard let arr = out.featureValue(for: "token")?.multiArrayValue else {
                throw HarnessError("draft output missing 'token'")
            }
            return arr[0].intValue
        }
        if let arr = out.featureValue(for: "logits")?.multiArrayValue,
           let backing = logitsBacking, !backing.isHonored(by: arr) {
            throw HarnessError("E2 violation: draft logits backing not honored")
        }
        return argmaxLogits()
    }

    private func argmaxLogits() -> Int {
        guard let logitsPtr else { return 0 }
        var best = 0
        var bestVal = logitsPtr[0]
        for i in 1..<vocab where logitsPtr[i] > bestVal {
            bestVal = logitsPtr[i]
            best = i
        }
        return best
    }

    public func propose(k: Int, ingest: [Int]) throws -> [Int] {
        precondition(!(pendingIngest + ingest).isEmpty, "propose needs at least one token to feed")
        var next = 0
        for token in pendingIngest + ingest { next = try feed(token) }
        pendingIngest = []
        var drafts: [Int] = []
        drafts.reserveCapacity(k)
        for _ in 0..<k {
            drafts.append(next)
            if drafts.count < k { next = try feed(next) }
        }
        pendingIngest = [drafts[k - 1]]
        return drafts
    }

    public func rollback(_ n: Int) throws {
        for _ in 0..<n {
            pos -= 1
            maskPtr[pos] = Self.masked
        }
        pendingIngest = []
    }
}

extension CoreMLDraft: SamplingDraftSource {
    /// Stochastic proposals need the full q distribution per position, so
    /// this path requires the logits-interface export (not the fused head).
    public func proposeSampled(k: Int, ingest: [Int], temperature: Float,
                               rng: inout SplitMix64) throws -> (tokens: [Int], dists: [[Float]]) {
        guard !tokenMode, let logitsPtr else {
            throw HarnessError("sampling requires a logits-interface draft model (fused-head export cannot provide q)")
        }
        for token in pendingIngest + ingest { try feed(token) }
        pendingIngest = []
        var tokens: [Int] = []
        var dists: [[Float]] = []
        for _ in 0..<k {
            var raw = [Float](repeating: 0, count: vocab)
            for i in 0..<vocab { raw[i] = Float(logitsPtr[i]) }
            let q = Sampling.tempSoftmax(raw, temperature: temperature)
            let next = Sampling.sample(q, rng: &rng)
            tokens.append(next)
            dists.append(q)
            if tokens.count < k { try feed(next) }
        }
        pendingIngest = [tokens[k - 1]]
        return (tokens, dists)
    }
}
