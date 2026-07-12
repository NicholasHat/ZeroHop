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
    private let logitsBacking: AlignedBacking

    private let idsPtr: UnsafeMutablePointer<Int32>
    private let posPtr: UnsafeMutablePointer<Int32>
    private let maskPtr: UnsafeMutablePointer<Float16>
    private let logitsPtr: UnsafePointer<Float16>
    private let vocab: Int
    private let context: Int

    private var pos = 0
    private var pendingIngest: [Int]
    private static let masked: Float16 = -30_000

    public init(compiledModelURL: URL, prompt: [Int]) throws {
        model = try M1Model.loadWithPlacementAssert(compiledModelURL: compiledModelURL)
        state = model.makeState()

        guard let outDesc = model.modelDescription.outputDescriptionsByName["logits"],
              let outShape = outDesc.multiArrayConstraint?.shape.map(\.intValue),
              let maskDesc = model.modelDescription.inputDescriptionsByName["causalMask"],
              let maskShape = maskDesc.multiArrayConstraint?.shape.map(\.intValue) else {
            throw HarnessError("draft model missing logits/causalMask descriptions")
        }
        vocab = outShape.last!
        context = maskShape.last!

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

        logitsBacking = try AlignedBacking(shape: outShape)
        logitsPtr = UnsafeRawPointer(logitsBacking.pointer).assumingMemoryBound(to: Float16.self)
        options = MLPredictionOptions()
        options.outputBackings = ["logits": logitsBacking.array]
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

    /// Feed one token at the current position; logits land in the backing.
    private func feed(_ token: Int) throws {
        idsPtr[0] = Int32(token)
        posPtr[0] = Int32(pos)
        maskPtr[pos] = 0
        let out = try model.prediction(from: features, using: state, options: options)
        if let arr = out.featureValue(for: "logits")?.multiArrayValue,
           !logitsBacking.isHonored(by: arr) {
            throw HarnessError("E2 violation: draft logits backing not honored")
        }
        pos += 1
    }

    private func argmaxLogits() -> Int {
        var best = 0
        var bestVal = logitsPtr[0]
        for i in 1..<vocab where logitsPtr[i] > bestVal {
            bestVal = logitsPtr[i]
            best = i
        }
        return best
    }

    public func propose(k: Int, ingest: [Int]) throws -> [Int] {
        for token in pendingIngest + ingest { try feed(token) }
        pendingIngest = []
        var drafts: [Int] = []
        drafts.reserveCapacity(k)
        for _ in 0..<k {
            let next = argmaxLogits()
            drafts.append(next)
            if drafts.count < k { try feed(next) }
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
