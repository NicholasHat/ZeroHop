import CoreML
import Foundation
import HarnessCore

/// ANE lane: model loading with mandatory placement verification (spec §8).
/// The compute-unit setting is a request, not a guarantee; a silent CPU/GPU
/// fallback would invalidate the entire premise, so placement is asserted
/// before anything is timed.
public enum M1Model {

    public static func loadWithPlacementAssert(compiledModelURL: URL) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        let model = try MLModel(contentsOf: compiledModelURL, configuration: config)
        let report = try placementReport(compiledModelURL: compiledModelURL, configuration: config)
        print("ANE placement report:")
        for line in report.lines { print("  \(line)") }
        guard report.aneOpCount > 0 else {
            throw HarnessError("PLACEMENT ASSERT FAILED: no operation prefers the Neural Engine. "
                + "Measuring this configuration would be meaningless (spec §8).")
        }
        return model
    }

    public struct PlacementReport {
        public var lines: [String] = []
        public var aneOpCount = 0
        public var otherOpCount = 0
    }

    static func placementReport(compiledModelURL: URL,
                                configuration: MLModelConfiguration) throws -> PlacementReport {
        // Bridge the async MLComputePlan API into this synchronous CLI.
        let sem = DispatchSemaphore(value: 0)
        var result: Result<MLComputePlan, Error>!
        Task {
            do {
                result = .success(try await MLComputePlan.load(
                    contentsOf: compiledModelURL, configuration: configuration))
            } catch {
                result = .failure(error)
            }
            sem.signal()
        }
        sem.wait()
        let plan = try result.get()

        var report = PlacementReport()
        guard case .program(let program) = plan.modelStructure,
              let main = program.functions["main"] else {
            report.lines.append("model is not an ML Program with a main function")
            return report
        }
        for op in main.block.operations {
            guard let usage = plan.deviceUsage(for: op) else { continue }
            let preferred = deviceName(usage.preferred)
            let supported = usage.supported.map(deviceName).joined(separator: ",")
            report.lines.append("\(op.operatorName): preferred=\(preferred) supported=[\(supported)]")
            if case .neuralEngine = usage.preferred {
                report.aneOpCount += 1
            } else {
                report.otherOpCount += 1
            }
        }
        return report
    }

    static func deviceName(_ device: MLComputeDevice) -> String {
        switch device {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "ane"
        @unknown default: return "unknown"
        }
    }
}

public struct HarnessError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}

/// Page-aligned Float16 output backing (the header-sanctioned Option B form)
/// plus the E2 identity assertion. Used at M1 already so the assert machinery
/// is exercised before M2 depends on it.
public final class AlignedBacking {
    public let pointer: UnsafeMutableRawPointer
    public let byteCount: Int
    public let array: MLMultiArray

    public init(shape: [Int]) throws {
        let elements = shape.reduce(1, *)
        let bytes = vm_size_t(elements * MemoryLayout<Float16>.size)
        byteCount = Int((bytes + vm_page_size - 1) & ~(vm_page_size - 1))
        guard let p = aligned_alloc(Int(vm_page_size), byteCount) else {
            throw HarnessError("aligned_alloc failed")
        }
        // Pre-touch every page (spec §7 item 3).
        memset(p, 0, byteCount)
        pointer = p
        var strides: [NSNumber] = []
        var acc = 1
        for dim in shape.reversed() {
            strides.insert(NSNumber(value: acc), at: 0)
            acc *= dim
        }
        array = try MLMultiArray(
            dataPointer: p,
            shape: shape.map { NSNumber(value: $0) },
            dataType: .float16,
            strides: strides,
            deallocator: { free($0) }
        )
    }

    /// E2: was the backing honored, or did CoreML silently substitute an
    /// internal buffer (copy path)? Identity of the base address decides.
    public func isHonored(by output: MLMultiArray) -> Bool {
        output.withUnsafeBytes { $0.baseAddress == UnsafeRawPointer(pointer) }
    }
}
