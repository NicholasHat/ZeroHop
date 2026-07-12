import Darwin

/// Raw-sample recorder with exact percentiles. Storage is preallocated so the
/// measured loop performs a single indexed store — no allocation, no ARC.
/// Percentiles are computed exactly by sorting after the run; the HDR-style
/// log2 bucket view is for terminal rendering only.
public final class SampleRecorder {
    public let name: String
    public let capacity: Int
    private var samples: UnsafeMutableBufferPointer<UInt64>
    private var flags: UnsafeMutableBufferPointer<UInt8> // bit0: thermal-ok
    public private(set) var count: Int = 0

    public init(name: String, capacity: Int) {
        self.name = name
        self.capacity = capacity
        self.samples = .allocate(capacity: capacity)
        self.flags = .allocate(capacity: capacity)
        // Pre-touch every page (spec §7 item 3: first-touch faults cost 100s of µs).
        samples.initialize(repeating: 0)
        flags.initialize(repeating: 1)
    }

    deinit {
        samples.deallocate()
        flags.deallocate()
    }

    @inline(__always)
    public func record(_ ns: UInt64, thermalOK: Bool = true) {
        guard count < capacity else { return }
        samples[count] = ns
        flags[count] = thermalOK ? 1 : 0
        count += 1
    }

    public var rawSamples: [UInt64] { Array(samples[0..<count]) }
    public var rawFlags: [UInt8] { Array(flags[0..<count]) }

    public struct Summary: Codable {
        public let name: String
        public let count: Int
        public let discardedThermal: Int
        public let p50, p95, p99, p999: UInt64
        public let min, max: UInt64
        public let mean: Double // reported, never a criterion (spec §2)
    }

    /// Exact percentiles over thermally-clean samples.
    public func summarize() -> Summary {
        var clean: [UInt64] = []
        clean.reserveCapacity(count)
        var discarded = 0
        for i in 0..<count {
            if flags[i] == 1 { clean.append(samples[i]) } else { discarded += 1 }
        }
        if clean.isEmpty { clean = [0] }
        clean.sort()
        func pct(_ p: Double) -> UInt64 {
            let idx = Int((Double(clean.count - 1) * p).rounded())
            return clean[idx]
        }
        let total = clean.reduce(0.0) { $0 + Double($1) }
        return Summary(
            name: name, count: clean.count, discardedThermal: discarded,
            p50: pct(0.50), p95: pct(0.95), p99: pct(0.99), p999: pct(0.999),
            min: clean.first!, max: clean.last!, mean: total / Double(clean.count)
        )
    }

    /// Terminal-friendly HDR-style histogram: log2 buckets from 256 ns up.
    /// Makes bimodality (e.g. cold-dispatch ramps, spec §7 item 2) visible.
    public func bucketRender(width: Int = 50) -> String {
        var buckets: [Int] = Array(repeating: 0, count: 40)
        for i in 0..<count where flags[i] == 1 {
            let v = max(samples[i], 1)
            // bucket b holds v with floor(log2 v) == b+8, i.e. v in [2^(b+8), 2^(b+9))
            let b = min(max(63 - v.leadingZeroBitCount - 8, 0), buckets.count - 1)
            buckets[b] += 1
        }
        let peak = max(buckets.max() ?? 1, 1)
        var out = ""
        for (b, n) in buckets.enumerated() where n > 0 {
            let lo = UInt64(1) << (b + 8)
            let label = Self.fmt(lo)
            let pad = String(repeating: " ", count: max(0, 10 - label.count))
            let bar = String(repeating: "#", count: max(1, n * width / peak))
            out += "  \(pad)\(label) | \(bar) \(n)\n"
        }
        return out
    }

    public static func fmt(_ ns: UInt64) -> String {
        switch ns {
        case ..<1_000: return "\(ns) ns"
        case ..<1_000_000: return String(format: "%.1f µs", Double(ns) / 1e3)
        case ..<1_000_000_000: return String(format: "%.3f ms", Double(ns) / 1e6)
        default: return String(format: "%.3f s", Double(ns) / 1e9)
        }
    }
}
