import Foundation

/// One run directory per matrix cell:
///   Results/<timestamp>-<milestone>-<cell>/{meta.json, samples.csv, summary.json}
/// All raw measurements exported machine-readable with metadata (spec §11).
public final class ResultSink {
    public let dir: URL

    public struct Meta: Codable {
        public let milestone: String
        public let cell: [String: String]
        public let env: EnvInfo
        public let warmupIterations: Int
        public let measuredIterations: Int
        public var thermalTimeline: [[String]] = []
        public var notes: [String] = []
        public var clockCorrelation: [String: [String: UInt64]] = [:]

        public init(milestone: String, cell: [String: String], env: EnvInfo,
                    warmupIterations: Int, measuredIterations: Int) {
            self.milestone = milestone
            self.cell = cell
            self.env = env
            self.warmupIterations = warmupIterations
            self.measuredIterations = measuredIterations
        }
    }

    public init(resultsRoot: URL, milestone: String, cellName: String) throws {
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "")
        dir = resultsRoot.appendingPathComponent("\(stamp)-\(milestone)-\(cellName)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    public func writeMeta(_ meta: Meta) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(meta).write(to: dir.appendingPathComponent("meta.json"))
    }

    public func writeSummaries(_ summaries: [SampleRecorder.Summary]) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(summaries).write(to: dir.appendingPathComponent("summary.json"))
    }

    /// Column-per-recorder CSV; all recorders must have equal counts.
    public func writeSamplesCSV(iterationsOf recorders: [SampleRecorder]) throws {
        guard let n = recorders.map(\.count).min() else { return }
        var csv = "iter," + recorders.map { "\($0.name)_ns" }.joined(separator: ",") + ",thermal_ok\n"
        csv.reserveCapacity(n * 32)
        let columns = recorders.map(\.rawSamples)
        let flags = recorders.first!.rawFlags
        for i in 0..<n {
            csv += "\(i)," + columns.map { String($0[i]) }.joined(separator: ",") + ",\(flags[i])\n"
        }
        try csv.data(using: .utf8)!.write(to: dir.appendingPathComponent("samples.csv"))
    }
}

public func printSummaryTable(_ summaries: [SampleRecorder.Summary]) {
    func lpad(_ s: String, _ w: Int) -> String {
        String(repeating: " ", count: max(0, w - s.count)) + s
    }
    func rpad(_ s: String, _ w: Int) -> String {
        s + String(repeating: " ", count: max(0, w - s.count))
    }
    let f = SampleRecorder.fmt
    print("  " + rpad("stage", 22) + ["p50", "p95", "p99", "p99.9", "max", "n"].map { lpad($0, 11) }.joined())
    for s in summaries {
        let cells = [f(s.p50), f(s.p95), f(s.p99), f(s.p999), f(s.max), String(s.count)]
        print("  " + rpad(s.name, 22) + cells.map { lpad($0, 11) }.joined())
    }
}
