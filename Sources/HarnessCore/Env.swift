import Foundation

/// Environment capture for result metadata (spec §11: every result file
/// records chip, macOS build, power state, thermal pressure).
public struct EnvInfo: Codable {
    public let chip: String
    public let macOSVersion: String
    public let macOSBuild: String
    public let hostname: String
    public let powerSource: String
    public let thermalStateAtStart: String
    public let timebaseNumer: UInt32
    public let timebaseDenom: UInt32
    public let capturedAt: String

    public static func capture() -> EnvInfo {
        EnvInfo(
            chip: sysctlString("machdep.cpu.brand_string") ?? "unknown",
            macOSVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            macOSBuild: sysctlString("kern.osversion") ?? "unknown",
            hostname: ProcessInfo.processInfo.hostName,
            powerSource: currentPowerSource(),
            thermalStateAtStart: thermalStateName(),
            timebaseNumer: MachClock.timebase.numer,
            timebaseDenom: MachClock.timebase.denom,
            capturedAt: ISO8601DateFormatter().string(from: Date())
        )
    }

    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
        return String(cString: buf)
    }

    static func currentPowerSource() -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
        p.arguments = ["-g", "batt"]
        let pipe = Pipe()
        p.standardOutput = pipe
        guard (try? p.run()) != nil else { return "unknown" }
        p.waitUntilExit()
        let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        if out.contains("AC Power") { return "AC" }
        if out.contains("Battery Power") { return "battery" }
        return "unknown"
    }

    public static func thermalStateName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

/// 1 Hz thermal-state sampler on a background (non-critical) thread. The
/// measured loop polls `isNominal` with a relaxed atomic-load-equivalent read;
/// non-nominal iterations are flagged and excluded from headline histograms.
public final class ThermalWatcher {
    private var timeline: [(String, String)] = []
    private let lock = NSLock()
    private var stopFlag = false
    private var thread: Thread?
    // Written by the watcher thread, read by the measured loop. A torn read is
    // impossible for a Bool; staleness of <1s is acceptable for a thermal gate.
    public private(set) var isNominal: Bool = true

    public init() {}

    public func start() {
        let t = Thread { [weak self] in
            while let self, !self.stopFlag {
                let name = EnvInfo.thermalStateName()
                self.isNominal = (name == "nominal")
                self.lock.lock()
                if self.timeline.last?.1 != name {
                    self.timeline.append((ISO8601DateFormatter().string(from: Date()), name))
                }
                self.lock.unlock()
                Thread.sleep(forTimeInterval: 1.0)
            }
        }
        t.qualityOfService = .utility
        t.start()
        thread = t
    }

    public func stop() -> [[String]] {
        stopFlag = true
        lock.lock(); defer { lock.unlock() }
        return timeline.map { [$0.0, $0.1] }
    }
}
