import Darwin
import Metal

/// mach_absolute_time <-> nanosecond conversion. Timebase is read once; on
/// Apple Silicon the ratio is not 1:1 (typically 125/3), so conversion is
/// integer-exact via numer/denom.
public enum MachClock {
    public static let timebase: mach_timebase_info_data_t = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return info
    }()

    @inline(__always)
    public static func now() -> UInt64 { mach_absolute_time() }

    @inline(__always)
    public static func toNanos(_ machTicks: UInt64) -> UInt64 {
        machTicks * UInt64(timebase.numer) / UInt64(timebase.denom)
    }

    @inline(__always)
    public static func fromNanos(_ ns: UInt64) -> UInt64 {
        ns * UInt64(timebase.denom) / UInt64(timebase.numer)
    }
}

/// CPU/GPU clock-domain correlation (spec §8, mandatory). Sample a simultaneous
/// (CPU, GPU) timestamp pair before and after a measured block and map GPU
/// timestamps to CPU nanoseconds by linear interpolation between them, which
/// also absorbs drift.
public struct GPUClockCorrelation {
    public struct Pair { public let cpu: MTLTimestamp; public let gpu: MTLTimestamp }

    public let before: Pair
    public private(set) var after: Pair?

    public init(device: MTLDevice) {
        self.before = Self.sample(device)
    }

    public mutating func finish(device: MTLDevice) {
        self.after = Self.sample(device)
    }

    private static func sample(_ device: MTLDevice) -> Pair {
        let ts = device.sampleTimestamps()
        return Pair(cpu: ts.cpu, gpu: ts.gpu)
    }

    /// Map a raw GPU timestamp into the CPU mach_absolute_time domain
    /// (nanoseconds). Requires finish() to have been called.
    public func gpuToCPUNanos(_ gpuTS: MTLTimestamp) -> UInt64 {
        guard let after, after.gpu != before.gpu else {
            // Degenerate correlation window: fall back to the 'before' anchor.
            return before.cpu &+ (gpuTS &- before.gpu)
        }
        // Both timestamps from sampleTimestamps are nanoseconds: the CPU one
        // is in the mach host-time domain (NOT mach ticks — observed on M3,
        // where the GPU value is even numerically identical to the CPU one).
        // Interpolating between the two pairs maps any linear GPU clock and
        // absorbs drift; in double precision the error stays sub-µs.
        let frac = (Double(gpuTS) - Double(before.gpu)) / (Double(after.gpu) - Double(before.gpu))
        let cpuNs = Double(before.cpu) + frac * (Double(after.cpu) - Double(before.cpu))
        return UInt64(max(0, cpuNs))
    }
}
