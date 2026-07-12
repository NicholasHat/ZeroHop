import os

/// os_signpost intervals per spec §8. Names are fixed so Instruments sessions
/// (Core ML template + Metal System Trace + Points of Interest) line up across
/// runs: ane_dispatch, ane_compute, handoff, gpu_wait_release, gpu_verify.
public enum Signposts {
    public static let log = OSLog(subsystem: "dev.zerohop.harness", category: .pointsOfInterest)

    public static let aneDispatch: StaticString = "ane_dispatch"
    public static let aneCompute: StaticString = "ane_compute"
    public static let handoff: StaticString = "handoff"
    public static let gpuWaitRelease: StaticString = "gpu_wait_release"
    public static let gpuVerify: StaticString = "gpu_verify"
}
