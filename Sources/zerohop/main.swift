import Foundation
import HarnessCore
import HarnessM0
import HarnessM1
import HarnessM2

// Hand-rolled CLI (no third-party dependencies, spec §11):
//   zerohop m0 [--warmup N] [--iters N]
//   zerohop m1 [--warmup N] [--iters N] [--model PATH] [--e3 sync|async|naive]
//              [--e4 default|qos-ui|rt] [--e5 none|500|100|50|10|saturated]

func parseFlags(_ args: [String]) -> [String: String] {
    var flags: [String: String] = [:]
    var i = 0
    while i < args.count {
        if args[i].hasPrefix("--"), i + 1 < args.count {
            flags[String(args[i].dropFirst(2))] = args[i + 1]
            i += 2
        } else {
            i += 1
        }
    }
    return flags
}

let args = Array(CommandLine.arguments.dropFirst())
let resultsRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent("Results")

guard let command = args.first else {
    print("usage: zerohop <m0|m1> [--warmup N] [--iters N] ...")
    exit(64)
}
let flags = parseFlags(Array(args.dropFirst()))
let warmup = Int(flags["warmup"] ?? "500") ?? 500
let iters = Int(flags["iters"] ?? "10000") ?? 10000

do {
    switch command {
    case "m0":
        try M0Baseline.run(resultsRoot: resultsRoot, warmup: warmup, iterations: iters)
    case "m1":
        var config = M1Config(
            modelPath: flags["model"] ?? "Models/m1_tiny.mlmodelc",
            completionStyle: M1Config.CompletionStyle(rawValue: flags["e3"] ?? "sync") ?? .sync,
            policy: ThreadPolicy(rawValue: flags["e4"] ?? "rt") ?? .timeConstraint,
            warmth: flags["e5"] ?? "none",
            warmup: warmup, iterations: iters
        )
        config.gpuWarm = flags["gpu-warm"] ?? "none"
        try M1Harness.run(resultsRoot: resultsRoot, config: config)
    case "m2":
        let config = M2Config(
            modelPathK8: flags["model-k8"] ?? "Models/m2_logits_k8.mlmodelc",
            modelPathK1: flags["model-k1"] ?? "Models/m2_logits_k1.mlmodelc",
            readPath: M2Config.ReadPath(rawValue: flags["e1"] ?? "A") ?? .A,
            granularity: M2Config.Granularity(rawValue: flags["e8"] ?? "multi") ?? .multi,
            mlockAttempt: flags["mlock"] == "on",
            pressure: flags["pressure"] == "on",
            gpuWarm: flags["gpu-warm"] ?? "none",
            warmup: warmup, iterations: iters
        )
        try M2Harness.run(resultsRoot: resultsRoot, config: config)
    case "pressure-helper":
        PressureHelper.runLoop(megabytes: Int(flags["mb"] ?? "4096") ?? 4096)
    case "placement":
        // Standalone MLComputePlan report for any compiled model (spec §8).
        guard let path = flags["model"] else {
            print("usage: zerohop placement --model <path.mlmodelc>")
            exit(64)
        }
        _ = try M1Model.loadWithPlacementAssert(compiledModelURL: URL(fileURLWithPath: path))
        print("placement assert PASSED")
    default:
        print("unknown command '\(command)'")
        exit(64)
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
