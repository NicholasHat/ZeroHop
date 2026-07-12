import Foundation
import HarnessCore
import HarnessM0
import HarnessM1

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
        let config = M1Config(
            modelPath: flags["model"] ?? "Models/m1_tiny.mlmodelc",
            completionStyle: M1Config.CompletionStyle(rawValue: flags["e3"] ?? "sync") ?? .sync,
            policy: ThreadPolicy(rawValue: flags["e4"] ?? "rt") ?? .timeConstraint,
            warmth: flags["e5"] ?? "none",
            warmup: warmup, iterations: iters
        )
        try M1Harness.run(resultsRoot: resultsRoot, config: config)
    default:
        print("unknown command '\(command)'")
        exit(64)
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
