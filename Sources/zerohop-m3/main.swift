import Foundation
import HarnessCore
import HarnessM3

// zerohop-m3 — M3 speculative-decoding runner (separate binary so the M0–M2
// measurement executable keeps zero third-party dependencies, spec §11).
//
//   zerohop-m3 [--target <hf-id>] [--draft <hf-id>] [--k N] [--n N] [--prompt S]

func parseFlags(_ args: [String]) -> [String: String] {
    var flags: [String: String] = [:]
    var i = 0
    while i < args.count {
        if args[i].hasPrefix("--"), i + 1 < args.count {
            flags[String(args[i].dropFirst(2))] = args[i + 1]
            i += 2
        } else { i += 1 }
    }
    return flags
}

let flags = parseFlags(Array(CommandLine.arguments.dropFirst()))
var config = M3Config(
    targetID: flags["target"] ?? "mlx-community/Llama-3.2-3B-Instruct-4bit",
    draftID: flags["draft"] ?? "mlx-community/Llama-3.2-1B-Instruct-4bit",
    k: Int(flags["k"] ?? "4") ?? 4,
    tokens: Int(flags["n"] ?? "200") ?? 200,
    prompt: flags["prompt"] ?? "The key ideas behind speculative decoding are"
)
config.coremlDraftPath = flags["coreml-draft"]
let resultsRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent("Results")

do {
    try await M3Runner.run(resultsRoot: resultsRoot, config: config)
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
