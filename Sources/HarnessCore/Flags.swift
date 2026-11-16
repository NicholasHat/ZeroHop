/// Hand-rolled `--name value` parser shared by both binaries (no third-party
/// dependencies, spec §11). Every flag takes a value; anything else is a
/// usage error rather than silently ignored — `--overlap --k 6` must not run
/// with overlap = "--k" and the default k.
public enum Flags {
    public struct UsageError: Error, CustomStringConvertible {
        public let description: String
    }

    public static func parse(_ args: [String]) throws -> [String: String] {
        var flags: [String: String] = [:]
        var i = 0
        while i < args.count {
            let arg = args[i]
            guard arg.hasPrefix("--"), arg.count > 2 else {
                throw UsageError(description: "unexpected argument '\(arg)'")
            }
            guard i + 1 < args.count, !args[i + 1].hasPrefix("--") else {
                throw UsageError(description: "flag '\(arg)' needs a value")
            }
            flags[String(arg.dropFirst(2))] = args[i + 1]
            i += 2
        }
        return flags
    }
}
