import HarnessCore
import Testing

@Suite struct FlagsTests {
    @Test func parsesNameValuePairs() throws {
        let flags = try Flags.parse(["--e3", "sync", "--gpu-warm", "saturated", "--iters", "10000"])
        #expect(flags == ["e3": "sync", "gpu-warm": "saturated", "iters": "10000"])
        #expect(try Flags.parse([]) == [:])
    }

    @Test(arguments: [
        ["--iters"],                  // trailing flag, no value
        ["--e3", "--e4", "rt"],       // value-less flag would swallow the next flag
        ["m1"],                       // stray positional
        ["--", "x"],                  // empty flag name
    ])
    func rejectsMalformedArguments(_ args: [String]) {
        #expect(throws: HarnessError.self) { try Flags.parse(args) }
    }
}
