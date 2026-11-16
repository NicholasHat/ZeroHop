import HarnessCore
import SpeculationCore
import Testing

@Suite struct FixedWindowCursorTests {
    @Test func fillsTheWindowThenFailsLoudly() throws {
        var c = FixedWindowCursor(context: 3)
        #expect(try [c.advance(), c.advance(), c.advance()] == [0, 1, 2])
        #expect(throws: HarnessError.self) { try c.advance() }
        #expect(c.position == 3)
    }

    @Test func rewindHandsBackTheSlotsToRemaskAndReusesThem() throws {
        var c = FixedWindowCursor(context: 8)
        for _ in 0..<5 { _ = try c.advance() }
        #expect(try c.rewind(2) == 3..<5)
        #expect(c.position == 3)
        #expect(try c.advance() == 3)
        #expect(try c.rewind(0) == 4..<4) // no-op rollback is legal
    }

    @Test func cannotRewindPastTheStart() throws {
        var c = FixedWindowCursor(context: 8)
        _ = try c.advance()
        #expect(throws: HarnessError.self) { try c.rewind(2) }
        #expect(c.position == 1)
    }
}
