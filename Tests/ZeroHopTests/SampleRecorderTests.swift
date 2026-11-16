import HarnessCore
import Testing

@Suite struct SampleRecorderTests {
    @Test func exactPercentilesOverSortedSamples() {
        let r = SampleRecorder(name: "x", capacity: 100)
        // Recorded out of order: summarize must sort.
        for v in (1...100).reversed() { r.record(UInt64(v)) }
        let s = r.summarize()
        #expect(s.count == 100)
        #expect(s.discardedThermal == 0)
        // Nearest-rank on index round((n-1)·p): 49.5→50, 94.05→94, 98.01→98, 98.9→99.
        #expect([s.p50, s.p95, s.p99, s.p999] == [51, 95, 99, 100])
        #expect(s.min == 1 && s.max == 100)
        #expect(s.mean == 50.5)
    }

    @Test func thermallyFlaggedSamplesAreExcludedButCounted() {
        let r = SampleRecorder(name: "x", capacity: 8)
        r.record(10)
        r.record(5_000_000, thermalOK: false) // a throttled outlier
        r.record(30)
        r.record(20)
        let s = r.summarize()
        #expect(s.count == 3)
        #expect(s.discardedThermal == 1)
        #expect(s.max == 30)
        #expect(s.p50 == 20)
        // The raw export keeps every sample and its flag.
        #expect(r.rawSamples == [10, 5_000_000, 30, 20])
        #expect(r.rawFlags == [1, 0, 1, 1])
    }

    @Test func allSamplesDiscardedYieldsAnEmptySummaryNotZeroLatency() {
        let r = SampleRecorder(name: "x", capacity: 4)
        for v: UInt64 in [100, 200, 300] { r.record(v, thermalOK: false) }
        let s = r.summarize()
        #expect(s.count == 0)
        #expect(s.discardedThermal == 3)
        #expect(s.p50 == 0 && s.max == 0 && s.mean == 0)
    }

    @Test func recordingPastCapacityIsDropped() {
        let r = SampleRecorder(name: "x", capacity: 2)
        for v: UInt64 in [1, 2, 3] { r.record(v) }
        #expect(r.count == 2)
        #expect(r.rawSamples == [1, 2])
    }
}
