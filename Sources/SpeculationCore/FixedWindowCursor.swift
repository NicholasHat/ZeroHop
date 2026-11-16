import HarnessCore

/// Position bookkeeping for a fixed-window (static-shape) KV cache, as the
/// ANE draft uses: the model always attends over all `context` slots and an
/// additive mask hides the unwritten ones. Advancing claims the next slot to
/// unmask; rolling back hands back the last n slots to re-mask — O(1) per
/// token, no cache surgery (stale entries are invisible, then overwritten).
/// The caller owns the mask buffer; this owns the arithmetic and its bounds.
public struct FixedWindowCursor {
    public let context: Int
    public private(set) var position = 0

    public init(context: Int) {
        precondition(context > 0, "context window must be positive")
        self.context = context
    }

    /// Claim the next slot. Past the window the mask write would land outside
    /// its buffer and the one-hot cache write would match no slot — fail
    /// loudly instead of silently degrading the draft.
    public mutating func advance() throws -> Int {
        guard position < context else {
            throw HarnessError("draft context window exhausted (\(context) positions): "
                + "prompt + generated tokens + speculation must fit; use a smaller --n")
        }
        defer { position += 1 }
        return position
    }

    /// Release the last n slots; returns them for re-masking.
    public mutating func rewind(_ n: Int) throws -> Range<Int> {
        guard n >= 0, n <= position else {
            throw HarnessError("draft rollback of \(n) exceeds the \(position) fed positions")
        }
        position -= n
        return position..<(position + n)
    }
}
