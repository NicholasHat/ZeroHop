/// The harness's one error type: a message for the operator. Runs fail loudly
/// rather than measure the wrong configuration.
public struct HarnessError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}
