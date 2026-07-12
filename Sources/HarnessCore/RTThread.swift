import Darwin

/// Thread policies under test (experiment axis E4). Applied to the calling
/// thread from inside the thread body.
public enum ThreadPolicy: String, CaseIterable, Codable {
    case `default` = "default"
    case qosUserInteractive = "qos-ui"
    case timeConstraint = "rt"

    /// Apply this policy to the current thread. Returns a human-readable
    /// failure description, or nil on success.
    @discardableResult
    public func applyToCurrentThread() -> String? {
        switch self {
        case .default:
            return nil
        case .qosUserInteractive:
            let err = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
            return err == 0 ? nil : "pthread_set_qos_class_self_np failed: \(err)"
        case .timeConstraint:
            // Conservative bounds so the scheduler does not demote us for
            // overrunning the declared computation (period 1 ms, computation
            // 200 µs, constraint 500 µs), spec §4.2 / PLAN §3.
            var policy = thread_time_constraint_policy(
                period: UInt32(MachClock.fromNanos(1_000_000)),
                computation: UInt32(MachClock.fromNanos(200_000)),
                constraint: UInt32(MachClock.fromNanos(500_000)),
                preemptible: 1
            )
            let thread = pthread_mach_thread_np(pthread_self())
            let count = mach_msg_type_number_t(
                MemoryLayout<thread_time_constraint_policy>.size / MemoryLayout<integer_t>.size)
            let kr = withUnsafeMutablePointer(to: &policy) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    thread_policy_set(thread, thread_policy_flavor_t(THREAD_TIME_CONSTRAINT_POLICY), $0, count)
                }
            }
            return kr == KERN_SUCCESS ? nil : "thread_policy_set(TIME_CONSTRAINT) failed: \(kr)"
        }
    }
}

/// A dedicated pthread (never a Dispatch queue — spec §4.2). The body runs
/// with the requested policy already applied; policy-application failures
/// abort the run rather than silently measuring the wrong configuration.
public final class DedicatedThread {
    private var thread: pthread_t?
    private let box: Box

    private final class Box {
        let policy: ThreadPolicy
        let body: () -> Void
        init(policy: ThreadPolicy, body: @escaping () -> Void) {
            self.policy = policy
            self.body = body
        }
    }

    public init(policy: ThreadPolicy, body: @escaping () -> Void) {
        self.box = Box(policy: policy, body: body)
        let arg = Unmanaged.passRetained(box).toOpaque()
        var t: pthread_t?
        let rc = pthread_create(&t, nil, { raw in
            let box = Unmanaged<Box>.fromOpaque(raw).takeRetainedValue()
            if let err = box.policy.applyToCurrentThread() {
                fatalError("thread policy '\(box.policy.rawValue)' not applied: \(err)")
            }
            box.body()
            return nil
        }, arg)
        precondition(rc == 0, "pthread_create failed: \(rc)")
        self.thread = t
    }

    public func join() {
        if let t = thread { pthread_join(t, nil) }
        thread = nil
    }
}

/// Raw mach semaphore — the cross-thread wake primitive for the handoff path.
/// Chosen over DispatchSemaphore/pthread_cond to keep libdispatch and mutexes
/// off the critical path (spec §4.2).
public final class MachSemaphore {
    private var sem = semaphore_t()

    public init() {
        let kr = semaphore_create(mach_task_self_, &sem, SYNC_POLICY_FIFO, 0)
        precondition(kr == KERN_SUCCESS, "semaphore_create failed: \(kr)")
    }

    deinit { semaphore_destroy(mach_task_self_, sem) }

    @inline(__always) public func signal() { semaphore_signal(sem) }
    @inline(__always) public func wait() { semaphore_wait(sem) }
}
