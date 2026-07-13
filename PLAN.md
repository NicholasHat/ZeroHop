# Implementation Plan — Heterogeneous Speculative Decoding Harness (ZeroHop)

Response to `hsd-ane-gpu-spec.md` §12. M0–M2 in full detail, M3–M4 in outline.

**Test environment (recorded per spec §11):** Apple M3 · macOS 26.5.1 (build 25F80) · Xcode 26.4 · MacOSX SDK 26.4 · Swift toolchain `/usr/bin/swiftc`.

---

## 0. SDK verification results (spec §9 item 10, §12 item 2)

Every API named in the spec was checked against the MacOSX 26.4 SDK headers before planning.

| API | Verdict | Notes |
|---|---|---|
| `MLPredictionOptions.outputBackings` | ✅ macOS 11+ | Header doc sanctions exactly two backing types: page-aligned `MLMultiArray(dataPointer:)` (spec Option B) and IOSurface-backed `CVPixelBuffer` (spec Option A). Header explicitly recommends `aligned_alloc(vm_page_size, round_page(n))`. |
| `MLMultiArray.dataPointer` | ⚠️ **deprecated** | Exists but marked `API_DEPRECATED("Use getBytesWithHandler…")`. **Deviation from spec §4.1:** the E2 pointer-identity assertion uses `withUnsafeMutableBytes { $0.baseAddress }` for Option B, and `IOSurfaceGetID` / `CVPixelBuffer` identity via `MLMultiArray.pixelBuffer` (macOS 12+, verified) for Option A. Semantics unchanged; API surface updated. |
| `MLMultiArray(pixelBuffer:shape:)` | ✅ macOS 12+ | Requires `kCVPixelFormatType_OneComponent16Half` (`'L00h'`, verified in CoreVideo headers) — Float16 only, matching ANE native format. |
| `MLComputeUnits.cpuAndNeuralEngine` | ✅ macOS 13+ | No ANE-*only* option exists; CPU fallback is always possible → placement assertion is load-bearing (spec §8 already requires it). |
| `MLComputePlan` | ✅ macOS 14+ | `computeDeviceUsage(for:)` per ML-Program operation; loads from model asset/URL. |
| `predictionFromFeatures:options:completionHandler:` | ✅ | Async arm of E3. Sync `predictionFromFeatures:options:error:` also present. |
| `MTLSharedEvent` / `newSharedEvent` | ✅ macOS 10.14+ | `signaledValue` is `readwrite` — CPU-side signal from the handoff thread is supported, verified. |
| `encodeWaitForEvent:value:` | ✅ macOS 10.14+ | On `MTLCommandBuffer` (verified) — wait is encoded at command-buffer scope, before any encoder, exactly as §4.3 needs. |
| `makeBuffer(bytesNoCopy:length:options:deallocator:)` | ✅ | Page-aligned, page-multiple length required (enforced at runtime). |
| `makeTexture(descriptor:iosurface:plane:)` | ✅ | Option A GPU import path. |
| `makeBuffer(descriptor:offset:length:)` | ❌ **does not exist** | Confirmed absent from all Metal headers in SDK 26.4. Spec §9 item 6 upheld; not implemented. |
| `MTLDevice.sampleTimestamps(_:gpuTimestamp:)` | ✅ | Simultaneous (CPU, GPU) pair for clock-domain alignment. |
| `MTLCounterSampleBuffer` + sampling | ✅ | **Caveat:** Apple GPUs support *stage-boundary* sampling only, not dispatch-boundary. Plan uses `MTLComputePassDescriptor.sampleBufferAttachments` (start/end of pass) and asserts `supportsCounterSampling(.atStageBoundary)` at startup. `MTLCommandBuffer.gpuStartTime` (mach-time domain, seconds) is retained as an independent cross-check. |
| `THREAD_TIME_CONSTRAINT_POLICY` | ✅ | `mach/thread_policy.h`. Parameters in mach ticks; the OS may demote threads that overrun their stated computation — parameters chosen conservatively (see M0). |
| `IOSurfaceLock/Unlock`, `IOSurfaceGetID`, `IOSurfaceGetBaseAddress` | ✅ | Coherency ordering point + identity assertions. |
| `aned` daemon name | ✅ | `/usr/libexec/aned` exists on this machine. |
| `os_signpost`, `mach_absolute_time`, `mach_wait_until` | ✅ | Standard. |

Other environment notes:
- `powermetrics` requires root. The harness itself records `ProcessInfo.thermalState`, `pmset -g batt` (AC/battery), and a wall-clock window per run; `powermetrics` is launched by the run *script* under `sudo` when available, and its absence is recorded in metadata rather than blocking the run (spec §7 item 2, degraded gracefully and honestly).
- `coremltools` is a build-time tool only (generates the M1/M2/E8 models); it lives in `Tools/.venv` and never touches the harness binary — consistent with the no-third-party-deps-on-critical-path rule (§11).

---

## 1. Repository layout

```
ZeroHop/
├── hsd-ane-gpu-spec.md          # the spec (input)
├── PLAN.md                      # this document
├── Package.swift                # SwiftPM; zero external dependencies
├── Sources/
│   ├── HarnessCore/             # shared library
│   │   ├── Clock.swift          # mach_absolute_time↔ns, CPU/GPU clock correlation (drift-interpolated)
│   │   ├── Histogram.swift      # raw-sample recorder (preallocated), exact p50/p95/p99/p99.9, HDR-style log buckets for print
│   │   ├── RTThread.swift       # pthread creation: default / QoS user-interactive / TIME_CONSTRAINT; core-type observation
│   │   ├── Env.swift            # chip, macOS build, SDK, thermal state, power source, git SHA
│   │   ├── ResultSink.swift     # per-run dir: meta.json + samples.csv + summary.json
│   │   ├── Signposts.swift      # os_signpost intervals: ane_dispatch, ane_compute, handoff, gpu_wait_release, gpu_verify
│   │   └── Assertions.swift     # backing-identity, ANE-placement, canary check helpers; hard-fail before measured runs
│   ├── HarnessM0/               # baseline noise floor executable pieces
│   ├── HarnessM1/               # empty-model round-trip
│   ├── HarnessM2/               # real buffer shapes + backing A/B
│   └── zerohop/                 # CLI entry point: `zerohop m0|m1|m2 [--cell …]` (hand-rolled arg parsing, no deps)
├── Tools/
│   ├── make_models.py           # coremltools: M1 trivial ANE model, M2 logits-shape models, E8 multi-token variant
│   └── plot_results.py          # offline analysis (anything goes here)
├── Models/                      # generated .mlpackage / compiled .mlmodelc (gitignored, reproducible)
└── Results/                     # <timestamp>-<milestone>-<cell>/ {meta.json, samples.csv, summary.json}
```

## 2. Measurement infrastructure (used by all milestones)

**Raw samples, exact percentiles.** Each matrix cell preallocates `UnsafeMutableBufferPointer<UInt64>` arrays (one per instrumented sub-segment) before the measured loop; the loop writes by index — zero allocation, zero ARC traffic on the critical path (§4 design principle). 10,000 × 8 B × ~6 stages ≈ 0.5 MB — trivially resident, pre-touched at startup (§7 item 3). Percentiles are computed exactly post-run by sorting; raw samples are always exported (§11 requires machine-readable raw data anyway). HDR-style log₂ bucket rendering is print-side only.

**Output format (per run directory):**
- `meta.json` — chip, macOS build+SDK, git SHA, power source, thermal-state timeline (sampled 1 Hz on a *non*-critical thread), matrix-cell coordinates (`{experiment, level}` pairs), warmup/measured counts, timebase info, CPU/GPU timestamp-correlation pairs (before & after).
- `samples.csv` — one row per measured iteration; columns are stage timestamps/durations in ns (schema per milestone, below).
- `summary.json` — per stage: p50/p95/p99/p99.9/min/max/mean (mean reported, never a criterion), sample count, discarded-window count (thermal).

**Clock-domain alignment (spec §8, mandatory):** `sampleTimestamps` called immediately before and after each measured block; GPU timestamps mapped to CPU nanoseconds by linear interpolation between the two correlation pairs. Both raw pairs stored in `meta.json` so alignment can be re-derived offline.

**Assertion set (hard-fail, before any measured iteration):**
1. **ANE placement** (M1+): every op of the draft model reports `.neuralEngine` among `MLComputePlan` supported devices *and* preferred device; abort otherwise.
2. **Backing acceptance** (M2, E2): pointer/IOSurface identity between supplied backing and returned output — asserted on every warmup iteration and re-checked every measured iteration (cheap compare).
3. **Counter support**: `supportsCounterSampling(.atStageBoundary)`.
4. **Thermal gate**: runs start only in `.nominal`; iterations falling in non-nominal windows are flagged in `samples.csv` and excluded from headline histograms (kept in raw data).
5. **Canary** (M2, E6): GPU kernel writes a per-iteration validation flag to a results buffer; CPU asserts after each iteration.

**Protocol:** ≥500 warmup, ≥10,000 measured per cell (§8), plugged-in check recorded (warn, don't abort — but metadata marks it).

---

## 3. M0 — Baseline noise floor (full detail)

**Question:** what does *this process on this machine* pay for "another thread signalled me and I woke and read memory" — before any ANE involvement? Every M1+ number is read as a delta over this.

**Two sub-benchmarks × three thread policies (2×3 cells):**

- **`timer`** — pinned thread sleeps with `mach_wait_until(target)`, wakes, reads `mach_absolute_time()`. Sample = `actual − target` (pure scheduler wake jitter). Period 1 ms.
- **`xwake`** — models stage 4 of the critical path: a signaler thread stores `mach_absolute_time()` with release ordering, then `semaphore_signal` (raw mach semaphore — no libdispatch, per §4.2 rationale); the measured thread is parked in `semaphore_wait`, wakes, reads the clock. Sample = wake − signal. Signaler paces at 1 ms. This is the direct analog of "CoreML completion → handoff thread wakes".

Thread policies (E4 preview, reused in M1):
1. `default` — untouched pthread.
2. `qos-ui` — `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`.
3. `rt` — `THREAD_TIME_CONSTRAINT_POLICY`: period = 1 ms, computation = 200 µs, constraint = 500 µs, preemptible = true (in mach ticks). Conservative computation bound to avoid RT-demotion by the scheduler.

**csv schema:** `iter, sample_ns, thermal_ok`.
**Deliverable:** 6 histograms; the `xwake/rt` p99 is *the* noise floor constant for §2 arithmetic.

## 4. M1 — Empty-model round-trip harness (full detail)

**Model:** smallest CoreML ML Program that is ANE-eligible and non-trivial enough not to be constant-folded: a single fp16 conv (e.g. 1×64×32×32, 3×3, same-pad), built directly with the coremltools MIL builder (no torch). Compiled with `xcrun coremlcompiler`. Loaded with `computeUnits = .cpuAndNeuralEngine`; **placement asserted via MLComputePlan** before anything is timed. A second variant with a `[1, k, d]`-shaped output serves E8 transport measurement later (dummy weights are fine — spec E8).

**Timeline instrumentation (spec §5 stages → measurable points):**

| point | meaning | source |
|---|---|---|
| t0 | dispatch: `prediction()` called | `mach_absolute_time` on caller thread |
| t2′ | ANE completion *observed in userspace* (stages 2–3 inclusive; true t2 is invisible from public API — recorded as a known limitation, per-stage attribution of 2 vs 3 requires Instruments’ CoreML template, which the signposts enable) | sync: `prediction()` returns; async: first line of completion handler |
| t4 | handoff thread awake | `mach_absolute_time` after wake (sync arm: same as t2′) |
| t5 | after coherency point + `sharedEvent.signaledValue = n` | `mach_absolute_time` |
| t6 | first GPU work of the waiting command buffer | stage-boundary counter sample at pass start (GPU clock → CPU ns via correlation); cross-checked with `gpuStartTime` |

Headline metric **t2′→t6**; sub-segments t2′→t4 (scheduler), t4→t5 (ordering+signal), t5→t6 (event propagation + GPU release). `os_signpost` intervals mirror these for Instruments sessions.

**GPU side:** one `MTLCommandQueue`; a ring of **pre-encoded, pre-committed** command buffers (ring depth 16, refilled off-critical-path after each completion), each: `encodeWait(sharedEvent, value: n)` → compute pass (descriptor carries counter sample at stage start) → no-op kernel (reads one word from a dummy buffer so the wait can’t be elided; in M2 this becomes the real reader/canary kernel). Metal library compiled at startup from source string (`makeLibrary(source:)`) — no build-system Metal step, off the critical path.

**Experiment axes run at M1:**
- **E3** completion style: (a) sync `prediction()` on the pinned thread — thread parked in-kernel, handoff work inline; (b) async completion handler (which arrives on a CoreML-owned queue) doing only a semaphore-signal to the dedicated handoff thread. Note: the truly naive arm (do everything in the handler on a dispatch queue) is included as a third level for the “abstraction tax” comparison.
- **E4** thread policy: the M0 three, applied to the prediction/handoff thread.
- **E5** warmth: heartbeat prediction from a low-priority thread at swept period {none, 500 ms, 100 ms, 50 ms, 10 ms, saturated}; measured cell runs at a fixed 20 Hz dispatch rate so cold gaps come only from the sweep variable. Histograms segmented; bimodality reported explicitly (bucket render makes it visible).
- Re-derivation of the 0.095 ms dispatch figure: t0→t2′ distribution is exactly this number, measured (spec §1, §9 item 9).

**csv schema:** `iter, t0, t2p, t4, t5, t6_gpu, t6_cpu_ns, handoff_ns, thermal_ok, warm_state`.

**Kill criterion (preserved verbatim, §12 item 4):** if p99(t2′→t6) > ~2 ms across the best (E3×E4×E5) combination, the architecture is dead at M1; the deliverable is the histogram set and per-stage attribution, and M3 is *not* planned as assumed-reachable.

## 5. M2 — Real buffer shapes + backing A/B (full detail)

**Shapes:** draft-logits tensor `[1, k, V]` fp16, k ∈ {4, 8}, V = 32,768 → 256 KB / 512 KB. Also the k-token E8 variant. Models regenerated by `make_models.py` with real output shapes (weights still dummy — transport is what’s measured).

**E1 — GPU read path A/B:**
- **Option A:** `CVPixelBufferCreate` with IOSurface properties, `OneComponent16Half`, width = V, height = k (row-stride padding read from the surface, never assumed); wrap as `MLMultiArray(pixelBuffer:shape:)` for CoreML, import into Metal via `makeTexture(descriptor:iosurface:plane:0)` (`r16Float`, `textureType: .type2D`, usage `.shaderRead`). Kernel reads texture.
- **Option B:** `aligned_alloc(vm_page_size, round_page(bytes))`; wrap as `MLMultiArray(dataPointer:)` and `makeBuffer(bytesNoCopy:)` `.storageModeShared`. Kernel reads buffer.
- Both ping-pong A/B backings (§4.1); “which option avoids the hidden copy on M3 (chip)” is a headline result.

**E2 — backing acceptance assert:** Option B — output’s `withUnsafeMutableBytes` base address == supplied base address; Option A — output’s `.pixelBuffer` resolves to the same `IOSurfaceGetID`. Checked every iteration.

**E6 — coherency canary:** model output’s tail region is made input-dependent (identity-conv passthrough of an input tail the CPU rewrites with an incrementing pattern each iteration), so the expected tail value is known per iteration; the GPU kernel validates the tail *before* touching the rest and writes pass/fail + observed value to a results buffer. ≥10k iterations; any stale read is recorded with its iteration context. The coherency ordering point on the CPU side is `IOSurfaceLock/Unlock(readOnly)` around observation for Option A (and a fence-free release-store protocol note for Option B) — per §9 item 7, no Metal barriers pretend to order ANE DMA.

**E7 — memory pressure:** helper subprocess (spawned by the harness) allocating and touching memory in a ramp (target: sustained `memory_pressure`-visible pressure), on/off; × `mlock` attempt on Option B’s malloc’d backing region on/off (`mlock` on Option A’s IOSurface mapping attempted once, errno recorded — expected redundant/ineffective, §9 item 5). 2×2 cells, handoff histograms compared.

**E8 — dispatch granularity:** k sequential 1-token predictions vs one k-token prediction; total draft-side wall clock and per-cycle jitter-sample count compared.

**csv schema:** M1 schema + `cell(E1 option, E7 state, …), canary_ok, backing_ok`.

**Deliverable:** the zero-copy path that is actually zero-copy on this silicon, with proof (E2 asserts + E6 canary outcome + histograms).

## 6. M3 — Real models (outline)

- **Draft (ANE):** 0.3–1B fp16 decoder converted via coremltools with stateful KV cache (`MLState`, macOS 15+ — present in this SDK as `MLModel+MLState.h`, verified). Candidates: OpenELM-270M/450M, Llama-3.2-1B. ANE placement asserted per-op; ops that fall off the ANE are reported (attention variants sometimes do).
- **Target (GPU) — open choice, options with trade-offs (§12 item 5):**
  1. **mlx / mlx-swift** — Swift-native, easy logits access for verify-scoring of k positions; adds a third-party dependency to the *model* path (not the handoff path); MLX manages its own Metal queue, so the pre-committed-wait pattern needs adapting (likely: MLX evaluates on demand after the shared-event wait is observed by a thin custom kernel, or MLX’s stream is triggered from the handoff thread — to be prototyped).
  2. **llama.cpp (Metal)** — mature Q4 7–8B support and an existing speculative example to crib acceptance logic from; C API; integrating the shared-event handoff into its Metal backend requires patching its encode path.
  3. **Minimal custom Metal decode path** — full control over command-buffer pre-commit and counters (cleanest for measurement integrity), but a large engineering cost for a 7–8B transformer.
  - *Leaning:* (1) for accepted-tokens/s end-to-end numbers with (3)-style thin custom verify kernel for the measurement-critical boundary; decided after M1/M2 numbers exist.
- Acceptance/rejection + rollback wired per Leviathan et al.; rejection cost folded into expected-tokens/s model (§2), measured against a GPU-only baseline of the same target.

## 7. M4 — Writeup (outline)

Generated from `Results/` by `Tools/plot_results.py`: full histograms per matrix cell, per-stage attribution, E5 warmth curve, E6 canary verdict, E1/E2 backing findings, §2 go/no-go arithmetic with measured constants (noise floor, T_handoff p99, T_draft, T_verify, rejection margin). Negative results reported with the same rigor.

## 8. Build order & status

1. ✅ SDK verification (this document, §0)
2. ✅ HarnessCore + M0 — noise floor: xwake/rt p99 10.2 µs
3. ✅ M1 — kill criterion PASS (p99 handoff 855–914 µs, →130 µs GPU-warm);
   biggest surprise: GPU idle-ramp dominates, added `--gpu-warm` axis
4. ✅ M2 — both zero-copy paths real, E6 canary clean, E7 null, E8 decisive
5. ✅ M3.1 — MLX speculative loop, greedy-equivalence PASS; same-device
   control = 0.58× (the number that motivates heterogeneity)
6. ✅ M3.2 — 1B stateful Llama draft on the ANE (recipe: fixed-window
   attention + MLState + ≤1 GB palettized weights); heterogeneous pipeline
   proven correct; 0.15× at k=4/3B with two named levers (per-call ANE
   latency → multi-token head; 4-bit acceptance collapse → 6-bit LUT)
7. ⏳ 6-bit draft A/B; 8B target (spec-faithful §2 arithmetic); M3.3 overlap
   gated on the config becoming draft-fast; M4 writeup from FINDINGS.md

Measured detail lives in FINDINGS.md; per-run raw data in Results/.
