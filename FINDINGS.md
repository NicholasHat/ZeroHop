# Findings — ANE→GPU handoff characterization (M0–M2)

**Hardware/software:** Apple M3 · macOS 26.5.1 (25F80) · Xcode 26.4 / SDK 26.4.
All numbers below are from `Results/` runs on this machine, plugged in, thermal
state nominal unless flagged. Session runs use a reduced protocol (≥1,000
measured iterations per cell at a 20 Hz dispatch pace); re-run with
`--warmup 500 --iters 10000` for publication-grade histograms (commands in
README).

## Verdict so far (M1 kill criterion, spec §10)

**PASS.** p99 handoff (t2′→t6) stayed inside the ~2 ms budget in every
configuration measured — 855–914 µs without GPU keep-warm, ~420 µs with it,
against a process noise floor of ~10 µs. The architecture survives its first
gate; the dominant costs are power-state ramps, not scheduler jitter.

## M0 — process noise floor (10k iterations/cell)

| benchmark | policy | p50 | p99 | p99.9 |
|---|---|---|---|---|
| cross-thread wake (mach semaphore) | rt (`THREAD_TIME_CONSTRAINT_POLICY`) | 1.7 µs | 10.2 µs | 17.6 µs |
| cross-thread wake | qos-ui | 8.8 µs | 50.6 µs | 862 µs |
| cross-thread wake | default | 6.2 µs | 21.8 µs | 59.5 µs |
| timer wake (`mach_wait_until`) | rt | 9.6 µs | 27.4 µs | 42.4 µs |
| timer wake | default | 260 µs | 389 µs | 1.64 ms |

The RT time-constraint policy is worth ~50× at the p99.9 tail versus QoS-UI
(spec §7 ranking confirmed). Oddity worth keeping: **QoS-UI tails are worse
than default** on this build — recheck on other macOS versions.

## M1 — empty-model round trip (E3 × E4, 1500 iters/cell, no GPU warm)

- **Dispatch (t0→t2′)**: ~1.0 ms p50 / ~1.3–1.6 ms p99 for a 4-conv fp16
  ANE model. This re-derives the "0.095 ms" figure from the spec's claims
  registry: **the public CoreML round trip costs ~10× that**; the Orion number
  presumably measures a lower layer.
- **E3 (completion style)**: sync wins at p50 (637 µs vs 714 µs async vs
  711 µs naive) but all three converge at p99 (~855–890 µs) because **t5→t6
  dominates** (below). The libdispatch hop the spec feared (stage 4) measures
  only ~24 µs p50 / ~41 µs p99 (`t2→t4`, async arm) on a quiet machine.
- **E4 (thread policy)**: differences are within noise here for the same
  reason; policy matters exactly where M0 says it does (the wake edge), which
  is a small share of this handoff.

## GPU keep-warm — the dominant effect nobody asked about

The spec's warmth concern (E5) targets the ANE. The measured bottleneck was
the **GPU**: with one tiny verify kernel per 50 ms, t5→t6 (event signal →
kernel start, work pre-committed) is ~650 µs p50. A trickle of trivial
dispatches on a separate queue collapses it:

Batch A sweep (M1 model, sync/rt, 1000 iters/cell, release build):

| gpu-warm | t5→t6 p50 | p99 | p99.9 | max |
|---|---|---|---|---|
| none | 583 µs | 865 µs | **6.48 ms** | 9.76 ms |
| 10 ms trickle | 440 µs | 601 µs | 912 µs | 9.75 ms |
| 2 ms trickle | 418 µs | 529 µs | 673 µs | 3.93 ms |
| saturated | **62 µs** | **128 µs** | **186 µs** | **235 µs** |

Saturation improves the *whole* distribution by ~10× and, critically, kills
the multi-millisecond tail entirely — the p99.9 with no keep-warm (6.5 ms)
would have violated the kill criterion on its own if it appeared at p99.
Trickle periods only partially recover clocks. In real pipelined operation the
GPU is continuously busy verifying, so production gets saturation for free —
but any measurement (and any idle gap in the pipeline, e.g. after a rejection)
must account for this ramp.

## M2 — transport (E1/E2/E6/E7/E8)

- **E1/E2 — both zero-copy paths are real on M3.** Option A (IOSurface-backed
  CVPixelBuffer → `MLMultiArray(pixelBuffer:)` on the CoreML side,
  `makeTexture(descriptor:iosurface:plane:)` on the Metal side) and Option B
  (`aligned_alloc` → `MLMultiArray(dataPointer:)` + `makeBuffer(bytesNoCopy:)`)
  both pass the per-iteration identity assert — **no hidden copy on either
  path**, latencies statistically indistinguishable at these sizes (256 KB
  logits). V=16384 was chosen deliberately: it is Metal's maximum texture
  width, making Option A a single r16Float texture of width V.
- **E6 — no stale read ever observed.** The canary (an identity block inside
  the model's own weight matrices, so the ANE matmul itself computes the
  passthrough) validated on 100% of iterations across both read paths. The
  IOSurface + shared-event boundary appears coherent on M3 without extra
  ordering work. (Reportable either way per spec; this is the good outcome.)
- **E8 — dispatch granularity.** One 8-token call: ~3.3 ms. Eight 1-token
  calls: ~54 ms/cycle (confounded by the deliberately heavier k=1 model —
  see below — but the amortization direction is unambiguous). **Sharper
  finding:** at k=1 with the same architecture, `MLComputePlan` reports the
  whole program CPU-preferred — CoreML's cost model refuses per-token-sized
  ANE dispatches entirely at draft-head scale. Multi-token draft heads are
  not just an optimization; at these sizes they are the only way onto the ANE
  through public API.
- **CoreML cost-model gate (M1+M2):** a single small conv and a single
  520→16384 linear were both ANE-capable but CPU-preferred. The placement
  assert (spec §8) caught this on the first run in both milestones — silent
  fallback is real and would have invalidated everything.
- **E7 — mlock:** succeeds trivially on the 32 KB malloc'd backing (nothing
  interesting); IOSurface-mapping attempt recorded per run. Pressure results
  in batch below.

## E5 — ANE warmth sweep (batch B: sync/rt, GPU saturated, 1000 iters/cell)

| heartbeat | dispatch p50 | handoff p99 | handoff max |
|---|---|---|---|
| none | 299 µs | 167 µs | **1.50 ms** |
| 500 ms | 294 µs | 116 µs | 172 µs |
| 100 ms | 301 µs | 124 µs | 174 µs |
| 50 ms | 309 µs | 128 µs | 181 µs |
| 10 ms | 303 µs | 140 µs | 176 µs |
| saturated | **225 µs** | 132 µs | 184 µs |

- At a 20 Hz measured cadence, **any heartbeat ≥500 ms already suppresses the
  cold tail completely** (max 1.5 ms → ~175 µs); the sweep is flat below that.
  The ANE's power-gating timescale at this cadence is therefore coarser than
  500 ms — exposing deeper cold states needs a slower measured cadence (future
  cell: pace 1–10 s).
- A saturated heartbeat *lowers* dispatch p50 by ~25% (fully-clocked ANE
  outweighs contention at this model size).

## Methodology bug caught by the batches

Debug-build harness overhead inflated t0→t2′ from ~300 µs to ~1.0 ms (the M1
E3/E4 sweep above ran in debug). Handoff sub-segments were unaffected (they
bracket framework/GPU work, not harness code). **Release builds are mandatory
protocol from here on**; the corrected dispatch figure for the 4-conv model is
~300 µs p50 — still ~3× the claims-registry number, now measured rather than
assumed. The E3/E4 *comparisons* remain valid (same build per sweep), but
their absolute dispatch numbers supersede as above.

## Batch C — M2 transport cells (release, GPU saturated, 1000 iters/cell)

| cell | dispatch p50 | handoff p50 | handoff p99 | E2 | E6 canary |
|---|---|---|---|---|---|
| A (IOSurface→texture) | 1.44 ms | 73 µs | 145 µs | honored | clean |
| B (bytesNoCopy buffer) | 1.42 ms | 66 µs | 119 µs | honored | clean |
| B + 6 GiB pressure | 1.42 ms | 65 µs | 122 µs | honored | clean |
| B + pressure + mlock | 1.42 ms | 66 µs | 125 µs | honored | clean |
| A + 6 GiB pressure | 1.43 ms | 65 µs | 123 µs | honored | clean |
| B, E8 seq (8×k1 calls) | 50.8 ms/cycle | 71 µs | 126 µs | honored | clean |

- **E1 verdict:** both paths real; Option B is marginally better at the tail
  (p99 119 vs 145 µs) and simpler. Recommend B as default, A as fallback.
- **E6 verdict:** zero stale reads in ~4,800 measured iterations across every
  cell — IOSurface + shared-event boundary is coherent on M3 as exercised.
- **E7 verdict:** 6 GiB of continuously-touched pressure moved nothing
  (handoff p99 ±3 µs); mlock changed nothing on top (spec §9 item 5
  expectation met). *Caveat:* the balloon may not have driven this machine
  into real page-pressure; a swept-balloon cell (up to memory limit) is future
  work before calling E7 closed.
- **E8:** 8 sequential 1-token calls cost 50.8 ms/cycle vs 1.42 ms for one
  8-token call. Per-call cost of the (deliberately 400 MB-heavy) k1 model is
  ~6.4 ms — consistent with ANE weight-streaming being the per-dispatch cost
  floor. Combined with the placement finding, multi-token heads are settled.

## §2 arithmetic with measured constants (k=8 draft slot)

```
T_draft(k=8)  ≈ 1.4 ms  (3-layer 43 MFLOP/row head; real 0.3–1B draft will be larger)
T_handoff(p99)≈ 0.15 ms (GPU busy — the realistic pipelined state)
T_verify      ≈ 20–50 ms (7–8B Q4 target, spec estimate; measured at M3)
margin        : one wasted draft slot per rejection ≈ T_draft
```

1.4 + 0.15 ≪ 20 − 1.4: **the handoff is not the constraint on this chip; the
draft model's own latency budget is.** GO for M3.

## M3.1 — speculative decoding, both lanes on GPU (the control experiment)

Llama-3.2-1B-4bit draft → Llama-3.2-3B-4bit target, MLX, k=4, 200 tokens,
greedy:

- **Correctness: PASS** — speculative output is token-identical to the
  target-only baseline (greedy equivalence), so acceptance/rollback and both
  KV-cache trim paths are right.
- Acceptance 36.7%, 2.47 tokens per verify round.
- **Speedup 0.58× — same-device speculation *loses*.** Draft propose costs
  34.6 ms p50 (k=4 sequential 1B calls) and verify 57.8 ms p50, on the same
  GPU the baseline uses exclusively at 46.2 tok/s. The draft steals the
  verifier's device. This is the motivating measurement for the heterogeneous
  architecture: the ANE draft's job is to make T_draft disappear from the
  GPU's timeline.
- Also notable: verify(k=4) ≈ 2.7× a single decode step on MLX at 3B — the
  "verify ≈ one decode" memory-bound assumption (spec §2) does NOT hold at
  3B/4-bit on MLX; it needs re-measurement at 7–8B before the §2 arithmetic
  is finalized.

## M3.2 — getting a real 1B stateful Llama onto the ANE (the recipe)

Three successive walls, each isolated with a discriminating experiment:

1. **Dynamic shapes poison everything.** Tensor-valued slice bounds
   (`k[:, :, :pos+n]`) make the attention graph dynamic and `MLComputePlan`
   reports the *entire* program `supported=[cpu]` — not per-op fallback,
   wholesale rejection. Fix: fixed-window attention — always attend over the
   full 768-slot cache; the full-width additive causal mask excludes
   unwritten slots. (Bonus: rollback becomes O(1) — move the position
   pointer back and re-mask; no cache trim at all.)
2. **MLState is NOT the problem.** A toy stateful model reports
   `supported=[cpu,ane]` — states are ANE-eligible.
3. **Size is.** Full 1B at fp16 (2.3 GB): wholesale CPU. Same architecture
   truncated to 2 layers (~0.7 GB): transformer ops `preferred=ane`. 4-bit
   kmeans palettization of the full model (591 MB compiled): **all
   transformer compute `preferred=ane`**, with only the 32 state
   read/write ops on CPU. This is why ANEMLL ships LUT-quantized chunks.

Also required: coremltools 9.0 `_cast` workaround (rejects 1-element arrays;
Llama-3.2's RoPE scaling factor `[32.]` hits it), torch ≤2.7,
transformers ≤4.x (v5 rewrote the Cache API).

**Net: a 1B stateful KV-cache Llama draft runs on the ANE through public
API.** Heterogeneous benchmark results below.

## M3.2 — first heterogeneous benchmark (ANE draft + GPU target)

Same setup as M3.1 but the draft on the ANE (k=4, 200 tokens, serial loop —
no overlap yet):

- **Correctness: PASS** — greedy equivalence holds with the ANE draft, so
  the whole heterogeneous path (CoreML MLState draft, O(1) mask rollback,
  cross-framework token flow) is sound.
- **Speed: 6.9 tok/s vs 45.9 baseline (0.15×).** Attribution:
  1. **Per-call ANE latency: ~34 ms/token** (draft_propose 136.6 ms p50 at
     k=4). Consistent with public-API ANE 1B decode rates (~30 tok/s class);
     the GPU/MLX 1B draft does the same call in ~8.6 ms. The E8 lesson says
     the fix is a multi-token draft head — k drafts in ONE ANE dispatch
     amortizes this to ~1/k.
  2. **Acceptance collapsed to 11.2%** (vs 36.7% for the MLX 4-bit draft) —
     4-bit kmeans palettization degrades the draft's agreement with the
     target far more than MLX's grouped 4-bit. 6-bit LUT or
     grouped-quant-aware settings are the lever; draft quality only affects
     acceptance, never correctness.
- Pipelining (M3.3) cannot rescue this configuration: T_draft(k=4) ≈ 137 ms
  > T_verify ≈ 67 ms — the pipeline would be draft-bound. The configuration
  must first become draft-fast (multi-token head, smaller/better-quantized
  draft) before overlap pays.

**Standing result:** the architecture is *mechanically* proven end to end on
public API — first known instance of ANE-draft speculative decoding — and
the measured gap decomposes into two named, addressable levers (ANE dispatch
amortization via E8-style multi-token heads; palettization quality).

## Open items

- E5 warmth curve interpretation (bimodality per period).
- Full 10k-iteration protocol runs for all headline cells.
- Instruments session (CoreML template + Metal System Trace) to attribute the
  ~1 ms dispatch across stages 2–3 (aned/XPC vs firmware).
- M3 (real models) gated on nothing now — kill criterion passed; target-side
  runtime choice per PLAN §6.
