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
  ANE model (debug build; ~300 µs p50 in release — see the methodology-bug
  note below). This re-derives the "~0.095 ms" dispatch figure from the
  spec's claims registry: **the public CoreML round trip costs ~3× that
  (release build)**. Attribution correction: that figure originates from
  maderix's ANE reverse-engineering benchmarks (XPC + IOKit dispatch
  overhead via the private `_ANEClient` path), which Orion credits — it is
  not an Orion measurement, and it measures the *private* dispatch layer.
  The ~200 µs gap between it and our ~300 µs is a first estimate of the
  public-API (CoreML) tax on the dispatch path — exactly the "abstraction
  tax" number spec §7 item 5 asked for.
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
public API — as of July 2026 we found no published characterization of
two-model ANE-draft/GPU-verify speculation through public CoreML (see the
landscape section at the end) — and the measured gap decomposes into two
named, addressable levers (ANE dispatch amortization via E8-style
multi-token heads; palettization quality).

## M3.2 follow-up — the acceptance collapse was a draft bug, not quantization

A 6-bit rerun left acceptance unchanged (11.5% vs 11.2%) — falsifying the
quantization hypothesis. A torch-vs-CoreML greedy A/B (the missing
draft-health assertion; now part of the conversion protocol) showed the
CoreML model degenerating into token repetition: **`torch.jit.trace` bakes
Python slice bounds to their trace-time constants**, so the KV write
`k[..., begin:end, :] = …` silently stored every token in slot 0. The
benchmark's greedy-equivalence check could never catch this — the target
corrects every wrong draft, so output stays perfect while acceptance craters.

Fixing the write took threading three constraints at once (each hit
empirically): slice bounds bake under trace; `index_put_` advanced indexing
trips a coremltools frontend dtype clash (fp16 update vs fp32-upcast state
read); fp32 states are rejected by the backend (states must be fp16). The
formulation that survives all three is a **one-hot blend**:
`cache = cache·(1−w) + onehot(pos)ᵀ·kv` — pure fp16 elementwise + matmul,
ANE-native, no dynamic indexing.

**Fixed draft vs 3B target: acceptance 49.6%** (better than the MLX 4-bit
draft's 36.7% — the fp16 ANE draft tracks the target more faithfully),
2.99 tokens/round, 20.6 tok/s vs 46.2 baseline (0.45×), draft propose
75.4 ms p50 (~19 ms/token on the ANE). Against the 3B target the GPU
baseline is simply too fast for a 1B serial draft; the spec-faithful 8B
matrix follows.

## 8B target matrix (spec §2 configuration) — the decisive table

Meta-Llama-3.1-8B-Instruct-4bit target, k=4, 204 tokens, greedy, serial loop
(no overlap). Both cells: greedy equivalence PASS, acceptance **85.9%**,
4.43 tokens/verify-round — a 1B Llama is an outstanding speculator for the
8B at temperature 0.

| cell | draft propose p50 | verify p50 | tok/s | vs baseline 19.1 |
|---|---|---|---|---|
| GPU-only baseline | — | — | 19.1 | 1.00× |
| MLX draft (same GPU) | 42.7 ms | 135.5 ms | 24.8 | **1.28×** |
| ANE draft (serial) | 95.6 ms | 144.4 ms | 18.2 | 0.95× |

### Final §2 arithmetic, all constants measured

```
T_draft(k=4)   =  95.6 ms   (1B fp16/6-bit-LUT on ANE, 4 sequential calls)
T_handoff(p99) ≈   0.15 ms  (M1/M2, GPU kept busy — negligible)
T_verify       = 144.4 ms   (8B-4bit scoring k+1 positions on Metal)

95.6 + 0.15  <  144.4  ✓  — the spec §2 viability inequality HOLDS.
```

The draft fits *inside* the verify window with 34% margin. That is the
architectural go signal: with the M3.3 overlap (draft batch N+1 on the ANE
while the GPU verifies batch N), the round time collapses to ~T_verify:

- projected pipelined heterogeneous: 4.43 tokens / 144.4 ms ≈ **30.7 tok/s
  ≈ 1.6× baseline** — and unlike the MLX draft, the ANE draft consumes zero
  GPU time, so the projection doesn't cannibalize verify. The GPU-draft
  variant cannot pipeline this way at all (draft and verify serialize on the
  same device); its measured 24.8 tok/s is close to its ceiling.
- ANE per-call latency remains the top optimization target (~24 ms/token at
  8B-run conditions vs 42.7 ms/k=4 for MLX): an E8-style multi-token head
  would cut propose toward one dispatch (~30–40 ms), pushing the pipelined
  budget toward k=6–8.

**Verdict: GO.** Serial break-even today; the measured constants satisfy the
spec's §2 inequality with margin, and the overlap implementation (M3.3) is
projected to beat both the baseline (1.6×) and same-device speculation
(1.24× relative) — with the draft entirely off the GPU.

## M3.3 — pipelined overlap: the thesis measurement

Implementation: the draft lane runs on its own real-time pthread (E3/E4
winners), speculating a k+1 self-continuation of batch N+1 while the GPU
verifies batch N; a hit (all k accepted AND the draft's first speculative
token equals the target's bonus) makes the remaining k tokens the aligned
next batch; a miss pays the wasted-draft-slot resync (O(1) mask rollback of
2k−j positions + one serial re-propose) — precisely the rejection cost §2's
margin budgets.

Measured (8B target, k=4, 204 tokens, greedy):

| configuration | tok/s | vs its baseline |
|---|---|---|
| GPU-only baseline | 20.6 | 1.00× |
| MLX draft, same GPU, serial | 24.8 | 1.28× |
| ANE draft, serial | 18.2 | 0.95× |
| **ANE draft, pipelined (M3.3)** | **26.4** | **1.28×** |

- **Pipeline hit rate 80.4%** (37/46 speculative batches landed).
- **The draft is fully hidden**: overlap-span p50 150.465 ms vs verify p50
  150.463 ms — the k+1 ANE propose fits inside the verify window with
  nothing left over. On hit rounds the draft costs zero wall-clock; round
  p50 equals verify p50. Misses show up as the round p95 (235 ms).
- **Greedy equivalence PASS** — pipelined rollback bookkeeping is exact.
- The heterogeneous pipeline matches same-device speculation's ratio (1.28×)
  and beats its absolute throughput (26.4 vs 24.8 tok/s) while consuming
  **zero GPU time for drafting** — the GPU does nothing but verify. Any
  further ANE draft speedup (multi-token head) now converts directly into
  headroom for larger k rather than fighting the verify for the device.

**Final verdict: the architecture works, measured end to end on public
API.** ANE-drafted, GPU-verified speculative decoding on one Apple Silicon
chip is real, correct (greedy-exact), and the fastest configuration tested.

## Optimization round — draft cost → speculation window

Two changes to the draft export: 4-bit kmeans palettization (591 MB — the
acceptance collapse previously blamed on 4-bit was the cache bug; post-fix
its torch A/B matches for 3 tokens then diverges coherently) and a **fused
greedy head** (the model returns the argmax token id — 4 bytes/call instead
of a 256 KB logits tensor, reduction on-device). Then a k-sweep, all
pipelined, 8B target (ratios are the robust metric; these ran on battery):

| draft, k | tok/s | speedup | tokens/round | hit rate | verify p50 |
|---|---|---|---|---|---|
| 6-bit, k=4 | 26.4 | 1.28× | 4.43 | 80.4% | 150 ms |
| 4-bit fused, k=4 | 25.4 | 1.33× | 4.34 | 78.7% | 147 ms |
| 6-bit, k=6 | 26.6 | 1.39× | 5.83 | 74.3% | 186 ms |
| **4-bit fused, k=6** | **28.1** | **1.47×** | 5.69 | 72.2% | 176 ms |
| 4-bit fused, k=8 | 26.3 | 1.38× | 6.87 | 66.7% | 216 ms |

- In every cell the overlap span equals verify p50 to the microsecond — the
  draft (up to 9 sequential ANE calls at k=8) stays fully hidden. The
  faster draft's value is exactly what §2's arithmetic says: a wider viable
  speculation window, not lower round latency.
- **The knee is k=6 at 1.47×**: past it, per-token acceptance decay (78% →
  73%) and verify growth (176 → 216 ms) outpace the extra tokens/round.
- Greedy equivalence PASS in all cells.
- A trained multi-token (Medusa-class) head is the one lever left untouched
  — it needs training, out of PoC scope; everything reachable with public
  models and public API is now measured.

## Open items

- E5 warmth curve interpretation (bimodality per period).
- Full 10k-iteration protocol runs for all headline cells.
- Instruments session (CoreML template + Metal System Trace) to attribute the
  ~1 ms dispatch across stages 2–3 (aned/XPC vs firmware).
- M3 (real models) gated on nothing now — kill criterion passed; target-side
  runtime choice per PLAN §6.

## Landscape as of July 2026 (related work; framing for the report)

A targeted search (2026-07-13) found no published characterization of
public-API, two-model (separate draft + target) ANE↔GPU speculative decoding
with handoff measurement on consumer Apple Silicon. Three close neighbors,
all of which belong in the report:

- **`ane.cpp`** — experimental "speculative decode" flag for Qwen3, but
  *self*-speculative (truncated layers of the same model as the draft),
  ANE-only, via the reverse-engineered private `_ANEClient` API; the
  project's own notes report it currently slower than plain decoding. A
  useful negative result: naive self-speculation on the ANE doesn't
  trivially pay. Different architecture on all three axes (single model,
  single accelerator, private API).
- **`maderix/ANE`** — the reverse-engineering effort Orion credits as
  foundational. Includes demo scripts for GPU↔ANE zero-copy IOSurface
  transport and a GPU-prefill→ANE-decode pipeline — the handoff *concept*
  shown, though as demos on the private API, without rigorous latency
  characterization. **Attribution correction propagated through this
  document:** the ~0.095 ms dispatch figure originates from maderix's
  benchmarking (XPC + IOKit, private path), credited by Orion — not from
  Orion's own measurements.
- **`CoreML-LLM`** — public-API, ANE-only LLM inference (no GPU pairing,
  no speculation): evidence the sanctioned path is viable for inference
  per se.

**The honest framing this dictates:** every serious ANE performance effort
found (maderix, Orion, ane.cpp, ANE training work) abandoned the public
CoreML API for reverse-engineered private access — precisely the escape
hatch spec §7 item 5 gated off by default. The gap this project fills is
therefore probably not overlooked white space; it is the path the
knowledgeable avoided *because of* its overhead. That makes the
contribution "the measured cost and viability of the sanctioned path" —
directly useful to anyone who must ship through the App Store — rather than
a novelty claim. The public-vs-private dispatch delta (~300 µs release-build
CoreML round trip vs ~95 µs private-path dispatch) is the first number of
that comparison; this is exactly the "abstraction tax" quantification the
spec called publishable. The space is moving quickly (the private-API
cluster gained visibility within months of this work); any claims here are
time-stamped 2026-07 accordingly.

## Full-protocol (10k-iteration) reruns — 2026-07-14

Twelve headline cells rerun at the spec's full protocol (500 warmup +
10,000 measured, release build, GPU-warm as noted). Values below supersede
the 1k tables above where they differ; per-cell CSVs in Results/.

**GPU idle-ramp (M1, sync/rt, e5=none):**

| gpu-warm | t5→t6 p50 | p99 | p99.9 | max |
|---|---|---|---|---|
| none | 552 µs | 1.33 ms | 6.29 ms | 22.9 ms |
| 10 ms trickle | 464 µs | 1.11 ms | 9.32 ms | 21.2 ms |
| 2 ms trickle | 421 µs | 1.06 ms | 4.21 ms | 17.1 ms |
| saturated | 72 µs | 165 µs | 657 µs | 8.95 ms |

Two refinements over the 1k picture: (a) **trickle keep-warm does not
remove the deep tail** — every trickle period still shows multi-ms p99.9;
(b) even saturation only *thins* the extreme tail (~1-in-10k events still
reach ~9 ms) — events at that rarity are more consistent with OS
preemption/interrupt noise than GPU power state; per-event attribution via
Instruments is future work.

**E5 ANE warmth (M1, sync/rt, gpu-warm saturated):**

| heartbeat | dispatch p50 | handoff p99 | handoff max |
|---|---|---|---|
| none | 292 µs | 117 µs | 327 µs |
| 500 ms | 293 µs | 127 µs | 189 µs |
| 100 ms | 301 µs | 126 µs | 438 µs |
| 50 ms | 307 µs | 126 µs | 214 µs |
| 10 ms | 303 µs | 126 µs | 193 µs |
| saturated | 228 µs | 147 µs | 259 µs |

**E5 conclusion revised:** the 1.5 ms "cold event" in the 1k none-cell did
not reproduce at 10k — at a 20 Hz dispatch duty cycle the measured loop's
own traffic is heartbeat enough; the entire heartbeat axis is flat, and
explicit keep-warm buys nothing except saturation's ~22% dispatch-p50
reduction. Genuinely cold ANE states require idle gaps this cadence never
produces (slower-cadence sweep: future work).

**M2 transport (gpu-warm saturated):**

| read path | handoff p50 | p99 | p99.9 | max | E2 | E6 |
|---|---|---|---|---|---|---|
| A (IOSurface→texture) | 65 µs | 118 µs | 171 µs | 192 µs | honored ×10k | clean ×10k |
| B (bytesNoCopy) | 66 µs | 129 µs | 173 µs | 230 µs | honored ×10k | clean ×10k |

**E1 verdict revised:** the 1k-based "B marginally better" does not hold at
10k — A and B are statistically indistinguishable; choose by GPU-side
ergonomics (buffer vs texture reads). **E6 cumulative: zero stale reads in
26,000+ measured iterations.**
