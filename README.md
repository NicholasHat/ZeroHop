# ZeroHop

Measurement harness for **heterogeneous speculative decoding on Apple Silicon**:
a draft model on the Apple Neural Engine feeding a verify pass on the Metal GPU,
with the ANE→GPU handoff latency as the quantity under test.

- Spec: [`hsd-ane-gpu-spec.md`](hsd-ane-gpu-spec.md)
- Implementation plan + SDK verification: [`PLAN.md`](PLAN.md)

## Requirements

- Apple Silicon Mac, macOS 15+ (developed on M3 / macOS 26.5.1 / Xcode 26.4)
- Xcode command-line tools (`swift`, `xcrun coremlcompiler`)
- Python 3.12 for model generation only (`Tools/.venv`, coremltools)

## Setup

```sh
swift build -c release

# one-time: generate + compile the harness CoreML models
python3.12 -m venv Tools/.venv
Tools/.venv/bin/pip install coremltools numpy
Tools/.venv/bin/python Tools/make_models.py Models
xcrun coremlcompiler compile Models/m1_tiny.mlpackage Models/
```

## Running

Run plugged in, on a quiet machine, and **always use the release build for
measurements** — debug-build harness overhead was measured inflating the
dispatch segment ~3× (see FINDINGS.md). Full protocol is ≥500 warmup +
≥10,000 measured iterations per cell; results land in
`Results/<timestamp>-<cell>/` as `meta.json` + `samples.csv` + `summary.json`.
Headline findings so far: [`FINDINGS.md`](FINDINGS.md).

```sh
# M0 — baseline process-jitter noise floor (6 cells, ~2 min)
.build/release/zerohop m0

# M1 — ANE round-trip + handoff, one cell per invocation
# e3: sync | async | naive     (completion style)
# e4: default | qos-ui | rt    (thread policy)
# e5: none | 500 | 100 | 50 | 10 | saturated   (keep-warm heartbeat period, ms)
.build/release/zerohop m1 --model Models/m1_tiny.mlmodelc --e3 sync --e4 rt --e5 none

# full E3×E4 sweep example
for e3 in sync async naive; do
  for e4 in default qos-ui rt; do
    .build/release/zerohop m1 --model Models/m1_tiny.mlmodelc --e3 $e3 --e4 $e4 --e5 none
  done
done

# M2 — transport: E1 read path A|B, E8 multi|seq, E7 pressure/mlock,
# GPU keep-warm (none | <ms> | saturated)
.build/release/zerohop m2 --e1 B --e8 multi --gpu-warm saturated
.build/release/zerohop m2 --e1 A --e8 multi --gpu-warm saturated --pressure on --mlock on
```

## M3 — speculative decoding (separate binary)

`zerohop-m3` links mlx-swift for the target lane and therefore **must be
built with xcodebuild** (command-line SwiftPM cannot compile MLX's Metal
shaders; the binary dies with "Failed to load the default metallib"):

```sh
xcodebuild build -scheme zerohop-m3 -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .build/xcode

.build/xcode/Build/Products/Release/zerohop-m3 --k 4 --n 200 \
  --target mlx-community/Llama-3.2-3B-Instruct-4bit \
  --draft  mlx-community/Llama-3.2-1B-Instruct-4bit
```

Models download to `~/.cache/huggingface` on first run. The M0–M2
measurement binary (`zerohop`) intentionally has zero third-party
dependencies and still builds with plain `swift build`.

Optional: run `sudo powermetrics --samplers ane_power,gpu_power -i 1000` in a
second terminal during measured runs and note the window in the run's
`meta.json`; the harness itself records thermal state and power source.

Hard assertions before anything is measured: ANE placement via `MLComputePlan`
(the harness refuses to measure a model that fell back to CPU/GPU) and
`outputBackings` pointer identity (hidden-copy detection, experiment E2).
