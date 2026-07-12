#!/usr/bin/env python3
"""Generate the harness CoreML models (build-time tool only; never on the
critical path — spec §11).

M1: the smallest ANE-eligible ML Program that is not constant-foldable — a
single fp16 conv. Built directly with the MIL builder so no torch dependency
is needed. Weights are random; only transport/dispatch is measured (spec E8
note: a dummy model with the correct shapes measures transport identically).

Usage:  python3 make_models.py [output-dir]
Then:   xcrun coremlcompiler compile Models/m1_tiny.mlpackage Models/
"""
import sys
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def m1_tiny(outdir: str) -> None:
    # A single small conv is ANE-*capable* but CoreML's cost model prefers the
    # CPU for it (observed on M3: preferred=cpu supported=[cpu,ane]). Stacked
    # convs tip the cost model to the ANE and give the dispatch a realistic
    # few-ms compute body, closer to one draft-token step.
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 128, 32, 32), dtype=types.fp16)],
        opset_version=ct.target.iOS17,
    )
    def prog(x):
        for i in range(4):
            w = ((np.random.rand(128, 128, 3, 3) - 0.5) * 0.05).astype(np.float16)
            x = mb.conv(x=x, weight=w, strides=[1, 1], pad_type="same",
                        name="logits" if i == 3 else f"conv_{i}")
        return x

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    path = f"{outdir}/m1_tiny.mlpackage"
    model.save(path)
    print(f"wrote {path}")


def m2_logits(outdir: str, k: int, H: int = 2048) -> None:
    """M2 transport model: one fp16 linear producing draft-logits shapes.

    V = 16384 (Metal's max texture width, so Option A can map the row-major
    logits [k, V] onto a single r16Float IOSurface texture of width V).
    The last C columns of the weight matrix are an identity block over the
    last C input features, so logits[row, V-C+j] == x[row, D+j] exactly in
    fp16 — the E6 coherency canary is computed by the same ANE matmul that
    produces the logits, not by a separate (possibly CPU-placed) op.
    """
    V, D, C = 16384, 512, 8
    rng = np.random.default_rng(7)

    def carrier(out_dim: int, in_dim: int, scale: float, name: str) -> np.ndarray:
        """Weight [out+C, in+C]-shaped (or [V, in+C] for the projection) whose
        bottom-right block is the CxC identity: the last C features pass
        through every layer unchanged (exact in fp16 — one nonzero term per
        canary row), so the E6 canary is computed by the same ANE ops as the
        logits. ReLU between layers is identity for the nonnegative canary."""
        W = np.zeros((out_dim + C, in_dim + C), dtype=np.float16)
        W[:out_dim, :in_dim] = ((rng.random((out_dim, in_dim)) - 0.5) * scale).astype(np.float16)
        W[out_dim:, in_dim:] = np.eye(C, dtype=np.float16)
        return W

    # A single [520 -> 16384] linear is ANE-capable but CPU-preferred by the
    # CoreML cost model (observed on M3, same as M1's single conv). Two hidden
    # layers push the program onto the ANE and look like a real draft-head MLP.
    W1 = carrier(H, D, 0.02, "w1")
    W2 = carrier(H, H, 0.002, "w2")
    Wp = carrier(V - C, H, 0.002, "wp")  # projection: out = (V-C)+C = V

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, k, D + C), dtype=types.fp16)],
        opset_version=ct.target.iOS17,
    )
    def prog(x):
        x = mb.relu(x=mb.linear(x=x, weight=W1))
        x = mb.relu(x=mb.linear(x=x, weight=W2))
        return mb.linear(x=x, weight=Wp, name="logits")

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    path = f"{outdir}/m2_logits_k{k}.mlpackage"
    model.save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "Models"
    m1_tiny(outdir)
    m2_logits(outdir, k=8)  # multi-token draft head (E8 'multi')
    # E8 'seq' finding on M3: at k=1 the same H=2048 program is CPU-preferred —
    # CoreML's cost model won't put per-token-sized dispatches on the ANE at
    # all. To measure seq-arm *transport* anyway, the k=1 model is widened so
    # one call carries a realistic draft-step weight load (~400 MB fp16).
    m2_logits(outdir, k=1, H=8192)
