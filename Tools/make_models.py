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


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "Models"
    m1_tiny(outdir)
