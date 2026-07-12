#!/usr/bin/env python3
"""Convert Llama-3.2-1B to a stateful (KV-cache) fp16 CoreML model for the
ANE draft lane (M3.2).

Pattern: the KV cache lives in CoreML MLState buffers (registered as torch
buffers on a wrapper module and declared via ct.StateType); each forward
slice-updates the cache in place. Two enumerated input lengths keep shapes
static for the ANE: 128 (prompt prefill chunk) and 1 (decode step).

Weights come from an ungated mirror of meta-llama/Llama-3.2-1B-Instruct so no
HF token is required; the tokenizer/vocab matches the mlx-community 4-bit
variants used for the M3 target.

Usage: Tools/.venv/bin/python Tools/convert_draft.py [output-dir]
Then:  xcrun coremlcompiler compile Models/draft_llama32_1b.mlpackage Models/
"""

import sys

import numpy as np
import torch
import coremltools as ct
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
CONTEXT = 768
PREFILL = 128

# coremltools 9.0 bug workaround: the torchscript `int`/`float` cast handler
# calls dtype(x.val) which rejects 1-element arrays (`[32.]`, Llama-3.2's RoPE
# scaling factor, reaches it as shape-(1,)). torch semantics allow int() on
# any single-element tensor, so squeeze before delegating.
from coremltools.converters.mil.frontend.torch import ops as _ct_torch_ops
from coremltools.converters.mil import Builder as _mb

_orig_cast = _ct_torch_ops._cast

def _patched_cast(context, node, dtype, dtype_name):
    x = context[node.inputs[0]]
    val = getattr(x, "val", None)
    if val is not None and getattr(val, "size", 0) == 1 and getattr(val, "ndim", 0) > 0:
        context.add(_mb.const(val=dtype(val.reshape(())), name=node.name))
        return
    _orig_cast(context, node, dtype, dtype_name)

_ct_torch_ops._cast = _patched_cast


class SliceUpdateCache(Cache):
    """Cache implementation that writes K/V into externally-owned buffers via
    in-place slice assignment (traceable; becomes CoreML state updates)."""

    def __init__(self, key_cache: torch.Tensor, value_cache: torch.Tensor):
        super().__init__()
        self.k = key_cache
        self.v = value_cache

    # Only reached if the model needs a length and cache_position was not
    # provided — we always provide it, and the mask is passed pre-built as a
    # 4D tensor, so a static answer keeps everything traceable.
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0

    def get_max_cache_shape(self) -> int:
        return self.k.shape[3]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        pos = cache_kwargs["cache_position"] if cache_kwargs else None
        begin, end = pos[0], pos[0] + key_states.shape[2]
        self.k[layer_idx, :, :, begin:end, :] = key_states
        self.v[layer_idx, :, :, begin:end, :] = value_states
        return (
            self.k[layer_idx, :, :, :end, :],
            self.v[layer_idx, :, :, :end, :],
        )


class StatefulDraft(torch.nn.Module):
    def __init__(self, model_id: str, context: int):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        cfg = self.model.config
        cache_shape = (
            cfg.num_hidden_layers, 1, cfg.num_key_value_heads, context,
            cfg.hidden_size // cfg.num_attention_heads,
        )
        self.register_buffer("keyCache", torch.zeros(cache_shape, dtype=torch.float16))
        self.register_buffer("valueCache", torch.zeros(cache_shape, dtype=torch.float16))

    def forward(self, inputIds, causalMask, firstPosition):
        seq_len = inputIds.shape[1]
        # Stay tensor-valued end to end: any int()/item() here becomes an
        # untraceable aten scalar cast (breaks coremltools conversion).
        cache_position = firstPosition[0] + torch.arange(seq_len, dtype=torch.long)
        past = SliceUpdateCache(self.keyCache, self.valueCache)
        out = self.model(
            input_ids=inputIds,
            attention_mask=causalMask,
            cache_position=cache_position,
            past_key_values=past,
            use_cache=True,
        )
        return out.logits


def convert(outdir: str, seq_len: int = 1) -> None:
    """Fully static shapes: [1, seq_len] tokens against the fixed CONTEXT
    window. seq_len=1 is the decode-step model (prefill feeds the prompt one
    token at a time — a one-time cost per generation). Static shapes are also
    the ANE-friendly choice; EnumeratedShapes tripped a coremltools 9.0
    symbolic-shape bug (aten int cast on a 1-D value) and dynamic shapes tend
    to fall off the ANE anyway."""
    torch.set_grad_enabled(False)
    wrapper = StatefulDraft(MODEL_ID, CONTEXT).eval()

    ids = torch.zeros((1, seq_len), dtype=torch.long)
    mask = torch.zeros((1, 1, seq_len, CONTEXT), dtype=torch.float16)
    first = torch.zeros((1,), dtype=torch.long)
    traced = torch.jit.trace(wrapper, (ids, mask, first))

    cache_shape = tuple(wrapper.keyCache.shape)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputIds", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="causalMask", shape=(1, 1, seq_len, CONTEXT), dtype=np.float16),
            ct.TensorType(name="firstPosition", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        states=[
            ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
                         name="keyCache"),
            ct.StateType(wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
                         name="valueCache"),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    suffix = "" if seq_len == 1 else f"_s{seq_len}"
    path = f"{outdir}/draft_llama32_1b{suffix}.mlpackage"
    mlmodel.save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else "Models")
