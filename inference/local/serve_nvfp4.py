"""Serve the local NVFP4 Ripple-65B checkpoint with the DFlash2 draft on 2x RTX PRO 6000.

Adapted from the checkpoint's shipped ``inference/local/spark.py`` (a DGX Spark offline
engine, hard-pinned to sm_121 TP1). Differences:

- HTTP server via ``launch_server`` instead of the offline ``Engine``.
- TP2 across two sm_120 GPUs (RTX PRO 6000); radix cache kept (spark.py disables it for a
  single offline stream).
- Same NVFP4 flag set as spark.py: flashinfer attention, flashinfer-cutlass FP4 GEMMs,
  FP8-GEMM autotune skipped, DFLASH speculative decoding with the BF16 draft.
- BF16 KV cache by default (vendor parity). The quantized export carries FP8 k/v scales;
  spark.py neutralizes them for BF16 KV, and this entrypoint does the same unless
  RECIPE_KV_CACHE_DTYPE=fp8_e4m3 (the long-context profile, which keeps the real scales).
- spark.py's sm_121-only PDL softmax fix is omitted (sm_120 does not take that path).

Model and parsers are registered exactly as the shipped launcher does. The recipe's
start.sh appends settings from .env after these defaults; argparse keeps the last occurrence.
"""

import functools
import os
import shlex
import sys
from pathlib import Path

sys.dont_write_bytecode = True

# Extend the draft's positional cache beyond its 128Ki checkpoint limit when the recipe asks
# for a longer target context; the draft attention window itself stays fixed (upstream serve.py
# sets the same default).
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

# Top-level registration also runs in SGLang's spawned worker processes.
from sglang.srt.models.registry import ModelRegistry

from model import RicursiveForCausalLM
import ripple_format

ModelRegistry.models["RicursiveForCausalLM"] = RicursiveForCausalLM
ripple_format.register()


def _neutralize_kv_scales():
    """Quantized exports carry FP8 k/v scales although the KV cache is BF16 (spark.py's fix)."""
    if os.environ.get("RECIPE_KV_CACHE_DTYPE", "bfloat16") != "bfloat16":
        return
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

    load_model = ModelRunner.load_model

    @functools.wraps(load_model)
    def load(self):
        result = load_model(self)
        for name, module in self.model.named_modules():
            if isinstance(module, RadixAttention):
                for kind in ("k", "v"):
                    scale = getattr(module, kind + "_scale", None)
                    if scale is not None:
                        scale.data.fill_(1.0)
                        setattr(module, kind + "_scale_float", 1.0)
        return result

    ModelRunner.load_model = load


_neutralize_kv_scales()

DEFAULT_ARGS = shlex.split("""
    --model-path /model --host 0.0.0.0 --port 30001 --served-model-name ripple-65b-nvfp4
    --trust-remote-code
    --dtype bfloat16 --kv-cache-dtype bfloat16 --tp-size 2
    --attention-backend flashinfer --fp4-gemm-backend flashinfer_cutlass
    --flashinfer-autotune-skip-ops fp8_gemm --page-size 64
    --speculative-algorithm DFLASH --speculative-draft-model-path /dflash2
    --speculative-draft-model-quantization unquant
    --speculative-num-draft-tokens 32 --speculative-draft-window-size 16384
    --speculative-draft-attention-backend triton
    --context-length 131072 --mem-fraction-static 0.78 --max-running-requests 4
    --cuda-graph-backend-prefill disabled
    --tool-call-parser ripple --reasoning-parser ripple
""")
DEFAULT_ARGS += [
    "--chat-template",
    "/model/chat_template.jinja",
]


def main():
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    args = prepare_server_args(DEFAULT_ARGS + sys.argv[1:])
    if args.dtype != "bfloat16":
        raise ValueError("Only BF16 compute dtype is supported")
    try:
        launch_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
