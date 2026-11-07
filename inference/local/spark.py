"""Offline Ripple inference on DGX Spark: NVFP4 MLPs, BF16 attention + BF16 DFlash2."""

import fnmatch
import functools
import json
import os
import shlex
from pathlib import Path

# Keep the Spark kernel choices used by the original notebook.
os.environ["SGLANG_FLASHINFER_USE_PAGED"] = "1"
os.environ["FLASHINFER_USE_CUDA_NORM"] = "1"
os.environ.setdefault("MAX_JOBS", "2")
os.environ.setdefault("NVCC_THREADS", "1")

import torch
from huggingface_hub import snapshot_download
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import prepare_server_args

from model import RicursiveForCausalLM

import ripple_format

ModelRegistry.models["RicursiveForCausalLM"] = RicursiveForCausalLM
ripple_format.register()


def checkpoint(value):
    """Accept a complete local checkpoint or download a Hugging Face repo."""
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    if str(value).startswith(("/", "~", ".", "gs://")):
        raise FileNotFoundError(value)
    return Path(snapshot_download(str(value)))


def check_checkpoints(target, draft):
    """Require NVFP4 MLPs, BF16 attention, and an unquantized DFlash2 draft."""
    config = json.loads((target / "config.json").read_text())
    if config.get("architectures") != ["RicursiveForCausalLM"]:
        raise ValueError("Target must be a RicursiveForCausalLM checkpoint")
    quant = config.get("quantization_config") or {}
    if (target / "hf_quant_config.json").exists():
        quant = json.loads((target / "hf_quant_config.json").read_text())["quantization"]
    # ModelOpt stores the same recipe as config_groups/ignore in config.json
    # and as group_size/exclude_modules in the optional legacy sidecar.
    group_size = quant.get("group_size")
    if "config_groups" in quant:
        groups = list(quant["config_groups"].values())
        if len(groups) != 1 or groups[0].get("targets") != ["Linear"]:
            raise ValueError("Expected one NVFP4 group targeting Linear layers")
        for kind in ("weights", "input_activations"):
            scheme = groups[0].get(kind, {})
            if any(scheme.get(k) != v for k, v in
                   {"type": "float", "num_bits": 4, "group_size": 16, "dynamic": False}.items()):
                raise ValueError(f"Expected static NVFP4 {kind}, group size 16")
        group_size = 16
    if quant.get("quant_algo") != "NVFP4" or group_size != 16:
        raise ValueError("Spark requires an NVFP4/BF16 ModelOpt checkpoint (MLP group size 16)")
    if quant.get("kv_cache_quant_algo") is not None or quant.get("kv_cache_scheme") is not None:
        raise ValueError("Spark requires an unquantized BF16 KV cache")
    if quant.get("quantized_layers"):
        raise ValueError("Per-layer mixed-precision overrides are not supported")
    excluded = quant.get("exclude_modules", quant.get("ignore", []))

    def require(name, bf16):
        is_excluded = any(fnmatch.fnmatchcase(name, pattern) for pattern in excluded)
        if is_excluded != bf16:
            raise ValueError(f"Expected {'BF16' if bf16 else 'NVFP4'} for {name}")

    for name in ("lm_head", "model.embed_tokens"):
        require(name, True)
    for index in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{index}"
        for name in ("gate", "up", "gate_up", "down"):
            require(f"{prefix}.mlp.{name}_proj", False)
        heads = ("q", "k", "v", "o", "g") if config.get("use_gated_attention", True) else ("q", "k", "v", "o")
        for name in heads:
            require(f"{prefix}.self_attn.{name}_proj", True)
    draft_config = json.loads((draft / "config.json").read_text())
    if draft_config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("Draft must be a DFlash2DraftModel checkpoint")
    if draft_config.get("quantization_config") or (draft / "hf_quant_config.json").exists():
        raise ValueError("The DFlash2 draft must be unquantized BF16")


def run_worker(*args, **kwargs):
    """Retain the two numerical fixes required by the Spark attention path."""
    from sglang.kernels.ops.speculative.dspark import dspark_accept
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.scheduler import run_scheduler_process
    from sglang.srt.model_executor.model_runner import ModelRunner

    softmax = dspark_accept._flashinfer_softmax

    @functools.wraps(softmax)
    def safe_softmax(*args, **kwargs):
        kwargs["enable_pdl"] = False  # SM121 PDL softmax can produce NaNs.
        return softmax(*args, **kwargs)

    dspark_accept._flashinfer_softmax = safe_softmax
    load_model = ModelRunner.load_model

    @functools.wraps(load_model)
    def load(self):
        result = load_model(self)
        for name, module in self.model.named_modules():
            if isinstance(module, RadixAttention):
                # Quantized exports can carry FP8 scales although our KV is BF16.
                for kind in ("k", "v"):
                    scale = getattr(module, kind + "_scale", None)
                    if scale is not None:
                        scale.data.fill_(1.0)
                        setattr(module, kind + "_scale_float", 1.0)
            weight = getattr(module, "weight", None)
            if self.is_draft_worker and weight is not None and weight.dtype != torch.bfloat16:
                raise ValueError(f"Draft weight {name} must be BF16")
        return result

    ModelRunner.load_model = load
    return run_scheduler_process(*args, **kwargs)


class Spark(Engine):
    """One local inference engine; no HTTP server or port configuration."""

    run_scheduler_process_func = staticmethod(run_worker)

    def __init__(self, target, draft, *, speculative=True):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError("Run this notebook on a DGX Spark / GB10 GPU")
        target, draft = checkpoint(target), checkpoint(draft)
        check_checkpoints(target, draft)
        spec_args = [
            "--speculative-draft-model-path", str(draft),
            "--speculative-algorithm", "DFLASH", "--speculative-draft-model-quantization", "unquant",
            "--speculative-num-draft-tokens", "32", "--speculative-draft-window-size", "16384",
            "--speculative-draft-attention-backend", "triton",
        ] if speculative else []
        args = prepare_server_args(
            [
                "--model-path",
                str(target),
                "--chat-template",
                str(Path(__file__).resolve().parents[2] / "chat_template.jinja"),
            ]
            + spec_args + shlex.split("""
                --trust-remote-code --dtype bfloat16 --kv-cache-dtype bfloat16 --tp-size 1
                --attention-backend flashinfer --fp4-gemm-backend flashinfer_cutlass
                --flashinfer-autotune-skip-ops fp8_gemm
                --context-length 131072 --max-total-tokens 139264 --mem-fraction-static 0.78
                --max-running-requests 1 --cuda-graph-bs-decode 1
                --cuda-graph-backend-prefill disabled --disable-radix-cache
                --weight-loader-disable-mmap --weight-loader-drop-cache-after-load
                --model-loader-extra-config '{"enable_multithread_load": false}'
                --random-seed 42 --watchdog-timeout 1800 --log-level warning
                --tool-call-parser ripple --reasoning-parser ripple
            """)
        )
        super().__init__(server_args=args)
        self._chat = OpenAIServingChat(self.tokenizer_manager, self.template_manager)

    def complete(self, messages, **kwargs):
        """Return an OpenAI-style response with reasoning and tool calls, locally."""
        request = ChatCompletionRequest(
            **{
                "model": "ripple",
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0,
                "chat_template_kwargs": {"clear_thinking": False},
                **kwargs,
            }
        )
        if request.stream:
            raise ValueError("complete() returns a full response; use generate() for raw streaming")
        response = self.loop.run_until_complete(self._chat.handle_request(request, None))
        if isinstance(response, ChatCompletionResponse):
            return response.model_dump(exclude_none=True)
        raise ValueError(json.loads(response.body))

    def chat(self, prompt, *, max_tokens=256, temperature=0):
        """Return final-answer text; use complete() to retain reasoning and tool calls."""
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        response = self.complete(messages, max_tokens=max_tokens, temperature=temperature)
        return response["choices"][0]["message"].get("content") or ""
