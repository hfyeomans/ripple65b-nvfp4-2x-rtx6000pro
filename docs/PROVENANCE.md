# Provenance, download record and decision record (NVFP4 recipe)

## Checkpoints (downloaded into the local hub 2026-11-07)

All downloads went through `hf download` (huggingface_hub 1.29.0) with
`HF_HOME=/models_engrams/huggingface HF_HUB_CACHE=/models_engrams/huggingface/hub`, i.e. the same
hub layout the BF16 recipe reads (`/models_engrams` → `/mnt/models_engrams`). Snapshots are
symlinks into per-repo `blobs/` dirs; the recipe mounts the hub root and both snapshot dirs at
their real host paths inside the container.

| Repository | Revision | What it is | Size |
| --- | --- | --- | --- |
| `ricursiveAI/Ripple-65B-Preview-NVFP4` | `99eea4477a001a41b86914e41feb4b2ed86105e5` | NVFP4 (ModelOpt, group 16) MLP + BF16 attention instruction-tuned checkpoint, 6 shards | 57,213,636,640 bytes |
| `ricursiveAI/Ripple-65B-Preview-DFlash2` | `35609b3be3dd08a822b3ba1257c1ea5dd6f87632` | Unquantized BF16 DFlash2 draft (single `model.safetensors`), 16K training rows, 16K attention window | see `config/model-lock.json` |
| `ricursiveAI/Ripple-65B-Preview` (BF16) | `e27e171792f303d74c048184e8259ba663cfa0d5` | Already local; served by the sibling BF16 recipe | 129,565,378,720 bytes |

Licenses: Apache 2.0 for both new repositories.

## Vendored files

Copied verbatim from the NVFP4 snapshot's `inference/local/` (Apache 2.0). Hashes are pinned in
`config/model-lock.json` and re-checked by `scripts/check.sh`:

| File | sha256 |
| --- | --- |
| `inference/local/model.py` | `007ec77d0d64d3007d68a256f726ff69318abf1a45ab5de08828979425e5b986` |
| `inference/local/ripple_format.py` | `19e432554ad48c57a1b67b989695ff4e35a44ac769821ad92ba229ecf5266ace` |
| `inference/local/spark.py` | `17cbc3414673d9e0e98de82360e485e9b73c9a2bbc9c93f39c09e2ea2548a8e6` |

`spark.py` is kept unmodified as the vendor's DGX Spark (sm_121, TP1, offline engine) reference;
`serve_nvfp4.py` (this repository) is the host-fitted HTTP entrypoint.

## Serving image

- `lmsysorg/sglang:v0.5.19` pinned by digest
  `sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9` —
  the same image the BF16 recipe qualified (Python 3.12.3, torch 2.13.0+cu130,
  transformers 5.12.1, flashinfer 0.6.18). NVFP4 GEMMs run on `flashinfer_cutlass`;
  FP8-GEMM autotune is skipped per the vendor's flags.

## Decision record

1. **Separate repository** (`ricursiveAI-recipe-2x-rtx6000pro-nvfp4`): different checkpoint,
   quantization path (ModelOpt NVFP4), speculation config and memory geometry than the qualified
   BF16 recipe. Mirrors the per-engine/per-quant repo convention of the guide recipes.
2. **Vendor stack kept**: the NVFP4 repo's own SGLang `model.py` threads a `QuantizationConfig`
   through every linear (the BF16 repo's version refuses quantized checkpoints). SGLang picks up
   the ModelOpt NVFP4 recipe from `config.json` (`quant_method: modelopt`, `quant_algo: NVFP4`,
   group 16); no explicit `--quantization` flag, matching spark.py.
3. **DFLASH on**: the draft was downloaded (vendor requires an unquantized BF16
   `DFlash2DraftModel`); 32 draft tokens, 16K window, triton draft attention — the vendor's
   numbers. Draft weights are BF16; `model.py`'s loader patches keep strict draft loading.
4. **BF16 KV default, FP8 opt-in**: the vendor's Spark recipe pins BF16 KV and neutralizes the
   FP8 k/v scales the quantized export carries; `serve_nvfp4.py` reproduces that for
   `KV_CACHE_DTYPE=bfloat16` and keeps the real scales only under `fp8_e4m3`.
5. **sm_121-only fixes dropped**: spark.py's PDL softmax NaN fix and its GB10 capability check
   are omitted (sm_120 host, two discrete GPUs, TP2).
6. **Pool math**: measured SGLang 0.5.19 budget `free-after-load − total × (1 − f)` ≈ `93.9f − 26.9`
   GiB with NVFP4 weights → ≈ 46 GiB at the default f=0.78. See the README memory section.
