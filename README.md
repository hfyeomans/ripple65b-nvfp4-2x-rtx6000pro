# Ripple-65B Preview NVFP4 + DFlash2 on two RTX PRO 6000 with SGLang

Serve and test the local NVFP4 checkpoint of
> **Roadmap:** benchmarking, tuning and post-training plan — see
> [../ricursiveAI-2x-rtx6000pro-plan/PLAN.md](../ricursiveAI-2x-rtx6000pro-plan/PLAN.md) and the
> consolidated [SESSION-RECORD.md](../ricursiveAI-2x-rtx6000pro-plan/SESSION-RECORD.md) in the
> optional sibling planning repository. This recipe runs without it.
[ricursiveAI/Ripple-65B-Preview-NVFP4](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4)
with its [DFlash2 draft](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2) on two
RTX PRO 6000 Blackwell (96 GiB each, sm_120) GPUs, tensor parallel 2, in one Docker container
running **SGLang 0.5.19**.

The NVFP4 checkpoint is the vendor's "experimental" conservative quantization: **NVFP4 (ModelOpt,
group size 16) for the MLP layers, BF16 for attention, embeddings and LM head** — 53.3 GB on disk
versus 120.7 GB for the BF16 repo. This recipe adds the vendor's **native DFlash2 speculative
decoding** (32 draft tokens, 16K draft window), the setup their shipped recipe targets.

```sh
./prepare.sh            # pull the pinned SGLang image once (already pulled for the BF16 recipe)
./doctor.sh             # docker, image, GPUs, NVFP4 snapshot, DFlash2 draft, port
DRY_RUN=1 ./start.sh    # print the exact docker run plan
./start.sh              # start, wait for the API, send a completion
./smoke.sh --suite      # reasoning, tools, streaming acceptance
./bench.sh              # TTFT / prefill / decode throughput (speculative)
./stop.sh               # stop; archive the run logs
```

## What differs from the BF16 recipe (`ricursiveAI-recipe-2x-rtx6000pro`)

| | BF16 recipe | This recipe |
| --- | --- | --- |
| Checkpoint | `Ripple-65B-Preview` @ `e27e171…` (120.7 GB) | `Ripple-65B-Preview-NVFP4` @ `99eea44…` (53.3 GB) |
| Weights resident | ≈ 60.4 GiB/GPU | ≈ 26.6 GiB/GPU (MLP FP4, attention BF16) |
| Speculation | off (draft was not local) | **DFLASH on**: draft `Ripple-65B-Preview-DFlash2` @ `35609b3…`, 32 draft tokens, 16K window |
| FP4 GEMMs | — | `--fp4-gemm-backend flashinfer_cutlass`, FP8-GEMM autotune skipped |
| KV cache | BF16 only | BF16 default (vendor parity); `KV_CACHE_DTYPE=fp8_e4m3` opt-in |
| Port / model name | 30000 / `ripple-65b-preview` | 30001 / `ripple-65b-nvfp4` |

The serving entrypoint is [inference/local/serve_nvfp4.py](inference/local/serve_nvfp4.py),
adapted from the checkpoint's shipped `inference/local/spark.py` (a DGX Spark offline engine,
hard-pinned to sm_121 TP1): HTTP server, TP2, radix cache kept, spark's sm_121-only PDL softmax
fix omitted. Its k/v-scale neutralization (quantized exports carry FP8 scales although KV is
BF16) is kept for `KV_CACHE_DTYPE=bfloat16` and skipped for `fp8_e4m3`, which keeps the real
scales. `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` is set by default so the target context can
exceed the draft checkpoint's 128Ki positional limit (as the upstream launcher does); the draft
attention window stays fixed. [inference/local/model.py](inference/local/model.py) and
[ripple_format.py](inference/local/ripple_format.py) are vendored verbatim from the checkpoint
(hashes in [config/model-lock.json](config/model-lock.json)).

## Memory on 96 GB cards

NVFP4 weights total 57,213,636,640 bytes (≈ 26.6 GiB per GPU at TP2). The measured SGLang 0.5.19
pool budget is `free-after-load − total × (1 − MEM_FRACTION_STATIC)`, so with ≈ 67 GiB free after
load the pool is ≈ `93.9 × f − 26.9` GiB. At the default `MEM_FRACTION_STATIC=0.78` that is
≈ 46 GiB — about 700K compact-BF16 KV tokens, capped by the engine ceiling below.

| KV dtype | Per-token cost (compact, global layers) | Pool at 0.78 | Servable context |
| --- | --- | --- | --- |
| `bfloat16` (default) | 64 KiB | ≈ 46 GiB | up to ~512K comfortably; 1M needs 64 GiB + slack |
| `fp8_e4m3` | 32 KiB | measured pool 597,440 tokens at 0.78 | 1M needs `MEM_FRACTION_STATIC≈0.90` (pool ≈ 57 GB) — the long-context profile |

Defaults keep `MAX_MODEL_LEN=131072`, the vendor's qualified ceiling for the local recipe.
`validate_config` allows up to 1,048,576; raise it deliberately and watch the startup log
("Use sliding window memory pool. full_layer_tokens=…") — the API rejects longer inputs with the
exact ceiling. Contexts above the draft checkpoint's 128Ki positional limit need
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, which `serve_nvfp4.py` sets by default; the draft
attention window stays fixed at `DRAFT_WINDOW_SIZE`.

## Settings

Layered: defaults in `scripts/common.sh`, then literal `.env`, then exported environment
variables, which win.

| Setting | Default | Meaning |
| --- | --- | --- |
| `HF_CACHE_ROOT` | `/models_engrams/huggingface` | Host hub cache; both snapshots resolve below it |
| `MODEL_REVISION` / `MODEL_SNAPSHOT_DIR` | `99eea44…` / hub path | NVFP4 checkpoint, mounted read-only |
| `DRAFT_REVISION` / `DRAFT_SNAPSHOT_DIR` | `35609b3…` / hub path | DFlash2 draft, mounted read-only |
| `IMAGE` | `lmsysorg/sglang:v0.5.19@sha256:d6e728…` | Pinned serving image (same as the BF16 recipe) |
| `CONTAINER_NAME` / `RUNTIME_VOLUME` | `ripple65b-nvfp4-sglang-tp2` / `ripple65b-nvfp4-runtime` | Container and cache volume |
| `GPUS` / `TP_SIZE` | `0,1` / `2` | GPU indices and tensor-parallel size; `TP_SIZE` must equal the `GPUS` length and divide the 80 attention / 8 KV heads (valid: 1, 2, 4, 8) |
| `HOST` / `PORT` | `0.0.0.0` / `30001` | API address |
| `SERVED_MODEL_NAME` | `ripple-65b-nvfp4` | Model ID clients send |
| `MAX_MODEL_LEN` | `131072` | Per-request prompt + output ceiling (8K–1M) |
| `MEM_FRACTION_STATIC` | `0.78` | Static/weights fraction; higher f → bigger pool here (see memory math) |
| `ATTENTION_BACKEND` | `flashinfer` | `flashinfer` or `triton`; fa3 is sm80–90 only |
| `KV_CACHE_DTYPE` | `bfloat16` | `bfloat16` (vendor parity) or `fp8_e4m3` (long-context profile) |
| `MAX_RUNNING_REQUESTS` | `4` | Concurrent requests admitted |
| `NUM_DRAFT_TOKENS` | `32` | `--speculative-num-draft-tokens` (engine notes the draft's `block_size` is 16) |
| `DRAFT_WINDOW_SIZE` | `16384` | Draft attention window (fixed by the draft's 16K training rows) |
| `SWA_RATIO` | unset (engine default 0.8) | SWA pool size as a fraction of the full pool; the 1M profile sets `0.02` (SWA layers only need ~1024 tokens per request) |
| `SERVE_EXTRA_ARGS` | empty | More SGLang CLI flags, word-split |
| `SGLANG_API_KEY` | unset | If exported: every route except `/health` needs a bearer key |
| `MIN_GPU_FREE_MIB` | `88000` | Free memory required per GPU |
| `WAIT_TIMEOUT`, `SMOKE_TIMEOUT`, `STOP_TIMEOUT` | `1800`, `300`, `45` | Seconds |
| `RESTART_POLICY` | `no` | Inspect failures; `on-failure:3` after acceptance |

## API

Same OpenAI-compatible surface, template and ripple parsers as the BF16 recipe — reasoning is
`message.reasoning_content`, final answer `message.content`, calls in `message.tool_calls`; tool
round trips use the `<tool_call>` layout with `strict: true` +
`additionalProperties: false` for `tool_choice: "required"`. See the BF16 recipe's README for the
full request shapes; DFLASH speculation is invisible on the wire except for latency.

## Operations

```sh
./status.sh          # container state, API health, models, GPU usage
./logs.sh            # follow server logs
./smoke.sh --suite   # completion, reasoning math, tool round trip, streaming
./start.sh restart   # recreate only this recipe's container
./stop.sh            # stop; preserve container, snapshots and runtime volume
```

Run receipts land in `logs/run-<UTC>/` (symlinked `logs/latest`); `stop.sh` archives them as
`ripple65b-nvfp4-*.tar.gz`. `scripts/check.sh` runs shell lint, Python syntax, lock and
vendored-hash checks — CPU-only. The hub mounts (root + both snapshots at their real host
paths, read-only) preserve the hub symlink layout, as in the BF16 recipe.

## Verification record

- Engine qualification: the NVFP4 `model.py`/`ripple_format.py` import clean in the pinned image
  (ModelOpt FP4 + DFLASH paths present); SGLang 0.5.19 auto-detects the checkpoint as
  "Detected nvfp4 checkpoint … experimental" and loads `quant=modelopt_fp4, quant_algo=NVFP4`.
- Live acceptance on this host (2026-11-07): server up in ≈93 s after `./start.sh` (27.1 GB
  weights + 4.1 GB draft per GPU, 66.9 GB free after load); the suite passes all four checks
  (completion, reasoning-math, tools round trip, streaming). Receipts: `logs/acceptance/*.json`.
- Measured pool at `MEM_FRACTION_STATIC=0.78`, BF16 KV: full-layer pool 305,088 tokens
  (engine ceiling 131,072 at the default `MAX_MODEL_LEN`); the draft loads at 4.14 GB/GPU and
  the engine warns that `speculative_num_draft_tokens=32` differs from the draft's
  `block_size=16` — the vendor's own launcher passes 32, so it is kept.
- Throughput with DFLASH (32 draft tokens): 3.6K prompt → TTFT 0.81 s, ~4.5K prefill tok/s,
  decode 642 tok/s; 28.7K prompt → TTFT 6.22 s, ~4.6K prefill tok/s, decode 102 tok/s
  (vs ≈19 tok/s BF16 without a draft). Prefill is slower than the BF16 recipe (FP4 cutlass
  GEMMs on sm_120); receipts in `logs/bench/benchmark.json`.
- Long context at `MAX_MODEL_LEN=262144` (the pool holds 305,088): a 229,413-token prompt
  completed successfully (BF16 KV, DFLASH on). The 131K default is the vendor's ceiling; 256K
  is the verified upper bound on this host without switching `KV_CACHE_DTYPE` to `fp8_e4m3`.
- 1M-token context verified (fp8 KV): `KV_CACHE_DTYPE=fp8_e4m3 MAX_MODEL_LEN=1048576
  MEM_FRACTION_STATIC=0.90 SWA_RATIO=0.02` sizes the full-layer pool at 2,466,752 tokens and a
  787,537-token prompt completed (TTFT 625 s, ~1.3K prefill tok/s). Two pool knobs matter at
  1M: the static fraction (0.78 → pool 597K, 0.90 → 761K tokens) and `SWA_RATIO` (the default
  0.8 SWA pool wastes ~31 GB; 0.02 recovers it).

## Notes and limitations

- **NVFP4 is the vendor's "experimental" variant**: the benchmark tables on the model cards were
  produced with the BF16 checkpoint. Treat FP4 weights (and FP8 KV, if enabled) as a serving
  trade-off to re-evaluate against the BF16 reference recipe on the shared smoke prompts.
- **`fp8_e4m3` KV is opt-in and unvalidated**: the vendor's Spark recipe pins BF16 KV and
  neutralizes the export's FP8 k/v scales; this recipe keeps that behavior for BF16 KV and only
  keeps real scales under `KV_CACHE_DTYPE=fp8_e4m3`.
- **Both recipes share the two GPUs**: only one server runs at a time. `./stop.sh` in the other
  recipe (or `docker stop`) frees the cards; `./doctor.sh` refuses to start on busy GPUs.
- The DFlash2 draft is BF16 and unquantized (required by the vendor's loader); it was downloaded
  into the hub alongside the checkpoint. See [docs/PROVENANCE.md](docs/PROVENANCE.md) for pins
  and hashes and [docs/DOWNLOADS.md](docs/DOWNLOADS.md) for the download record.
