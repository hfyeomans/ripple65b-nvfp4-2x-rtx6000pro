# Download record — 2026-11-07

Everything below was downloaded into the local hub
`/models_engrams/huggingface/hub` (real path `/mnt/models_engrams/huggingface/hub`) with
`hf download` (huggingface_hub 1.29.0),
`HF_HOME=/models_engrams/huggingface HF_HUB_CACHE=/models_engrams/huggingface/hub`.
The BF16 Ripple-65B snapshot (revision `e27e171…`) was already local and is unchanged.

## ricursiveAI/Ripple-65B-Preview-NVFP4

- Revision: `99eea4477a001a41b86914e41feb4b2ed86105e5` (repo HEAD at download time)
- Files: `config.json`, `tokenizer*.json`, `chat_template.jinja`, `generation_config.json`,
  `modeling_ricursive.py`, `configuration_ricursive.py`, `model.safetensors.index.json`,
  `model-00001..6-of-00006.safetensors`, `inference/local/{model.py,ripple_format.py,spark.py,ripple.ipynb}`
- Total weight bytes: 57,213,636,640 (≈ 53.3 GiB; 6 shards)
- License: Apache 2.0
- Local path: `hub/models--ricursiveAI--Ripple-65B-Preview-NVFP4/snapshots/99eea44…/`

## ricursiveAI/Ripple-65B-Preview-DFlash2

- Revision: `35609b3be3dd08a822b3ba1257c1ea5dd6f87632` (repo HEAD at download time)
- Files: `config.json`, `model.safetensors` (single shard), `README.md`, `LICENSE`
- Total weight bytes: 8,075,513,104 (≈ 7.5 GiB, unquantized BF16)
- License: Apache 2.0
- Local path: `hub/models--ricursiveAI--Ripple-65B-Preview-DFlash2/snapshots/35609b3…/`

## Notes for future reference

- `hf download <repo> --include PAT...` (CLI 1.29.0) silently ignored some `--include` patterns
  ("Ignoring `--include` since filenames have been explicitly set"); `config.json` had to be
  fetched with an explicit positional filename. Use explicit positional filenames for metadata
  files when patterns miss.
- Both snapshots are hub-layout symlinks into per-repo `blobs/` dirs. Serving mounts the hub
  root and each snapshot dir at their real host paths, read-only (see `scripts/common.sh`).
- Re-downloading an already-pinned revision is a no-op (`hf` resumes/skips complete blobs).
