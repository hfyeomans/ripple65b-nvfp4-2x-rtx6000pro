#!/usr/bin/env bash
# shellcheck disable=SC2034
# Shared configuration. .env is literal data, never sourced or evaluated.
# Layering: built-in defaults, then .env, then exported environment (highest).
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RECIPE_LABEL=ripple65b-nvfp4-sglang-v1
MODEL_LOCK=$REPO_ROOT/config/model-lock.json
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '%s\n' "$*"; }
plan() { printf '  '; printf '%q ' "$@"; printf '\n'; }

# .env first: only keys not already exported.
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || fail ".env: invalid key '$key'"
    value=${value%$'\r'}
    [[ -v "$key" ]] || export "$key=$value"
  done < "$REPO_ROOT/.env"
fi

export HF_CACHE_ROOT=${HF_CACHE_ROOT:-/models_engrams/huggingface}
export MODEL_REVISION=${MODEL_REVISION:-99eea4477a001a41b86914e41feb4b2ed86105e5}
export MODEL_SNAPSHOT_DIR=${MODEL_SNAPSHOT_DIR:-$HF_CACHE_ROOT/hub/models--ricursiveAI--Ripple-65B-Preview-NVFP4/snapshots/$MODEL_REVISION}
export DRAFT_REVISION=${DRAFT_REVISION:-35609b3be3dd08a822b3ba1257c1ea5dd6f87632}
export DRAFT_SNAPSHOT_DIR=${DRAFT_SNAPSHOT_DIR:-$HF_CACHE_ROOT/hub/models--ricursiveAI--Ripple-65B-Preview-DFlash2/snapshots/$DRAFT_REVISION}
export IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.19@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9}
export CONTAINER_NAME=${CONTAINER_NAME:-ripple65b-nvfp4-sglang-tp2}
export RUNTIME_VOLUME=${RUNTIME_VOLUME:-ripple65b-nvfp4-runtime}
export GPUS=${GPUS:-0,1}
export TP_SIZE=${TP_SIZE:-2}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-30001}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-ripple-65b-nvfp4}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.78}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfloat16}
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-4}
export NUM_DRAFT_TOKENS=${NUM_DRAFT_TOKENS:-32}
export DRAFT_WINDOW_SIZE=${DRAFT_WINDOW_SIZE:-16384}
export SWA_RATIO=${SWA_RATIO:-}
export MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-88000}
export WAIT_TIMEOUT=${WAIT_TIMEOUT:-1800}
export SMOKE_TIMEOUT=${SMOKE_TIMEOUT:-300}
export STOP_TIMEOUT=${STOP_TIMEOUT:-45}
export RESTART_POLICY=${RESTART_POLICY:-no}
export DOCKER_RUN_FLAGS=${DOCKER_RUN_FLAGS:-}
export PYTHONDONTWRITEBYTECODE=1

STATE_DIR=$REPO_ROOT/.state
LOG_DIR=$REPO_ROOT/logs
API_HOST=$HOST
[[ "$API_HOST" != 0.0.0.0 && "$API_HOST" != :: ]] || API_HOST=127.0.0.1
[[ "$API_HOST" != *:* ]] || API_HOST=[$API_HOST]
RECIPE_BASE_URL=http://$API_HOST:$PORT
export BASE_URL=${BASE_URL:-$RECIPE_BASE_URL}

validate_config() {
  [[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "GPUS must be a comma-separated list of GPU indices (got '$GPUS')"
  GPU_COUNT=$(tr ',' '\n' <<<"$GPUS" | wc -l)
  (( GPU_COUNT >= 1 && GPU_COUNT <= 8 )) || fail "GPUS must list 1..8 GPU indices"
  [[ "$TP_SIZE" =~ ^[1-8]$ ]] || fail "TP_SIZE must be 1..8 (got '$TP_SIZE')"
  (( TP_SIZE == GPU_COUNT )) || fail "TP_SIZE ($TP_SIZE) must equal the GPUS count ($GPU_COUNT): one container runs the whole tensor-parallel group"
  case "$TP_SIZE" in 1|2|4|8) ;; *) fail "TP_SIZE must divide the model's 80 attention heads and 8 KV heads: use 1, 2, 4 or 8" ;; esac
  (( MAX_MODEL_LEN >= 8192 && MAX_MODEL_LEN <= 1048576 )) || fail "MAX_MODEL_LEN must be in [8192, 1048576] (got $MAX_MODEL_LEN)"
  python3 -c "
import sys
f = float('$MEM_FRACTION_STATIC')
sys.exit(0 if 0.05 < f <= 0.98 else 1)" || fail "MEM_FRACTION_STATIC must be in (0.05, 0.98]"
  [[ "$ATTENTION_BACKEND" == flashinfer || "$ATTENTION_BACKEND" == triton ]] \
    || fail "ATTENTION_BACKEND must be one of: flashinfer, triton"
  [[ "$KV_CACHE_DTYPE" == bfloat16 || "$KV_CACHE_DTYPE" == fp8_e4m3 ]] \
    || fail "KV_CACHE_DTYPE must be bfloat16 (vendor parity) or fp8_e4m3 (long-context profile)"
  [[ "$PORT" =~ ^[0-9]+$ ]] || fail "PORT must be numeric"
  if [[ -n "$SWA_RATIO" ]]; then
    python3 -c "
r = float('$SWA_RATIO')
t = r * $MAX_MODEL_LEN
import sys
sys.exit(0 if r > 0 and t >= 1088 else 1)" || fail "SWA_RATIO must be positive and leave the SWA pool at least 1088 tokens (sliding window 1024 + page 64)"
  fi
  (( NUM_DRAFT_TOKENS >= 1 && NUM_DRAFT_TOKENS <= 64 )) || fail "NUM_DRAFT_TOKENS must be in [1, 64]"
  [[ "$RESTART_POLICY" == no || "$RESTART_POLICY" == always || "$RESTART_POLICY" == unless-stopped || "$RESTART_POLICY" == on-failure* ]] \
    || fail "RESTART_POLICY must be no, always, unless-stopped or on-failure[:N]"
  [[ -n "$IMAGE" ]] || fail "IMAGE is unset"
}

lock_recipe() {
  mkdir -p "$STATE_DIR"
  exec 9>"$STATE_DIR/recipe.lock"
  flock -w 120 9 || fail 'Another recipe command holds the lock; retry shortly.'
}

container_exists() { docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; }

build_serve_args() {
  # The hub snapshots are symlinks into the repo's blobs/ dirs; both sides must be visible
  # at the same absolute paths inside the container for the links to resolve.
  HUB_REAL=$(realpath -m "$HF_CACHE_ROOT")
  SNAP_REAL=$(realpath -m "$MODEL_SNAPSHOT_DIR")
  DRAFT_REAL=$(realpath -m "$DRAFT_SNAPSHOT_DIR")
  MODEL_CONTAINER_PATH=$SNAP_REAL
  DRAFT_CONTAINER_PATH=$DRAFT_REAL
  SERVE_ARGS=(--host "$HOST" --port "$PORT" --served-model-name "$SERVED_MODEL_NAME"
    --model-path "$MODEL_CONTAINER_PATH" --chat-template "$MODEL_CONTAINER_PATH/chat_template.jinja"
    --tp-size "$TP_SIZE"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --speculative-draft-model-path "$DRAFT_CONTAINER_PATH"
    --speculative-num-draft-tokens "$NUM_DRAFT_TOKENS"
    --speculative-draft-window-size "$DRAFT_WINDOW_SIZE"
    --context-length "$MAX_MODEL_LEN" --mem-fraction-static "$MEM_FRACTION_STATIC"
    --attention-backend "$ATTENTION_BACKEND" --max-running-requests "$MAX_RUNNING_REQUESTS")
  if [[ -n "${SERVE_EXTRA_ARGS:-}" ]]; then
    # Intentional word-split: extra SGLang CLI flags.
    # shellcheck disable=SC2206
    SERVE_ARGS+=($SERVE_EXTRA_ARGS)
  fi
  if [[ -n "$SWA_RATIO" ]]; then
    SERVE_ARGS+=(--swa-full-tokens-ratio "$SWA_RATIO")
  fi
}

build_docker_args() {
  local image_id=$1
  mkdir -p "$STATE_DIR"
  docker volume inspect "$RUNTIME_VOLUME" >/dev/null 2>&1 || docker volume create "$RUNTIME_VOLUME" >/dev/null
  DOCKER_ARGS=(docker run -d --name "$CONTAINER_NAME" --restart "$RESTART_POLICY"
    --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864
    --gpus "\"device=$GPUS\""
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false
    -e RECIPE_KV_CACHE_DTYPE="$KV_CACHE_DTYPE"
    -v "$HUB_REAL:$HUB_REAL:ro"
    -v "$SNAP_REAL:$SNAP_REAL:ro"
    -v "$DRAFT_REAL:$DRAFT_REAL:ro"
    -v "$REPO_ROOT/inference:/recipe/inference:ro"
    -v "$RUNTIME_VOLUME:/root")
  if [[ -n "${SGLANG_API_KEY:-}" ]]; then
    # Passed by name, never printed.
    DOCKER_ARGS+=(-e SGLANG_API_KEY="$SGLANG_API_KEY")
  fi
  if [[ -n "${DOCKER_RUN_FLAGS:-}" ]]; then
    # shellcheck disable=SC2206
    DOCKER_ARGS+=($DOCKER_RUN_FLAGS)
  fi
  DOCKER_ARGS+=(--entrypoint python "$image_id" /recipe/inference/local/serve_nvfp4.py "${SERVE_ARGS[@]}")
}
