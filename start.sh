#!/usr/bin/env bash
# Usage: ./start.sh [restart|--dry-run]. Starts only this recipe's container.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
MODE=start
case "${1:-}" in
  restart) MODE=restart ;;
  --dry-run) DRY_RUN=1 ;;
  '') ;;
  *) fail 'Usage: ./start.sh [restart|--dry-run]' ;;
esac
(( $# <= 1 )) || fail 'Too many arguments.'
validate_config
# Lifecycle acceptance must check the endpoint this launch is configured to bind.
export BASE_URL=$RECIPE_BASE_URL
log "Ripple-65B NVFP4 + DFlash2 | 2 GPUs ($GPUS) | SGLang | port $PORT | context $MAX_MODEL_LEN | kv $KV_CACHE_DTYPE | draft $NUM_DRAFT_TOKENS"
if [[ -z "${SGLANG_API_KEY:-}" && "$HOST" != 127.0.0.1 && "$HOST" != ::1 && "$HOST" != localhost ]]; then
  log "WARNING: HOST=$HOST exposes an unauthenticated API beyond loopback; set SGLANG_API_KEY or bind HOST to 127.0.0.1."
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  lock_recipe
  build_serve_args
  build_docker_args 'DRY-RUN'
  log 'Dry run: the plan below would be executed.'
  plan "${DOCKER_ARGS[@]}"
  exit 0
fi
lock_recipe
docker info >/dev/null 2>&1 || fail 'Docker daemon is unavailable.'
if container_exists && [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == true ]]; then
  if [[ $MODE == start ]]; then
    log "Container $CONTAINER_NAME is already running; waiting for readiness and smoke-testing."
    if python3 "$REPO_ROOT/scripts/smoke.py" --wait "$WAIT_TIMEOUT" --output "$LOG_DIR/latest"; then
      log "Ready: $BASE_URL/v1 | model: $SERVED_MODEL_NAME"
      exit 0
    fi
    fail 'The running server never became ready; inspect ./logs.sh and ./status.sh.'
  fi
  log "Stopping existing container for restart."
  docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" >/dev/null
  docker rm "$CONTAINER_NAME" >/dev/null
fi
python3 "$REPO_ROOT/scripts/doctor.py" || fail 'Preflight failed. Correct the listed blockers, then rerun.'
RUN_IMAGE=$(docker image inspect "$IMAGE" --format '{{.Id}}')
if [[ -f "$STATE_DIR/image.json" ]]; then
  recorded=$(python3 -c "import json;print(json.load(open('$STATE_DIR/image.json'))['image_id'])" 2>/dev/null || true)
  [[ "$RUN_IMAGE" == "$recorded" ]] || log "WARNING: $IMAGE resolves to $RUN_IMAGE, but .state/image.json records $recorded; run ./prepare.sh to re-pin."
fi
build_serve_args
build_docker_args "$RUN_IMAGE"
RUN_DIR=$LOG_DIR/run-$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p "$RUN_DIR"
ln -sfn "${RUN_DIR##*/}" "$LOG_DIR/latest"
python3 - "$MODEL_SNAPSHOT_DIR" "$RUN_DIR/model.json" <<'EOF'
import json, sys
from pathlib import Path
lock = json.loads(Path('config/model-lock.json').read_text())
meta = {'model_id': lock['target']['model_id'], 'revision': lock['target']['revision'],
        'dtype': lock['target']['dtype'], 'snapshot_dir': sys.argv[1],
        'total_size_bytes': lock['target']['total_size_bytes'],
        'quant': lock['target']['quant_method'] + '/' + lock['target']['quant_algo'],
        'draft_revision': lock['draft']['revision']}
Path(sys.argv[2]).write_text(json.dumps(meta, indent=2) + '\n')
EOF
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,driver_version --format=csv > "$RUN_DIR/gpus.csv"
plan "${DOCKER_ARGS[@]}" > "$RUN_DIR/launch-command.txt"
log "Weights: $MODEL_SNAPSHOT_DIR"
log "Evidence: $RUN_DIR"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
"${DOCKER_ARGS[@]}" > "$RUN_DIR/container-id.txt"
log "Container $CONTAINER_NAME started; waiting for the API (up to ${WAIT_TIMEOUT}s)."
if ! python3 "$REPO_ROOT/scripts/smoke.py" --wait "$WAIT_TIMEOUT" --output "$RUN_DIR"; then
  log "Readiness failed. Server logs: ./logs.sh  (run dir: $RUN_DIR)"
  exit 1
fi
log "Ready: $BASE_URL/v1 | model: $SERVED_MODEL_NAME"
log 'Use ./logs.sh for server logs; ./smoke.sh --suite for API checks; ./bench.sh for throughput.'
