#!/usr/bin/env bash
# Stop this recipe's container and archive its logs. Weights stay untouched.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
lock_recipe
if ! container_exists; then
  log "Container $CONTAINER_NAME does not exist; nothing to stop."
  docker volume inspect "$RUNTIME_VOLUME" >/dev/null 2>&1 && log "Runtime volume $RUNTIME_VOLUME is kept."
  exit 0
fi
docker stop -t "$STOP_TIMEOUT" "$CONTAINER_NAME" || true
docker rm "$CONTAINER_NAME" || true
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if compgen -G "$LOG_DIR/run-*" >/dev/null; then
  mkdir -p "$LOG_DIR/archive"
  tar -C "$LOG_DIR" -czf "$LOG_DIR/archive/ripple65b-nvfp4-$STAMP.tar.gz" run-* 2>/dev/null || true
  log "Run logs archived to $LOG_DIR/archive/ripple65b-nvfp4-$STAMP.tar.gz"
fi
log 'Stopped. The container and the model snapshot are preserved; ./start.sh brings it back.'
