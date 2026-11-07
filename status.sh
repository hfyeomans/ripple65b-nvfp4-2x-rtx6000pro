#!/usr/bin/env bash
# Container state, API readiness, GPU summary. Read-only.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
if container_exists; then
  printf 'container : %s (state=%s, restart=%s)\n' \
    "$CONTAINER_NAME" \
    "$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")" \
    "$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' "$CONTAINER_NAME")"
else
  printf 'container : %s (absent)\n' "$CONTAINER_NAME"
fi
if curl -fsS --max-time 5 "$BASE_URL/health" >/dev/null 2>&1; then
  printf 'api       : %s/v1 healthy\n' "$BASE_URL"
  curl -fsS --max-time 5 "$BASE_URL/v1/models" | python3 -c 'import json,sys; print("models    :", ", ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null || true
else
  printf 'api       : not responding at %s/v1\n' "$BASE_URL"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv 2>/dev/null || true
