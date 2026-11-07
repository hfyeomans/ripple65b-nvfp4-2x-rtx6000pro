#!/usr/bin/env bash
# Fetch only the pinned serving image; never fetch model weights or use GPUs.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
if [[ "${1:-}" == --dry-run || "${DRY_RUN:-0}" == 1 ]]; then
  plan docker pull "$IMAGE"
  log 'Would record the resolved image ID and registry digest in .state/image.json.'
  exit 0
fi
[[ $# == 0 ]] || fail 'Usage: ./prepare.sh [--dry-run]'
lock_recipe
docker info >/dev/null 2>&1 || fail 'Docker daemon is unavailable.'
docker pull "$IMAGE"
mkdir -p "$STATE_DIR"
docker image inspect "$IMAGE" --format '{{json .}}' | python3 -c '
import json, os, sys
from pathlib import Path
image = json.load(sys.stdin)
record = {"requested_image": os.environ["IMAGE"], "image_id": image["Id"],
          "repo_digests": image.get("RepoDigests", [])}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
' "$STATE_DIR/image.json"
log 'Container image prepared. Run ./doctor.sh before ./start.sh.'
