#!/usr/bin/env bash
# Throughput probe against a running server: TTFT, prefill and decode tok/s.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
export BASE_URL=$RECIPE_BASE_URL
OUT=${OUT:-$LOG_DIR/bench}
python3 "$REPO_ROOT/scripts/benchmark.py" --prompt-tokens 4096 32768 --max-tokens 256 --output "$OUT"
log "Receipt: $OUT/benchmark.json"
