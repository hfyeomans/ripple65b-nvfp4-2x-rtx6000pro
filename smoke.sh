#!/usr/bin/env bash
# API acceptance suite against a running server (./start.sh must have succeeded).
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
export BASE_URL=$RECIPE_BASE_URL
OUT=${1:-$LOG_DIR/acceptance}
[[ "$OUT" != --* ]] || OUT=$LOG_DIR/acceptance
python3 "$REPO_ROOT/scripts/smoke.py" --wait 5 --suite --output "$OUT"
log "Receipts: $OUT"
