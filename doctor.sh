#!/usr/bin/env bash
# Preflight: docker, image, GPUs, snapshot completeness, port. No side effects.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
validate_config
exec python3 "$REPO_ROOT/scripts/doctor.py"
