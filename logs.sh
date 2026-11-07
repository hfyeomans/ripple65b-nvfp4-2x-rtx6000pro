#!/usr/bin/env bash
# Follow this recipe's server logs. Args are passed to `docker logs`.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
if ! container_exists; then
  fail "Container $CONTAINER_NAME does not exist; run ./start.sh first."
fi
exec docker logs --tail 200 -f "$CONTAINER_NAME" "$@"
