#!/usr/bin/env bash
# Shell lint, Python syntax, lock and vendored-file integrity. CPU-only.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$REPO_ROOT"
fails=0
for f in start.sh stop.sh status.sh logs.sh prepare.sh doctor.sh smoke.sh bench.sh scripts/*.sh; do
  if ! bash -n "$f" 2> >(sed 's/^/  /' >&2); then
    printf 'FAIL bash syntax: %s\n' "$f" >&2
    fails=$((fails + 1))
  fi
done
for f in scripts/*.py inference/local/*.py; do
  if ! python3 -m py_compile "$f" 2> >(sed 's/^/  /' >&2); then
    printf 'FAIL python syntax: %s\n' "$f" >&2
    fails=$((fails + 1))
  fi
done
python3 - <<'EOF'
import hashlib, json, sys
from pathlib import Path
lock = json.loads(Path('config/model-lock.json').read_text())
bad = 0
for rel, want in lock['vendored'].items():
    got = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    if got != want:
        print(f'FAIL vendored {rel}: sha256 {got} != lock {want}', file=sys.stderr)
        bad += 1
target, draft = lock['target'], lock['draft']
if target['revision'] != '99eea4477a001a41b86914e41feb4b2ed86105e5':
    print(f"FAIL unexpected target revision {target['revision']}", file=sys.stderr)
    bad += 1
if target['total_size_bytes'] != 57213636640 or target['shards'] != 6:
    print('FAIL lock target shard/total_size mismatch', file=sys.stderr)
    bad += 1
if draft['revision'] != '35609b3be3dd08a822b3ba1257c1ea5dd6f87632':
    print(f"FAIL unexpected draft revision {draft['revision']}", file=sys.stderr)
    bad += 1
sys.exit(1 if bad else 0)
EOF
if [[ $? -ne 0 ]]; then fails=$((fails + 1)); fi
rm -f scripts/__pycache__/*.pyc inference/local/__pycache__/*.pyc 2>/dev/null || true
if [[ $fails -ne 0 ]]; then
  printf 'check.sh: %d failure(s)\n' "$fails" >&2
  exit 1
fi
printf 'check.sh: all checks passed\n'
