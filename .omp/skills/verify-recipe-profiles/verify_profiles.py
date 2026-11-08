#!/usr/bin/env python3
"""Property tests for the launcher contract. Stdlib only.

Property. Every named profile produces exactly the documented launch options and passes
validation, every undocumented profile name is rejected, and every invalid option set is
rejected by validate_config.

Why generated inputs instead of a verifier. The launcher is bash, and no formal verifier
runs on bash. These tests drive the real scripts with generated inputs and compare the
effective values against the golden table.

Usage.
  python3 verify_profiles.py            # run from a recipe root
  python3 verify_profiles.py --repo /path/to/recipe
Exit 0 when every check passes, 1 otherwise.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent

PROBE = r'''
source scripts/common.sh
validate_config >/dev/null 2>&1
valid=$?
build_serve_args >/dev/null 2>&1
built=$?
python3 - "$valid" "$built" <<'PYEOF'
import json, os, sys
d = {
    "gpus": os.environ.get("GPUS", ""),
    "tp": os.environ.get("TP_SIZE", ""),
    "kv": os.environ.get("KV_CACHE_DTYPE", ""),
    "frac": os.environ.get("MEM_FRACTION_STATIC", ""),
    "swa": os.environ.get("SWA_RATIO", ""),
    "maxlen": os.environ.get("MAX_MODEL_LEN", ""),
    "validate_ok": sys.argv[1] == "0",
    "build_ok": sys.argv[2] == "0",
}
print(json.dumps(d))
PYEOF
'''

NEGATIVE_CASES = [
    {"name": "fraction above ceiling", "env": {"MEM_FRACTION_STATIC": "0.99"}},
    {"name": "fraction below floor", "env": {"MEM_FRACTION_STATIC": "0.01"}},
    {"name": "TP not in 1 2 4 8", "env": {"GPUS": "0,1,2", "TP_SIZE": "3"}},
    {"name": "TP does not match GPU count", "env": {"GPUS": "0,1", "TP_SIZE": "4"}},
    {"name": "SWA floor violated", "env": {"SWA_RATIO": "0.001", "MAX_MODEL_LEN": "131072"}},
    {"name": "context above ceiling", "env": {"MAX_MODEL_LEN": "99999999"}},
    {"name": "context below floor", "env": {"MAX_MODEL_LEN": "1024"}},
    {"name": "unknown profile", "env": {}, "profile": "bogus"},
]


def run_probe(repo, profile, extra_env):
    env = {"PROFILE": profile}
    env.update(extra_env)
    out = subprocess.run(
        ["bash", "-c", PROBE], cwd=repo, env={**__import__("os").environ, **env},
        capture_output=True, text=True, timeout=60)
    return out.returncode, out.stdout.strip().splitlines()


def check(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else ": " + detail))
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path.cwd()))
    args = parser.parse_args()
    repo = Path(args.repo)
    golden = json.loads((SKILL_DIR / "profiles_golden.json").read_text())
    failures = 0

    for entry in golden["profiles"]:
        name = entry["name"]
        code, lines = run_probe(repo, name, {})
        if code != 0 or not lines:
            failures += 0 if check(name, False, f"probe failed rc={code}") else 1
            continue
        got = json.loads(lines[-1])
        want = entry["options"]
        mismatches = [k for k in ("gpus", "tp", "kv", "frac", "swa", "maxlen")
                      if str(got.get(k)) != str(want.get(k))]
        if not got.get("validate_ok"):
            mismatches.append("validate")
        if not got.get("build_ok"):
            mismatches.append("build")
        failures += 0 if check(name, not mismatches, str(mismatches) + " got " + json.dumps(got)) else 1

    for case in NEGATIVE_CASES:
        profile = case.get("profile", "")
        code, _ = run_probe(repo, profile, case["env"])
        # A negative case passes the property when the probe fails, either in
        # apply_profile or in validate_config.
        failures += 0 if check("reject " + case["name"], code != 0, f"rc={code}") else 1

    if failures:
        print(f"{failures} failure(s)")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
