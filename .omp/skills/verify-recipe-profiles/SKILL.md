---
name: verify-recipe-profiles
description: Use when a change touches the launcher settings, validation, or the PROFILE case in scripts/common.sh, or when a new profile is added. Verifies every named profile produces exactly its documented options and every invalid option set is rejected.
---

# Verify the launcher profile contract

Property. Every named profile produces exactly the documented launch options and passes validation. Every undocumented profile name and every invalid option set is rejected.

Harm if broken. A profile silently launches with the wrong GPUs, KV dtype, memory fraction, or context window. This is the exact bug class that burned this project twice, when duplicate `export` lines made the wrong value win, and it does not fail any other check in the repo.

Tool. The launcher is bash, and no formal verifier runs on bash, so these are generated-input property tests against the real scripts, not a model. The harness runs `scripts/common.sh` in a fresh bash, sets `PROFILE`, and compares the effective exported values and `validate_config` against `profiles_golden.json`.

## Coverage record

Covered files and what each check treats as one step.

- `scripts/common.sh`, function `apply_profile`, the `PROFILE` case. One step is one case arm. The check compares each arm's effective `GPUS`, `TP_SIZE`, `KV_CACHE_DTYPE`, `MEM_FRACTION_STATIC`, `SWA_RATIO`, and `MAX_MODEL_LEN` against the golden table.
- `scripts/common.sh`, function `validate_config`. The check requires it to reject eight invalid sets, named in `verify_profiles.py` under `NEGATIVE_CASES`.
- `scripts/common.sh`, function `build_serve_args`. The check requires it to succeed for every golden profile so the profile values reach the actual launch.

Sync commit. `0d37c70` is the commit at which this verification was confirmed to describe the code. If any covered file changes, run the check. Move the sync commit forward only after you confirm the verification still describes the code at the new commit.

Bounds and assumptions. This covers the shell layer only. It does not verify engine behavior, memory sizing, or anything inside the container. It assumes no `.env` file exists in the recipe root, because the contract states profiles win over `.env` for their keys and the repos ship none.

## Run it

From the recipe root.

```sh
python3 .omp/skills/verify-recipe-profiles/verify_profiles.py
```

One command, no dependencies, under a minute. Exit 0 means every check passed.

## Read a failure

Each FAIL line names the profile or the rejected case and the effective values the probe saw.

- A FAIL on a profile name means the profile's effective values drifted from the golden table. Open `profiles_golden.json` and the `apply_profile` case. If the code is wrong, fix the code. If the table was changed on purpose, update the golden file and move the sync commit.
- A FAIL on a reject case means `validate_config` or `apply_profile` stopped rejecting an invalid set, usually a removed or weakened check in `validate_config`. Restore the check.

## Update the verification when a change is intended

When you add a profile, add its row to `profiles_golden.json` with the exact documented values and run the harness. When you change a covered function on purpose, run the harness, confirm it either passes or the new behavior is intended, update the golden file, and move the sync commit in the Coverage record to the new HEAD. Never weaken a check just to make it pass.
