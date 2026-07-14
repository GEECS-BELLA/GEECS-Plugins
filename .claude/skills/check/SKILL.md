---
name: check
description: >
  Run repo lint + unit tests the way CI does, scoped to what changed. Use
  before committing ("run the checks", "does this pass", "validate my
  changes"), before pushing or opening a PR, or whenever you need to know
  which package suites a diff requires and which env runs each. Wraps
  scripts/check.sh. For environment-shaped failures (wrong python version,
  missing extras, missing venv) switch to /env-doctor.
---

# /check — lint + tests the way CI runs them

`scripts/check.sh` owns the CI mapping (which package runs from which env,
with which markers). This skill is about running the right tier at the
right moment and reading the output — do not re-derive the mapping by
hand, and update the script (not prose) when CI changes.

## Tiers

| When | Command | What runs |
|---|---|---|
| Before every commit | `./scripts/check.sh --lint` | pre-commit on changed files (~seconds) |
| Before push / while iterating | `./scripts/check.sh` | lint + the suites of changed packages |
| Before opening a PR | `./scripts/check.sh --all` | lint everything + every locally runnable suite |

Escalate in order — catch cheap things cheaply. `$ARGUMENTS`, if given,
is passed straight through (package names, `--base <ref>`, `--dry-run`).

## Reading the output

- The plan header lists which suites the diff implies and which touched
  packages have **no CI-mirrored suite** (e.g. GEECS-Scanner-GUI,
  GEECS-PythonAPI) — those are skipped, not covered. Say so when
  reporting results; silence is not coverage.
- Report **exact counts** ("480 passed, 1 skipped"), never "tests pass".
- A whole layer collecting as `1 skipped`, `Command not found: pytest`,
  or "Current Python version … is not allowed" is an environment
  problem, not a code problem → `/env-doctor`.
- Root `tests/` selecting nothing is normal (all integration-marked);
  the script tolerates pytest exit 5 there, same as CI.
- `integration`-marked tests are always deselected — they need the lab
  network and are out of scope for this script by design.

## Relationship to committing

`/check --lint` and `scripts/commit.sh` share the same pre-commit
invocation; commit.sh applies the auto-fixes at commit time regardless,
so a lint failure here is a preview, not a gate you can skip.
