## What & why

<!-- One concern per PR. If bundling was unavoidable, give a per-concern
     LOC breakdown so pieces can be vetoed independently. -->

## Checklist

- [ ] No version bump / CHANGELOG entry unless this PR deploys or tags a release (then `scripts/release_notes.py` drafts it)
- [ ] Title reads as a changelog bullet (`Package: what changed`)
- [ ] Tests: exact counts reported below (not "tests pass")
- [ ] Base branch is `master` (or your `users/<name>`) — see
      CONTRIBUTING.md § "Branch topology"; master merges are
      maintainer-only
- [ ] Public-repo hygiene: no lab accounts/hostnames/home paths
- [ ] Review posted at the tier CONTRIBUTING step 4 sets (adversarial,
      or one correctness round; a bulk promotion states its exception)
      and every finding dispositioned (brief in `.claude/skills/land/SKILL.md`)

## Overrides of maintainer rulings

<!-- Anything this PR does that departs from a ruling the maintainer made
     (a design decision, a "keep/drop" call, a review tier, a scope he
     set): what the ruling was, what this does instead, and why. "None" if
     none — say it explicitly, so silence never reads as "none". -->

<!-- None — or: the ruling / what this PR does instead / why -->

## Tests

<!-- e.g. "GeecsBluesky: 480 passed (3 new in tests/test_x.py)" -->

## Hardware verification

<!-- Required for anything touching scan execution, devices, the gateway,
     or shot control. One of:
     - VERIFIED: <what was run, where, and the numbers>
     - OWED: <exact acceptance test + expected result, for the next lab session>
     - N/A: <why this change has no hardware-observable surface> -->
