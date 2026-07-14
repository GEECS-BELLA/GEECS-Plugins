## What & why

<!-- One concern per PR. If bundling was unavoidable, give a per-concern
     LOC breakdown so pieces can be vetoed independently. -->

## Checklist

- [ ] `poetry version patch|minor` run for every package whose code changed
- [ ] `CHANGELOG.md` entry added under the new version (Keep a Changelog)
- [ ] Tests: exact counts reported below (not "tests pass")
- [ ] Base branch matches the content — see CONTRIBUTING.md
      § "Branch topology" (delete this line at M6)
- [ ] Public-repo hygiene: no lab accounts/hostnames/home paths

## Tests

<!-- e.g. "GeecsBluesky: 480 passed (3 new in tests/test_x.py)" -->

## Hardware verification

<!-- Required for anything touching scan execution, devices, the gateway,
     or shot control. One of:
     - VERIFIED: <what was run, where, and the numbers>
     - OWED: <exact acceptance test + expected result, for the next lab session>
     - N/A: <why this change has no hardware-observable surface> -->
