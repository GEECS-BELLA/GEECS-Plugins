# GeecsConfigurator — Developer Context for Claude

One editor over every GEECS-Schemas document in the configs repository
(`geecs-configurator`, port 8800). A stub while the configurator arc runs:
its design is `Planning/configurator.md`, its record the tracking issue
(#1307); the arc's last slice writes this file fully.

## Rules that already bind

- **The store reads; saving is `save.py` through `repository.py`.** The
  store (`store.py`) reads, validates and diffs, and never creates a
  directory. With the commit path on (`STATE_DIR` + `ORIGIN`) the configs
  root is the live checkout, which **nothing writes and nothing
  branch-switches**: it moves only by fast-forward to `origin/main`. Every
  edit is a staging worktree, removed on every path; a failed push reaches
  nothing; a moved base is a 409, never a merge. The worker's writes land
  in the inbox (`inbox.py`), claimed by rename and committed from there.
  Each rule is pinned against scratch bare repos (`tests/test_repository.py`,
  `test_save.py`, `test_inbox.py`).
- **Nothing is applied that did not validate.** Every fast-forward of the
  live checkout passes the gate (`service.CommitPath.accept`): the
  documents changed since the live commit must validate
  (`validation.problems`, the one rule the configs repo's CI runs too, from
  `deploy/configs-repo/validate.yml`). A refused commit is fetched, logged
  once, and blocks every save, every inbox publish and every retry
  until main is fixed by PR: nothing of the service is built on it. A new
  live checkout starts at the share's commit (accepted before, by hand)
  and passes the gate from there; with no usable share it starts
  **unverified** (the `UNVERIFIED` marker in the state directory: the gate
  checks only changes, so no advance clears it), says so, and holds the
  share until the maintainer deletes the marker. A credential failure (GitHub down) is a git failure: stale
  reads, a 502 save, a backed-off inbox, never a 500.
- **A landed commit is never "applied" by that fact.** A log row's state
  (`log.py`: applied / queued / pending / nothing to do) is the apply
  seam's answer (`apply.ServiceApplier`), and applied is the consumer's
  own answer: the scanner's reopen returned, the archiver's onboard sent
  its plan. `apply()` blocks on nothing (work is answered queued);
  `CommitPath.settle_queued` does the work on its own thread every
  `APPLY_SECONDS`, one reopen and one onboard per pass, **holding the
  repository's lock** so no fast-forward rewrites a file while the worker
  or the archiver reads it. A busy worker keeps a row queued; a restart,
  a mass-pause guard or a reader not surveyed is pending, named. A row's
  state is the least finished of its readers', and each reader's own is
  kept on the row (`Row.readers`): a pass retries only the readers still
  queued, so a worker is never reopened again for a row whose onboard
  waits. The archiver's onboard is applied only when it printed its plan
  (a crash exits 1 like drift). Each experiment has its own scanner
  (`SCANNER_URL` as `Exp=url,…`); a row of an experiment without one is
  never applied by another's reopen. A save while a pass runs is answered
  503 at once, never kept waiting on the lock.
- **Dry runs are the consumers' own** (`apply.py`): the scanner's
  `POST /api/preflight/trigger-profile` and `geecs-archiver onboard
  --dry-run --policy <draft>` over a temporary file, never re-implemented
  here. A consumer's refusal is the save's 422 (before any git); a check
  that could not run is a note on the save, never a refusal (the
  archiver's mass-pause guard is a fact about the plan, not the draft).
- **The scanner is reached over HTTP, the archiver as a subprocess**
  (`SCANNER_URL`, `ARCHIVER_COMMAND`); unset, their applies are the
  maintainer's (pending) and their dry runs say they did not run. The
  contract tests mount the scanner's demo app (a dev-only dependency) and
  drive the scanner's own preset saver against this app. The configs-root
  switch is `DEPLOYMENT.md`.
- **The share follows the service, never origin**: it fast-forwards from
  the bare repository's accepted `main`, needs no credential, and a share
  that cannot move is shown, never forced.
- **The App's token never leaves the process**: git gets it through
  `GIT_CONFIG_*` in its environment, never argv; no route, page or commit
  carries it. Tests use a fake GitHub (an httpx mock transport), never the
  network.
- **The save follows the served form**, and writes `.inf`/`.nan` as
  themselves: documents are answered through `app._served` (non-finite
  floats as `"Infinity"`/`"NaN"`), never FastAPI's default encoding, which
  serves `null` and so would write it back.
- **Storage and tiers are GEECS-Schemas'.** The file list comes from
  `KIND_STORAGE` (`path_pattern()`, every suffix, a shared folder split by
  its `dispatch` loader); the tier is the model's `tier` and a stricter
  field's `tier` in its property schema. Never restate either here. This
  package's own column is `kinds.py`: each file kind's readers, apply verb
  and dry run, pinned to the file kinds by test.
- **The form is the theme's.** Pages draw with GeecsWebTheme's `form.js`
  from `/theme`; the service's `store.diff_documents` is the diff a save
  will trust, held to `form.js`'s by `tests/test_form_parity.py`.
- **A thin client.** It imports GEECS-Schemas, GEECS-Core (`configs_repo`,
  never `GeecsDb`) and the theme; everything else is asked over HTTP
  (the `configurator-edges` contract). Nothing in the repo imports it.
- **One uvicorn worker** (the commit lock is per process). CORS
  answers only the origins in `GEECS_CONFIGURATOR_CORS_ORIGINS`.
