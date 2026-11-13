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
- **One configs repo, many facilities: a box owns its experiments.**
  `ownership.py` is the rule. A box commits, judges and logs only under the
  experiments in `GEECS_CONFIGURATOR_EXPERIMENTS` and its facility folders in
  the shared analysis tree (`_ANALYSIS_FOLDERS`). A commit path that owns
  nothing refuses to start. A save, proposal or inbox publish elsewhere, or
  through a link out of an owned folder, is a 403, and the repository refuses
  it again at the commit. The gate judges owned documents only, so another
  facility's invalid commit fast-forwards and never blocks this box. Another
  facility's commits get no log row and never reach the apply seam. Every
  experiment stays browsable; foreign documents are served `read_only`. The
  deploy (the App, once per org; a key and the keys per box) is
  `DEPLOYMENT.md`. Pinned by `tests/test_facilities.py`.
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
  seam's answer; `NoApplier` never says applied (slice 3 implements it).
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
