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
