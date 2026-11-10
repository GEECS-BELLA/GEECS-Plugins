# GeecsConfigurator — Developer Context for Claude

One editor over every GEECS-Schemas document in the configs repository
(`geecs-configurator`, port 8800). A stub while the configurator arc runs:
its design is `Planning/configurator.md`, its record the tracking issue
(#1307); the arc's last slice writes this file fully.

## Rules that already bind

- **Read-only until the commit path lands.** No route writes; the store
  (`store.py`) reads, validates and diffs, and never creates a directory.
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
- **One uvicorn worker** (the commit lock to come is per process). CORS
  answers only the origins in `GEECS_CONFIGURATOR_CORS_ORIGINS`.
