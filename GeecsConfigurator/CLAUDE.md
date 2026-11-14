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
  documents changed since the live commit must validate, and a changed
  document of the analyzers tree must not share its stem with another file
  there (a run loads it by stem, across every facility folder; a save that
  creates one is refused before any commit, `validation.stem_twins`, so a
  save with no portal to ask never pushes what this gate refuses)
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
  `POST /api/preflight/trigger-profile`, `geecs-archiver onboard
  --dry-run --policy <draft>` over a temporary file, and for the analysis
  kinds (recipes, `.himg` jobs, groups) the portal's
  `POST /api/analysis/validate/{kind}` (`PortalClient`, `_PORTAL_URL`: the
  recipe bound to the analysis core's registry, a stem the loader would
  refuse anywhere in the shared analyzers tree, a group's references; its
  errors come back at their fields and the page draws them there), never
  re-implemented here. An analysis kind's form is the portal's schema too
  (`GET /api/schema/{kind}` relays `/api/analysis/schema`: a recipe's steps
  and measure are the registry's), the document's own schema when the
  portal is unset or silent, `X-Schema-Source` saying which. The contract
  tests mount the portal's app (a dev-only dependency, its `analysis`
  extra: `tests/test_contract_portal.py`). A consumer's refusal is the save's 422 (before any git); a check
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
- **A save changes only the text of what changed.** The configs repo is
  hand-written and reviewed as diffs, so a save never re-dumps a file: it
  splices the draft into the stored text (`yaml_edit.py`, over PyYAML's
  node marks: the resolver the store reads with), keeping every untouched
  byte, and checks the text reads back as the draft before the commit.
  Only a new document, a legacy-dialect conversion and a change the splice
  cannot place exactly are written whole (the last with a note). Pinned by
  `tests/test_yaml_edit.py` and `test_save.py`.
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
- **Setups on the surface are the scanner's** (`setups.py`, the page's
  `static/setups.js`). Capture reads (`POST /api/setups/capture`: one
  `GET /api/readback` per variable through the experiment's scanner, a
  reading the scanner flags `not_numeric` read again as a word; one that did
  not answer stays unread) and answers a draft;
  nothing is written to a device, and the draft is saved through the one
  save route (`PUT` with `create`), so its author, tier and ownership are
  the save's. The diff and the apply are relayed status and body as the
  scanner answered; the apply's `acknowledged` goes on as the page sent it
  and this service never answers a question; an experiment this box does
  not own (`setups.applies_for`: a box naming its experiments applies only
  those, commit path or not) is a 403 before the scanner is asked. The page applies the
  setup as committed: Apply waits while the form holds an edit. Pinned by
  `tests/test_setups_surface.py` (a recording fake scanner and the
  scanner's demo app).
- **A save on the page asks the kind's dry run first** (`/api/dry-run`):
  a refusal stops it there; what the consumer said, or why it could not
  run, is shown with a Save to confirm; a kind with no check saves at
  once. The save route runs the dry run again: the page's is for the
  operator to see.
- **The embedded editor's mount contract is `static/embed.js`**
  (`GeecsConfigurator.open({api, kind, experiment, id, fields, operator,
  preview, onSaved})`, its docstring the contract): a host loads its own
  `/theme/kit.js` and `/theme/form.js` and this script from the
  configurator, never the package, and its origin goes in `CORS_ORIGINS`.
  The operator is the host's, asked on every save, stored nowhere here;
  `fields` limits what is drawn and the rest is saved as stored; the save
  is the page's (dry run, etag, 409/422/428). Pinned under node against
  this app's own answers (`tests/test_embed_js.py`). Hosts: the scanner's
  preset mini; the portal's Analysis tab (its `preview` a promise of the
  unsaved recipe drawn on the scan, its `toolbar` the shot and the name; a
  page off the kit gets the drawer in a `.kit` wrapper of its own).
- **One save flow.** `embed.js`'s `saveFlow` is the save of an existing
  document in the browser: the embedded drawer and this service's own page
  (`configurator.js`, which loads `embed.js` first) both save through it,
  each with its own elements and document state. A change to what a save
  asks, sends or shows is made there once. (Saving a captured setup as a new
  document is the setups surface's own `PUT` with `create`, `setups.js`.)
