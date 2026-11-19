# GeecsConfigurator — Developer Context for Claude

One editor over every GEECS-Schemas document in the configs repository
(`geecs-configurator`, port 8800): every file kind drawn from its schema,
validated, diffed and saved as its **tier** says, the save a commit by the
operator who made it; the consumers' own dry runs before a save and their
applies after it; setups captured, diffed and applied through the scanner;
an embedded editor any surface mounts. The arc's record (rulings, slices,
what the lab found) is its tracking issue, #1307; the derivations are in
git history. Deployment is `DEPLOYMENT.md`; the operator's view is
`docs/web_services/configurator.md`.

## Layout

```
geecs_configurator/
  __main__.py    # geecs-configurator: Settings.from_env, uvicorn on BIND:PORT
  settings.py    # Settings: every GEECS_CONFIGURATOR_* key (site.env)
  app.py         # the app factory: the API over the store, the save, delete,
                 #   blank and dry-run routes, the home and the log pages
  kinds.py       # the service's own column per file kind: readers, apply verb,
                 #   dry run, deletable, blank (storage and tier are GEECS-Schemas')
  store.py       # reads, validates, diffs; never writes, never creates a directory
  save.py        # a save, a create, a delete: tier, etag, ownership, the commit
  repository.py  # the bare repo, the live checkout, staging worktrees, the lock
  service.py     # CommitPath: the gate, the timer (fetch, inbox, apply passes)
  inbox.py       # the worker's writes, claimed by rename and committed
  apply.py       # the consumers: ScannerClient, PortalClient, Archiver;
                 #   ServiceApplier (dry runs, applies, the settle pass)
  validation.py  # the gate's rule (also the configs repo's CI): problems,
                 #   stem twins, referrers, deletable
  ownership.py   # what this box owns (experiments, analysis folders)
  blank.py       # a new document's draft and where it may go
  form_hints.py  # the layout hints an analysis kind's served schema carries
  setups.py      # the setups surface: capture, diff, apply (the scanner's)
  github.py      # the App's tokens and the pull requests
  yaml_edit.py   # the format-keeping splice of a save into the stored text
  log.py         # the commit log and its four states
  templates/     # home.html (catalogue, listing, the open document), log.html
  static/        # configurator.js (the page), embed.js (the mount contract,
                 #   the one save/copy/delete/create flow), setups.js
deploy/          # the unit, the configs repo's validate.yml and CODEOWNERS examples
```

## Tiers: what the Save button does

The tier is GEECS-Schemas' (`tier` on each document model, a stricter
field's `tier` in its property schema); the strictest tier among the changed
fields decides, a reordered list counting as changed.

| Tier | The save | Kinds (today) |
|---|---|---|
| operator | a commit to the configs repo's main, the operator its author ("Name via configurator"), this box the committer | trigger profiles, presets, scan variables, experiment defaults, setups, shot offsets, optimizer configs, screens, the home layout, the analysis recipes, jobs and groups, the archive policy's sampling overrides |
| maintainer | a pull request on the configs repo, the operator named in its body; the review is the approval and the merge the signature | operator policy, native devices, panel devices, derived channels, the archive policy's membership rules |
| code | read-only: a PR in GEECS-Plugins | family files (package data, shown from the package) |
| parameter, sudo | not built: an IOC record's tuning field (a later arc), an experiment up or down handed to the maintainer as a command | — |

There is no password. The operator's name is required on every write and is
the git author; a shared console login makes any local secret worthless.
The credential is a **GitHub App** installed on the configs repo alone
(contents and pull requests), its private key in a file beside site.env
(one key per box, revoked alone). The repo's ruleset says it: humans change
main by pull request; the App (and the maintainer's own account, for
regenerated catalogues) are on its bypass list, and that bypass is the
whole operator tier. A deploy key is the fallback: it pushes but opens no
pull request, so the maintainer tier shows the compare link instead.

## Setups: two executors, one document

A setup (`<experiment>/setups/<name>.yaml`, GEECS-Schemas' `Setup`) is
applied two ways, both reading the same files. The **queue path** is the
scanner's apply (`apply_setup` on the worker, one queue item, the hazard
confirmations asked as preflight questions, **no grant check**, as moves
and scans have none). The **Canopy module's executor** keeps its routes and
its gates (the operator policy's grants and hazards, the camera run lock)
for screen buttons and Squirrel's restore. Squirrel presets stay Canopy's:
setups do not use them. A setup `set` is the device's own completion; no
second confirmation layer is added.

## Rules that bind

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
  there (a run loads it by stem, across every facility folder; a save of
  such a document is refused before any commit, `validation.stem_twins`, so a
  save with no portal to ask never pushes what this gate refuses)
  (`validation.problems`, the one rule the configs repo's CI runs too, from
  `deploy/configs-repo/validate.yml`). This box never pushes to main what
  its gate would refuse: every commit, first try and each retry on a
  newer main, is judged by the same rule before its push
  (`CommitPath.judge`, the repository's `vet`; refused, nothing is pushed:
  a save's 422, or its 409 when main moved under it), so a check that
  depends on other files (a referrer, a stem twin) holds for the tree the
  push makes, not only the tree the save checked. A create refuses any
  file already at its path, whatever kind the shared folder reads it as
  (`DocumentStore.occupied`). A refused commit is fetched, logged
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
  and sent it (a crash exits 1 like drift); its refusals no retry clears,
  the mass-pause guard and the unresolved-registry guard ("nothing was
  sent"), are pending, the maintainer's by hand (`Archiver.onboard`). Each experiment has its own scanner
  (`SCANNER_URL` as `Exp=url,…`); a row of an experiment without one is
  never applied by another's reopen. A save while a pass runs is answered
  503 at once, never kept waiting on the lock.
- **Dry runs are the consumers' own** (`apply.py`): the scanner's
  `POST /api/preflight/trigger-profile`, `geecs-archiver onboard
  --dry-run --policy <draft>` over a temporary file, and for the analysis
  kinds (recipes, `.himg` and scalar-columns jobs, groups) the portal's
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
  will trust, held to `form.js`'s by `tests/test_form_parity.py`. How an
  analysis kind is laid out is `form_hints.py`, stamped into the schema
  `GET /api/schema/{kind}` serves (portal or fallback alike), so the page
  and every embedded drawer draw it the same: a recipe's `x-sections`
  (Source, Steps, Measure, Figure, Summaries, Scan) and `x-ndim-from` (the
  input kinds' own `ndim`), a group's entries' `x-suggest` (the analyzers
  tree's stems), and on the fallback the `x-form-note` saying why a step is
  a name there. Hints say how to draw, never what is valid. The page shows
  the YAML of the form as edited beside it (`/api/yaml`, the latest ask
  wins). Pinned by `tests/test_contract_portal.py` and the laid-out
  round trip in `test_form_corpus.py`.
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
  page off the kit gets the drawer in a `.kit` wrapper of its own). What
  the drawer shows answers the draft in it: the latest Preview's picture is
  shown whatever order they resolve in, an edit marks a shown preview stale,
  and a dry run answered for an earlier draft asks again rather than letting
  its Save save the edit (`tests/test_embed_stale_js.py`).
- **Delete and copy are the save's.** A delete (`DELETE /api/documents/...`,
  `save.delete`) is a `git rm` through the same repository mechanics: the
  etag served, the operator the author, the tier of everything it removes
  (a commit, or a proposal), the facility 403, a `deleted` row on `/log`.
  Only a document of a folder of named files is deleted (`validation.deletable`, served as the catalogue's `deletable`);
  a kind's one file (the archive policy, `home.yaml`) is a 405. It is
  refused while any document names it (`validation.referrers`, over
  GEECS-Schemas' `references` table: a group its recipe, an optimize preset its
  optimizer config and that its diagnostics, a preset or the
  defaults a trigger profile, a setup or a screen button a setup, the home
  layout a screen), referrers listed, no force; `POST /api/delete-check`
  is the same answer asked first. The gate refuses a delete made around
  the configurator the same way (`validation.problems(..., deleted)`, the
  live gate's deleted owned paths and the configs repo's CI `--deleted`; a
  rename is a delete and an add there, `--no-renames`).
  In the browser `embed.js`'s `deleteFlow` asks the check, then the kit's
  `<dialog>` whose Delete waits for the name typed; `copyFlow` saves the
  form as a new document beside it (the save route's `create`: its dry run,
  tier and stem rule), the stem-named `name` renamed and a recipe's
  `output_name` set to the new name. Both the drawer ("Copy or delete") and
  this service's page use them. A reference is compared as its reader
  resolves it (`geecs_schemas.READ_AS`: an optimizer config through the
  worker's resolver's `yaml_stem`, every other name as its literal stem).
  The drawer is shared, so an answer that arrives after another document
  was opened in it (a delete, a copy, the load) touches the drawer only
  while its own document is still the one open; the host's `onSaved`
  hears it either way. Pinned by `tests/test_delete.py`,
  `test_document_actions_js.py` and `test_embed_switch_js.py`.
- **A write's answer never takes an edit made after the write was sent.**
  The form stays editable while a save, convert-and-save, create, copy or
  delete is on its way, so every completion compares the form with what it
  sent: a later edit stays in the form, the drawer says it is unsaved, and
  the host hears it (`onSaved(answer, id, {unsaved})`; the configurator's
  page does not navigate then). A create edited since becomes the file's
  form in place (its next Save a save over it); a copy keeps the drawer on
  the source; a delete keeps the form as a new document at the same id
  (Save creates it again). Pinned per write by
  `tests/test_embed_write_edits_js.py` and `test_new_document_js.py`.
- **A new document starts from its consumer's blank.** A kind the
  catalogue marks `blank` (`kinds.BLANKS`: recipes, groups and scalar-columns jobs; a `.himg`
  job and every other kind are copied, never started blank) is created from
  a draft: `GET /api/blank/{kind}/{folder}/{name}.yaml` refuses first (not
  a stem 422, a folder this box does not own 403, a file there or, in the
  analyzers tree, another file's stem 409, no commit path 405), then relays
  the portal's `/api/analysis/blank` (a recipe that already binds: the
  registry's default measure for the input's frames), the portal's 422 its
  own; with the portal unset or silent the schema's fewest fields (no
  measure chosen), `source` saying why. `GET /api/folders/{kind}` lists
  where: this box's facility folders and the owned subfolders that hold the
  kind. Nothing is written until Save, which is the save route's create
  (dry run, tier, stem rule, ownership). In the browser
  `GeecsConfigurator.create` draws the start form in the drawer (name,
  folder, and for a recipe its data folder and frame kind with what each
  kind requires, read off the schema: `startChoices`) and opens the draft
  with `open({..., draft})`: no read at the id, the first Save a create, the
  file opened as stored once committed. Entry points: the page's "New
  recipe…" / "New group…" and the drawer's "New, copy or delete". Pinned
  by `tests/test_new_document.py`, `test_contract_portal.py` (every frame
  kind's draft binds and commits; a taken stem refused) and
  `test_new_document_js.py`.
- **One save flow.** `embed.js`'s `saveFlow` is the save in the browser
  (an existing document, a new one's first Save as its create): the embedded drawer and this service's own page
  (`configurator.js`, which loads `embed.js` first) both save through it,
  each with its own elements and document state. A change to what a save
  asks, sends or shows is made there once. (Saving a captured setup as a new
  document is the setups surface's own `PUT` with `create`, `setups.js`.)

## Tests

Its own env (`poetry install` here, the dev group: the scanner's demo app
and the portal with its `analysis` extra are dev-only path dependencies for
the contract tests): `poetry run pytest -q`, or `scripts/check.sh
GeecsConfigurator`. Hermetic: scratch bare repositories as origin, a fake
GitHub on an httpx mock transport, the scanner's and portal's apps on ASGI
transports; the browser files run under node against this app's own
answers (`test_*_js.py`), held answers and edits included. No test reaches
the network or a device.

## Out of scope

Per-user authentication and the operator registry (#882), and any grant
check on the queue path; one shared operator key across surfaces (it needs
a prefix-preserving front door); Squirrel; DB writes; the parameter tier's
IOC restarts; a non-web editor.
