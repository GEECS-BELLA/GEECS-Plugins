# The configurator — the brief (integration branch `feat/configurator`)

Scaffolding per `Planning/README.md`: the promotion PR to `master` deletes this
file after moving anything load-bearing into the owning `CLAUDE.md` files. The
coordinator runs the slices in order, one PR each into this branch, with the
`/land` ritual, and merges a lane PR after its review is dispositioned.
`master` is merged forward at least weekly. Promotion to `master` is the
maintainer's merge.

**The arc's record is its tracking issue** (CONTRIBUTING § "GitHub conduct"),
opened on the maintainer's word when slice 0 starts: its body is the current
plan, one comment per slice records what was planned, what review changed,
what the lab found and what the next slice needs. This file is the design
while the work is live; the dispositions below predate the issue and are
also on #1301, and every later one goes on the issue, not here.

**Exit checklist.** This file is deleted in the PR that completes or abandons
the work it describes, and a promotion alone does not mean completion. Before
the deletion: every invariant the code must keep goes into the docstring of
the code that keeps it; the package rules into `GeecsConfigurator/CLAUDE.md`
and the cross-package ones (the clones, the apply verbs, the tiers' home)
into the root `CLAUDE.md`; operator and deployment guidance into `docs/` and
the runbooks; outstanding work onto the tracking issue; the rationale stays
in git history.

## Goal (the maintainer's framing, 2026-11-06)

Operators have no path for a configuration change except an agent or a
terminal: the week's example was the archive deadband on ten supplies, changed
by hand on their IOC records. The maintainer wants **one editor over every
GEECS-Schemas document**, generated from the schema so every kind is edited in
one place, with the **tier of each document, or of a field inside it, declared
on the schema** and deciding what the save button does: a direct commit to
main, a pull request for the maintainer, or a command handed to him. The same
editor, embedded, serves any surface that wants a specific editor (the portal's
analysis recipes, the scanner's presets, a device page's parameters). Beside
it, **setups**: a named device state, optionally sequenced ("low power",
"high power"), captured from live readbacks, diffed against them, applied as
one queue item on the worker. Setups replace the carried-over action plans,
which the maintainer is fine throwing away. Storage is the configs repo, not
the GEECS DB (the maintainer's lean; ruling 3 below makes it explicit).

A clickable mockup (fake data, nothing commits) was reviewed by the maintainer
on 2026-11-06: "a bit busy, maybe a bit ugly", the idea "super nice". The look
is a later pass; this brief is the mechanics.

## What exists today (the survey and the red team, 2026-11-06)

- **A generic editor already half-exists.** `scan_analysis.config_editor`
  (mounted by the portal at `/configs` behind `--config-editor`) serves
  list / schema / validate / get / put-with-etag / delete / preview over
  `scan_analysis.config_store.ConfigStore`: atomic temp-and-rename writes,
  etag conflicts (a 409 with the current document), pydantic errors at field
  locations, `pending_changes()` as a read-only `git status`. It knows two
  kinds (analyzer, group), binds them to GEECS-Analysis's registry
  (`compile_recipe`, `recipe_schema`), writes into the **share's checkout
  uncommitted**, and commits nothing: its page says "saves write the configs
  tree; commit them yourself". Its
  `editor.js` is lazy-loaded into a drawer on the portal's run page, which is
  the embedded-editor precedent.
- **The catalogue exists, with non-file members.** `geecs_schemas.SCHEMA_REGISTRY`
  maps 22 kind names to document models, but six are not files in the configs
  repo: `journal_entry` and `log_entry` are database rows, `scan_request` is
  a plan parameter, `action_plan` a sub-document, `himg_job` a job, and
  `family_file` is package data read from the package's resources.
  `schema_export` emits JSON Schema for four kinds (and three sub-models);
  `docgen` renders every field description, both under no-drift tests. Field
  descriptions were written to double as GUI help. No model declares a tier.
  The folder constants for six kinds (trigger profiles, scan variables,
  presets, optimizer configs, experiment defaults, shot offsets) live in
  GeecsBluesky's `config_resolver`, not in GEECS-Schemas.
- **Three more writers already write the share uncommitted**: the scanner's
  `save_preset` (through `ConfigsRepoResolver.write_preset`), the worker's
  `write_shot_offsets` (inside `measure_shot_offsets`, read back at its next
  environment open), and the portal editor above. The scanner's scan-log list
  was deliberately kept out of the repo because an often-edited file keeps
  the checkout dirty and, once committed, blocks `git pull` on the host. That
  is the problem a commit path solves.
- **The configs root is the share's clone**: `GEECS_CONFIGS_ROOT` in site.env,
  rendered into `config.ini` as `scanner_config_root_path` (only when the file
  is absent), resolved by `geecs_core.configs_repo.scanner_configs_base()`;
  Canopy mounts the same root read-only into its container and the Control
  Room reads it through its own key. It is a Windows-authored CRLF checkout
  that LabVIEW also reads, with spaces in its path; the fleet map says never
  to fix line endings in place. The repo is private. No service holds a git
  credential of any kind; the service account clones over public HTTPS and
  the bootstrap never moves a checkout's HEAD.
- **Pickup differs by kind.** The worker reads trigger profiles, the scan
  variable catalogue, native devices and shot offsets at environment open,
  and a reopen is the only reload: tens of seconds of every device
  reconnecting (`geecs-qserver-ready` does one after every manager start; the
  queue client exposes no open or close verb today). The action library is
  re-read per call, experiment defaults per plan, optimizer configs per
  request; the scanner's resolver re-reads the scan-variable catalogue on
  mtime change, but the worker keeps the catalogue it read at environment
  open. Canopy reads the operator policy and
  the action library per request with a short cache. The scanner reads
  presets per submit; its preflight resolves a trigger profile **by name from
  disk**, while the preset itself travels inline. The archiver reads its
  policy when `onboard` runs; `onboard --dry-run` prints the plan and sends
  nothing, and `--policy PATH` names a file to read instead of the
  experiment's. The CA gateway loads the derived-channel file at startup and
  builds its PVs then. The Control Room reads screens and the home layout
  per request. The device registry (`native_devices.yaml` with
  `panel_devices.yaml`) has more readers than the worker: Canopy's directory
  adapter reads both files on every directory call (Raccoon caches its
  answers; how a change reaches that cache is not yet surveyed), the
  archiver's registry pass reads them when `onboard` runs, and the Control
  Room's device pages, home and screens see them through Canopy's module.
  (Corrected on #1312's review; the configurator's `kinds.py` carries the
  table.)
- **Action plans have two executors with different gates.** The Canopy
  module's executor (`POST /v1/actions/{name}/run`, polled at
  `/v1/actions/runs/{id}`) pre-checks every `set` against the operator
  policy's grants and hazards and the camera run lock, and waits each put;
  Squirrel's restore runs through that same executor. The worker's
  `run_action` compiles set / wait / check / run through `bps.abs_set`,
  `bps.sleep` and `bps.rd` with exact-match `values_match` and applies none
  of those gates: nothing in GeecsBluesky or the scanner reads the operator
  policy. Deployed screens bind no action today.
- **Readback confirmation exists three times.** `CaConfirmSettable` sets and
  polls a confirm readback (floats by tolerance, strings by equality, timeout,
  settle); `CaMotor.set()` resolves when the readback is within tolerance,
  with the stall rule; `NativeSetpoint` writes the setpoint record and is done
  when the readback is within tolerance. Only the plain `CaSettable` waits for
  the gateway put, the device's own reply, plus a settle time. So
  `abs_set(wait=True)` already *is* readback confirmation wherever a device
  has a readback to confirm by.
- **Canopy's module carries Squirrel presets** (`/v1/presets`, golden,
  restore): a second named-device-state mechanism, stored in Canopy's
  database, applied through the module's executor, unsequenced.
- **Operator identity is free text in localStorage under four different
  keys**, and localStorage is per origin: every surface has its own port and
  therefore its own store, so no key is shared without a front door. The
  queue manager's `user` is one per-process identity; the person rides in
  `md["geecs"]["operator"]` on preset submits only. `submit_plan` takes no
  `md`, and GEECS's `mv` is a wrap of `bps.mv`, which forwards keyword
  arguments to each device's `set()`, so an `md` there would reach the device.
- **#883 (2026-09-13) recorded a direction, not scheduled**: controls
  configs to the GEECS DB, analysis configs staying YAML, with the one-PR
  consequence that the arc then under way added no new YAML config kind.
  Since then the repo gained operator policy, screens, home layout, panel
  devices and the archive policy as YAML kinds. The direction is dead in
  practice and nothing in the tree cites it.

## The design

### One service, a client of the others

A new package **`GeecsConfigurator`** (`geecs-configurator`, port 8800; 8000
to 8700 are taken), FastAPI and Jinja on the kit, plain JS, no build step, one
uvicorn worker (the commit lock is per process). It imports **GEECS-Schemas**
(the registry, the models, the tier annotations, the folder constants),
**GEECS-Core** (`configs_repo` for the root and the per-experiment paths; never
`GeecsDb`) and **GeecsWebTheme** (web extra). Everything else it needs it asks
over HTTP, the way GEECS-MCP does: the scanner for preflights, readbacks,
settables, the environment reopen and applies; the portal for the analysis
kinds' validation; the archiver's plan by its CLI (below). Its import contract
forbids GeecsBluesky, GEECS-Analysis, the scanner, the portal, Canopy and the
Control Room; its contract tests mount the scanner's demo app and the portal's
app on an ASGI transport, as GEECS-MCP's do. Nothing in the repo imports it.
Its API answers CORS for the surfaces' origins, because the embedded editor
runs in the host page's origin.

**The form renderer is the theme's.** The schema-driven form (`form.js`: a
JSON Schema with tier annotations in, kit controls out, a document and its
diff back) lives in GeecsWebTheme beside `plot.js`, the one copy every surface
draws with, loaded from each surface's own `/theme` mount; the configurator's
home page is one more host of it. The service owns the API: documents, tiers,
dry runs, commits, applies, the log.

### The document model

The registry is the catalogue, and the service's table is the **subset that
is a file in the configs repo**, with an explicit `storage` column (`file`,
or `not-a-file` with the reason) so a registered kind is never silently
missing. Per file kind the table knows the **folder and file pattern** under
`<experiment>/` (slice 0 moves the six resolver constants into GEECS-Schemas
beside `ACTION_LIBRARY_FOLDER`, and the resolver imports them, so the schema
package is the one home and the configurator never imports GeecsBluesky), the
**tier**, the **consumers** with each one's apply verb, and the **dry-run
plug-in** (an HTTP call to the consumer, a CLI, or none). `GET
/api/schema/{kind}` serves `model_json_schema()` with the tier annotations in
it, which is what the form draws from; `schema_export` grows to every file
kind so OSPREY and the docs see the same schema. The schema package's own
tests pin that every file kind has a tier and a folder and every non-file
kind is listed as such. Family files are not a configs-repo document: the
editor shows them read-only from the schema package under the label `code`,
outside the file table and its test.

**Tiers are declared in GEECS-Schemas**, nowhere else: on each document
model a `tier: ClassVar[Tier]` (`operator` unless said otherwise) **and** the
same value in `model_config`'s `json_schema_extra`, so it reaches
`model_json_schema()` (a bare class attribute neither passes pydantic nor
appears in the schema); and `json_schema_extra={"tier": "maintainer"}` on a
field that is stricter than its document. The editor marks the field; the
strictest tier among the *changed*
fields decides the action, where "changed" is any difference at the field's
path, a reordered list included. `docgen` prints the tier beside each kind.

| Tier | What the button does | The gate | Kinds |
|---|---|---|---|
| operator | commits to the configs repo's main | the schema validates; the dry run passes; the base is current | trigger profiles, presets, scan variables, experiment defaults, setups, archive sampling overrides, screens, home layout, shot offsets, optimizer configs, analysis recipes and groups |
| maintainer | opens a pull request on the configs repo | the maintainer's review | operator policy (grants, hazards), native devices, panel devices, derived channels, the archive policy's include and exclude rules |
| parameter | commits to geecs-epics | the record exists; the value is in range | a tuning field on an IOC record (ADEL, MDEL, a trigger delay, a ROI pin) — **a later arc**, see rulings |
| sudo | prepares the hub verb's exact command and records the hand-off | the maintainer runs it | an experiment up or down |
| code | read-only | a PR in GEECS-Plugins | family files (package data, shown from the package; not a configs-repo file) |

There is **no password**. Identity is the operator's name, required and
refused when blank, as on every write today. The maintainer tier is a PR
because the review is the approval and the merge is the signature; a shared
console login makes any local secret worthless. If PRs prove heavy for small
grants, a maintainer token held in the service's environment can turn "open a
PR" into "approve here", still logged, still one person. Not built now.

### The store and the commit path

The store is `ConfigStore` generalized: every file kind, the same etag and
atomic-write discipline, pydantic errors at field locations. It lives in the
new package; the original is deleted when the analysis kinds move (slice 6).

**The service owns a bare repository and two kinds of checkout** under its
state directory on the services box. The **live checkout** is the Linux
services' configs root (ruling 4), world-readable, and holds **accepted
commits of main only**: the service never writes a draft there and never
checks a branch out there, so a consumer that reads a file per request
(the device module reads the operator policy on every put) can never see a
change before it is accepted. Every edit is staged in a **throwaway
worktree** of its own: there the service fetches and fast-forwards, writes
the draft, validates, runs the kind's dry run, commits with **the operator
as the git author** ("Name via configurator") and the service as the
committer, and pushes, with a bounded retry on a concurrent push; on success
the live checkout fast-forwards to the new main and the worktree is removed;
on a failed push (a non-fast-forward, the App unreachable) the worktree is
removed, nothing reaches the live checkout, and the operator sees the error
with their edit kept in the page. Maintainer tier: the same in its worktree,
on a branch pushed to origin, then a PR opened with the operator named in
the body; the branch never exists in the live checkout. **Before serving a
document** the service fetches and fast-forwards the live checkout under the
one lock, so the etag the editor holds is of the current base. **On save**,
if the base moved under this document the answer is the store's existing 409
with the current document and the diff, and the operator re-applies their
edit — **never an automatic merge** of a whole-document PUT. The log of what
was committed and whether it applied is the service's own table, served on a
page, with four states a row can be in: *applied*, *queued* (until the
worker is idle), *pending* (handed to the maintainer, named), *nothing to
do*; a commit that landed is never written up as applied by that fact alone.

**The credential is a GitHub App** on the org, installed on the configs repo
alone with contents and pull-requests write; its private key lives in a
secrets file beside site.env, never in a repo (Canopy's `secrets.env` is the
pattern). The service mints installation tokens as it needs them. **The
repo's ruleset says it plainly**: humans change main by pull request; the App
is on the bypass list, and that bypass is the whole operator tier. A deploy
key is the fallback: it pushes but cannot open PRs, so the maintainer tier
would show the compare link instead. Nobody types a GitHub credential on a
console.

**After a push the share's clone is fast-forwarded** by the service account
**from the service's clone** (added as a local remote, so the share needs no
credential for the private repo), which is the pull the maintainer does by
hand today, so LabVIEW keeps reading the file it always read. Two
consequences the deploy settles (ruling 4): the Linux services' configs root
becomes the service's clone, so they read a clean checkout the moment it is
pushed; and the share's clone becomes **read-only for humans and services**,
because an uncommitted edit on it blocks the fast-forward. The writers above
route through the service (presets in slice 3, the analysis kinds in
slice 6). **Nothing writes into the live checkout**, because a dirty
checkout cannot be fast-forwarded: git refuses to overwrite an uncommitted
file, identical bytes included (the maintainer reproduced it). The worker's
one write, `measure_shot_offsets` saving `shot_offsets.yaml`, therefore
lands elsewhere: the resolver gains a **write root** (site.env
`GEECS_CONFIGS_WRITE_ROOT`, defaulting to the read root so a laptop behaves
as today), and on the box it is the service's **inbox** directory. The
service **reconciles** the inbox on a timer of seconds and before any fetch:
it copies the file into a staging worktree on top of main, commits it
authored as the worker, pushes, fast-forwards the clean live checkout, and
removes the inbox file; the worker reads the committed offsets at its next
environment open, which is the apply the table already lists for that kind.
If main moved the same file meanwhile, the measurement wins and the log
says so. An operator's edit of shot offsets is an ordinary operator-tier
commit; last wins. The edits already sitting on the share are committed
once, by hand, before the switch.

**A merge on GitHub is an event too.** The service fetches main on a timer
(a webhook later): a commit on main that did not come from the service, a
merged maintainer-tier PR above all, fast-forwards the service's clone, then
the share's, then runs the kind's apply and logs it, so a merged grant or
registry change takes effect without anyone opening the editor.

### Apply, per kind

An apply is what makes a committed document take effect. The service knows
the verb per consumer and runs it after the push, or queues it:

- **Read per request** (screens, home layout, operator policy, setups,
  presets, optimizer configs, experiment defaults, analysis documents):
  nothing to do.
- **Trigger profiles, scan variables, native devices, shot offsets**: the
  worker environment reopens (the scanner's own picker sees a scan-variable
  change on mtime, the worker only at open). The queue client grows
  `environment_close` and `environment_open`;
  the scanner grows `POST /api/environment/reopen`, which holds the service
  lock through close and open, refuses unless the environment is idle **and**
  the queue is empty (an environment that is merely closed is not idle), and
  after the open starts the queue if items wait; the status chip reads
  "reopening". The configurator calls it, or reports "queued until idle" and
  retries.
- **Archive policy**: `geecs-archiver onboard` from the archiver's checkout;
  the dry run is `onboard --dry-run --policy <draft>` over a temp file, so the
  unsaved edit is what is planned. In v1 the service runs the installed CLI as
  a subprocess in the archiver's venv on the box (the archiver has no HTTP
  API and gets none for this); the consequence is that the configurator is
  tied to the host that has the archiver installed, said in its deploy notes.
- **Derived channels**: the CA gateway builds their PVs at startup, so the
  apply is a gateway restart, which drops every client's PVs for seconds
  and is never done during a scan. The service does not restart units: the
  row is *pending*, named to the maintainer with the unit to restart, until
  he says it was done. The same word, *pending*, covers every apply that is
  a restart or a hand-off.
- **Parameter tier**: an IOC restart on another host; not built in v1.

The dry runs are the consumers' own answers, never re-implemented, and they
must see the **unsaved** document: the scanner's preflight takes the preset
inline already and gains an optional inline trigger-profile document (slice
3); the portal exposes `validate` and `schema` for the analysis kinds, bound
to GEECS-Analysis's registry as its editor's are today (slice 6); the
archiver's plan reads the draft by `--policy`; the scanner's settables list
checks a setup's targets. Screens have no dry run in v1: the Control Room
checks a screen's bindings against the roster when it renders, and a
draft-screen check is a later route. A kind with no consumer check has no
dry run and says so.

### Setups

A **`Setup`** document kind (working name; ruling 1) in GEECS-Schemas:
`name`, `description`, ordered `steps`, and `captured` (by whom, when, from
which readbacks). Steps: `set` (device, variable, value), `wait` (seconds),
`check` (device, variable, expected; a number by tolerance, a string by
equality, as `CaConfirmSettable` matches today), `apply` (another setup,
inlined, cycles refused). Files at `<experiment>/setups/<slug>.yaml`, one per
setup, so a commit touches one setup. It is the action plan with a capture
record and one file per plan; **`ActionPlanLibrary` and
`action_library/actions.yaml` are deleted** by a one-shot converter that runs
once per experiment on the configs repo and is deleted after.

**A `set` is the device's own completion**: `abs_set(wait=True)` on the
namespace's settable, which already confirms by readback on a motor, a native
setpoint and a confirm settable, and waits for the device's reply on a plain
settable. The setup adds **no second confirmation layer** (the device's
verdict is the truth, the faithful-to-the-reply rule); a plain settable that
should be confirmed by a readback is declared a confirm settable in the
namespace, where that already lives, not in the setup.

**Two executors, one document** (ruling 2, amended by the red team). The
Canopy module's executor **stays**, with its routes and its gates (grants,
hazards, the run lock), for screen buttons and Squirrel's restore, and reads
`setups/` where it read the action library. The worker gets `apply_setup(name,
*, md)`, compiled as `run_action` is today (`run_action` and the library go);
`plan_names`, `bind_plans` and the permissions regex each gain it. The
scanner is the queue's entry: `GET /api/setups`, `POST /api/setups/{name}/apply`
(idle-only, operator required, the diff acknowledged like a preflight, and
**the hazard confirmations of the policy and the family files asked as
preflight questions**, which the scanner can read because it imports
GEECS-Schemas), `GET /api/setups/{name}/diff` (readbacks via the existing
`/api/readback` path). The queue path checks no grants, exactly as `move` and
scans check none today; the brief says so rather than pretending otherwise,
and the grants question goes to the operator registry (#882). GEECS-MCP gets
`apply_setup` for free as a thin client. A screen `button` that names an
action keeps calling the module. `apply_setup` takes `md` the way `count` and
`sweep` do; the operator attribution of `mv` is a separate fix (found along
the way).

The configurator's setups page is the editor plus three verbs: **capture**
(choose devices; the service reads their readbacks through the scanner and
writes set steps), **diff** (the scanner's diff route, drawn as a table with
"will change" per row), **apply** (the scanner's apply route, after the
acknowledgement). **Squirrel presets stay out of this** (ruling 5).

### The embedded editor

A host page draws the editor with the theme's `form.js` and one mount
contract: `GeecsConfigurator.open({api, kind, experiment, id, fields,
operator, preview})` draws it in a kit drawer, limited to `fields` when given,
with the **operator passed in by the host** (the host's own identity store,
since localStorage is per origin) and the host's `preview` callback wired to
the Preview button; the API base is the service's URL from the host's site
value. Hosts: the portal's run page (analysis recipes and groups, with its
`render_document_as_run` preview, slice 6), the scanner's preset panel
(slice 5), the Control Room's launcher slot
(`GEECS_CONTROL_ROOM_CONFIG_EDITOR_URL` points at the service's home) and,
later, a device page's parameters. The host never imports the package; the
contract is pinned by a test that runs `form.js` under node against a stub.
The scanner's own read routes for presets, profiles, variables and optimizer
configs stay, for its own page.

### Identity

The operator is the host's: each surface keeps its own store and passes the
name on every call, as today. The one-key unification (`geecs.operator`) is
possible only behind a prefix-preserving front door and is not this arc's
work. The operator registry as a DB table (#882) replaces free text later and
this design does not wait for it.

### Deployment

A unit template with `@PLACEHOLDER@`s, `TEMPLATES` in `render_units.sh`, the
bootstrap's tables and template case, site.env keys
(`GEECS_CONFIGURATOR_PORT`, `_CLONE_DIR`, `_SHARE_CLONE` (the path to
fast-forward), `_SCANNER_URL`, `_PORTAL_URL`, `_ARCHIVER_CHECKOUT`, the App's
id, installation id and key file), a fleet-map row, a Gatus row in
geecs-epics' host vars, the Control Room's launcher URL. The configs-root
switch (ruling 4) is a site.env change **and** a re-render of Canopy's compose
(it mounts the root read-only) **and** a hand edit of the rendered
`config.ini` on the box; the runbook step says all three. No facility value
in code.

## Rulings needed

Rulings 1 and 3 before slice 0 (it bakes in the name and the store); the
rest before slice 2. Nothing is built before 1 and 3.

1. **The word.** "Setup" is the working name; "preset" is taken twice (scan
   presets, DB presets).
2. **Two executors, one document.** The module keeps its executor and gates
   for screen buttons and Squirrel restore; the worker's `apply_setup` is the
   queue path through the scanner, with hazard confirmations asked and no
   grant check, as moves and scans have none today. (The red team's P1:
   deleting the module route would drop the gates and break restore.)
3. **The configs repo over the DB** for every operator-tier document, setups
   included. This reverses the direction #883 recorded (controls configs to
   the DB, no new YAML kind), which five kinds have already passed; close or
   retarget #883 when ruled.
4. **The clones.** Linux services read the service's live checkout, which
   holds accepted main commits only; the share's clone is fast-forwarded
   after each push from the service's repository and is read-only for
   humans and services; the uncommitted edits on it are committed once by
   hand first; the portal's editor is read-only from the switch until it
   moves onto the service; the worker's write root is the service's inbox,
   so the live checkout is never dirty.
5. **Squirrel.** Setups do not use Canopy's presets; the module's preset
   routes stay as they are, on the module's executor. Two mechanisms for one
   thing is the thing to avoid later, not now.
6. **The credential and the ruleset.** A GitHub App (the maintainer creates it
   once) on the configs repo's bypass list; humans change main by PR.
7. **The parameter tier** (geecs-epics records) is a follow-on arc, not v1.

## Slices

0. **Schemas.** `tier` on every file kind's model and the stricter fields; the
   `Setup` kind; the six folder constants moved from the resolver into
   GEECS-Schemas (the resolver imports them); the registry's file / not-a-file
   table; `schema_export` over every file kind; `docgen` prints tiers. The
   schema's own tests pin the table.
1. **The form and the service, read-only.** `form.js` in GeecsWebTheme under
   node tests (the mockup's renderer rewritten as a rules module); the
   package, unit template, site.env keys, the import contract, the store
   generalized (copied, not shared), the routes (list, schema, get, validate,
   diff, yaml), the tier and consumer panels, the home. Deployed on the box
   behind the Control Room's launcher as a read-only browser of every
   document. Screenshots light and dark.
2. **The commit path.** The service clone, the App credential, the lock, the
   fetch-and-fast-forward before serve and before save, the 409 on a moved
   base, the operator author, the operator-tier commit with push retry, the
   maintainer-tier PR, the share fast-forward from the local clone, the
   reconcile of worker-written files, the timer fetch that propagates a
   merge, the commit log. Proven on a scratch repo in CI and on the box
   against the real configs repo with a throwaway branch.
3. **Dry runs and applies.** The scanner's inline trigger-profile preflight
   and readback plug-ins; `environment_close` / `environment_open` on the
   queue client and `POST /api/environment/reopen` on the scanner (idle and
   empty, lock held, queue restarted); the archiver's `--dry-run --policy`
   subprocess; the scanner's `save_preset` routed through the service. The
   deploy switch of the Linux configs root to the service's live checkout
   (ruling 4): site.env, Canopy's compose, the rendered config.ini, the
   one-time commit of the share's edits, and **the portal's editor flipped
   to its read-only mode** (its router's `read_only` switch makes every save
   a 405) for the interval until slice 6, and the worker's write root
   pointed at the service's inbox (the resolver's new
   `GEECS_CONFIGS_WRITE_ROOT`), so nothing but the service's own
   fast-forward touches the live checkout.
4. **Setups, the engine.** `apply_setup` in GeecsBluesky with `md`; the
   scanner's setups routes with the hazard confirmations; the module reads
   `setups/`; the converter; `run_action` and `ActionPlanLibrary` deleted.
   Hardware: one setup captured and applied on the bench instance, laser off,
   and one screen button on the module path.
5. **Setups, the surface, and the embedded editor.** The setups page
   (capture, diff, apply); the mount contract with the operator passed in;
   the scanner's preset mini; the Control Room launcher.
6. **The analysis kinds move.** The portal exposes `validate` and `schema`
   for recipes and groups bound to GEECS-Analysis's registry, as the
   configurator's plug-in; the run page mounts the embedded editor with its
   preview; `scan_analysis.config_editor`, the `editor` extra and the
   portal's `--config-editor` flag are deleted (the kill rule, after the
   maintainer's look).
7. **Docs and the promotion body.** `GeecsConfigurator/CLAUDE.md`, the root
   map and graph rows, the docs page, the fleet-map row; this brief deleted
   with its rules moved.

## Proof of done

On the Mac sim stack with the demo scanner: every file kind of the sim
experiment lists, draws from its schema, validates, and shows a correct diff
and tier; a commit lands on a scratch repo with the operator as author and
the log row says what applied; a moved base answers 409 with the diff. On the
services box: an operator changes the laser-off profile's amplitude from a
browser, the commit reaches main under their name, the share's clone
fast-forwards, the worker reopens its environment when the queue is idle and
empty, and the next laser-off scan fires at the new amplitude; a setup
captured from live readbacks is applied as one queue item with every step
completed by its device; a grant change opens a PR as the App, and its merge
reaches both clones and applies on the timer without anyone opening the
editor. Light and dark, no literal colour, the theme's tests walking the
templates.

## Out of scope

Per-user authentication; the operator registry (#882) and any grant check on
the queue path; the one-key identity behind a front door; Squirrel; DB writes
of any kind; the parameter tier's IOC restarts; the look (a design pass after
the maintainer has used it); Phoebus or any non-web editor.

## Size and cost rules

A slice ≤ ~700 changed lines (slice 1 may reach ~1,000 with the form renderer
and its tests; say so in the body). Reviews bounded, a fresh verifier for
fixes. No unprompted follow-up issues; the arc's tracking issue is the one
issue, with one comment per slice; comments elsewhere only on the lane PRs;
no hosts, addresses or accounts in any committed file, issue or PR body. The
mockup is not cited in the repo.

## Found along the way (not this arc's work)

- The `mcp-edges` import contract's name still says MCP imports GeecsBluesky's
  client, Data-Utils and ScanAnalysis; its forbidden list is right. Rename.
- `mv` and `run_action` queue items carry no operator today: `submit_plan` has
  no `md`, and `mv` wraps `bps.mv`, which would forward an `md` to each
  device's `set()`. The fix is the item's metadata or an explicit `md` on the
  GEECS wrapper; a slice of its own.

## Status

### 2026-11-06 — the red team's dispositions (before any slice)

A fresh-context adversarial review of the brief against the worktree, three
lenses. Twelve findings, all accepted; the brief above is the edited text.

| # | Finding | Disposition |
|---|---|---|
| 1 P1 | "No tolerance confirm-on-readback helper" was false: `CaConfirmSettable`, `CaMotor.set()`, `NativeSetpoint` confirm by readback; only the plain settable waits for the reply | Claim struck. A setup `set` is `abs_set(wait=True)` with the device's own completion, no second layer; a `check` matches as the confirm settable does |
| 2 P1 | Deleting the module's action route drops the operator-policy gates screen buttons get today and breaks Squirrel restore, which runs on that executor | Ruling 2 rewritten: two executors, one document. The module keeps its routes and gates and reads `setups/`; the queue path asks hazard confirmations and states that it checks no grants |
| 3 P2 | localStorage is per origin; one key cannot be shared across ports; the embedded editor runs in the host's origin | The host passes `operator` into the mount contract; CORS on the API; one-key unification out of scope |
| 4 P2 | The registry holds six non-file kinds, so "every registered kind has a folder" cannot pass | The table is the file subset with a `storage` column; the test pins both halves |
| 5 P2 | Six folder constants live in GeecsBluesky's resolver, which the configurator may not import | Slice 0 moves them into GEECS-Schemas; the resolver imports them |
| 6 P2 | The preflight resolves a trigger profile from disk, so the stated validate-then-dry-run order cannot see the unsaved edit; the archiver has `--policy` | Inline trigger-profile preflight on the scanner (slice 3); the archiver's dry run over a temp file by `--policy` |
| 7 P2 | `write_shot_offsets` is the worker's, inside the calibration plan | Shot offsets stay the worker's file; the service reconciles worker-written files, authored as the worker |
| 8 P2 | `mv` wraps `bps.mv`, so an `md` there would reach each device's `set()` | `apply_setup` takes `md` like `count`; the `mv` attribution moved to "found along the way" |
| 9 P2 | The environment reopen was underspecified: a closed environment passed the idle gate; no client verbs; a scan submitted during the window waits with nobody to start the queue | The reopen route holds the lock through close and open, requires idle and an empty queue, restarts the queue after; the client gains the two verbs; the chip reads "reopening" |
| 10 P2 | A GitHub App committing to main and "branch protection" as the maintainer gate contradict; the private repo's fetch on the share needs a credential | The ruleset stated: humans by PR, the App on the bypass list; the share fast-forwards from the service's clone as a local remote |
| 11 P2 | "Rebase the operator's edit" is a whole-document merge the store does not do; a naive rebase reverts another field | Fetch-and-fast-forward before serving; a moved base at save answers the store's 409 with the diff; never an automatic merge |
| 12 P2 | The analysis kinds' validation and schema are bound to GEECS-Analysis's registry, which the configurator cannot import | The portal exposes `validate` and `schema` for those kinds as the dry-run plug-in (slice 6) |

P3s folded: four kinds in `schema_export`, not seven; a screen `button`
binding names an action (there is no `call` widget); Canopy mounts the root
read-only and `config.ini` is rendered only when absent, so the root switch
is three steps; the scanner's own read routes stay; the form renderer moved
to GeecsWebTheme beside `plot.js`; one uvicorn worker and a bounded push
retry; "changed" defined for list fields; the module's short cache cited by
behaviour, not by line.

The verdict kept: the configs repo as the one store with a real commit path,
tiers on the schema, the service as a thin HTTP client, one setup file per
plan, the per-kind apply table, deleting the half-editor rather than growing
it. Slice 0 is a safe first PR after rulings 1 and 3.

### 2026-11-06 — the PR's correctness round (#1301)

One correctness-only round on the diff, a fresh reader. Every code claim in
"What exists today" checked out; eight findings, all fixed in the follow-up
commit on the branch:

| # | Finding | Fix |
|---|---|---|
| 1 | Scan variables sat on the "nothing to do (mtime)" row, but only the scanner's resolver re-reads on mtime; the worker reads the catalogue at environment open | Moved to the reopen row; the survey bullet says which reader sees what |
| 2 | A maintainer PR merged on GitHub was neither a serve nor a push, so it never propagated or applied | The service fetches main on a timer: a foreign commit fast-forwards both clones and runs the kind's apply; proof of done extended |
| 3 | The read-only rule and the worker's uncommitted `shot_offsets.yaml` were stated without mechanics | The worker writes into the service's clone; the reconcile runs on a timer and before any fetch, the measurement wins a race, operator edits are ordinary commits |
| 4 | Family files had a tier but were not a file | Named as package data shown read-only under the label `code`, outside the file table and its test |
| 5 | "Rulings before slice 2" while slice 0 bakes in the name and the store | Rulings 1 and 3 before slice 0, the rest before slice 2 |
| 6 | The screens' dry run cited a roster route that does not exist | Screens have no dry run in v1; the render-time check stands |
| 7 | A bare `tier` class attribute neither passes pydantic nor reaches `model_json_schema()` | `ClassVar` plus `model_config`'s `json_schema_extra` |
| 8 | Two paraphrases in quotation marks | The editor's actual words; #883 described as a recorded direction, not a ruling |

### 2026-11-07 — the maintainer's review on #1301 (GPT, four findings)

| # | Finding | Disposition |
|---|---|---|
| P1 | The service's clone was both the live configs root and the place drafts and PR branches were written, so a draft operator policy was readable by the device module before its PR was approved | **Fixed.** A bare repository, a live checkout that holds accepted main commits only and is never written or branch-switched, and a throwaway staging worktree per edit; a failed push removes the worktree and nothing reaches the live checkout; the log's four states defined |
| P2 | Derived channels were editable (maintainer tier) with no apply: the CA gateway builds their PVs at startup | **Fixed.** A derived-channels row: a gateway restart, never during a scan, recorded as *pending* and named to the maintainer; a landed commit is never written up as applied by that fact |
| P2 | Between the root switch (slice 3) and the analysis kinds' move (slice 6) the portal's editor kept writing files into a checkout the service fast-forwards | **Fixed.** Slice 3 flips the portal's editor to its existing read-only mode for the interval; ruling 4 says so |
| P2 | "No GitHub issues, no comments outside the lane PRs" contradicted CONTRIBUTING's tracking-issue rule | **Fixed.** The arc gets its tracking issue on the maintainer's word; the size rules allow it and its one comment per slice; later dispositions go there, not here |

The review also asked for an explicit exit checklist; it is in the opening
section.

### 2026-11-07 — the maintainer's follow-up review (one P2)

| # | Finding | Disposition |
|---|---|---|
| P2 | The reconcile fast-forwarded a live checkout holding the worker's uncommitted `shot_offsets.yaml`; git refuses that, identical bytes included (reproduced in a scratch repository) | **Fixed.** Nothing writes into the live checkout. The resolver gains a write root (`GEECS_CONFIGS_WRITE_ROOT`, defaulting to the read root) that on the box is the service's inbox; the reconcile commits from the inbox in a staging worktree, pushes, fast-forwards the clean live checkout and removes the inbox file; the worker reads the result at its next environment open. Slice 3 and ruling 4 say so |
