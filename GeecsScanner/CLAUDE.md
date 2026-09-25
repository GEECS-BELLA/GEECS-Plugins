# GeecsScanner — Developer Context for Claude

The web scanner console: a FastAPI service on the worker host that submits,
watches and stops scans through `geecs_bluesky.qs_client`. The third surface
on the GEECS surface kit (GeecsWebTheme), a peer of the Data Portal and the
logbook, and the operator front end (it replaced the PySide6 GEECS-Console,
deleted 2026-09-14; #869).

**The Start gate's form-shape check is client-side only.** `S.formable`
in `static/scanner.js` disables the Start button for a "retired or
unsupported preset plan"; nothing on the server has an equivalent.
`POST /api/submit` does validate — `ScannerService.submit` runs the
`Preset` schema, then preflight, and raises `invalid_request` on any
refusal (an unregistered plan, a device outside the worker's tree) — but a
preset the page refuses to build a form for is still submittable by `curl`
if it validates and passes preflight. Treat the gate as an affordance,
never a validation boundary: a plan-shape rule that must hold belongs in
the service layer or the schema.

## Layout

```
geecs_scanner/
  service/        pure Python — the tested surface; no FastAPI anywhere in it
    scanner.py    ScannerService: the verbs over the injected client + resolver
    models.py     every answer as a Pydantic model; `state` fields are kit words
    summaries.py  a stock plan item → one line + planned shots (ported from the console)
    streams.py    ProgressCache: the document + console streams reduced to a picture
    demo.py       DemoQueueClient / DemoResolver / demo_preflight — a manager in memory
    errors.py     ScannerError(kind, message, **extra); kind → HTTP status in one table
    actions.py    the action-plan preview: flatten nested runs over the schema models
    scanlog.py    read the run's scan.log from the folder the start document names
    settables.py  the movable panel's list: GEECS-Core's numeric_settables over GeecsDb rows, cached per process
    readback.py   one aioca caget of the gateway's readback PV (geecs_core.pv_naming) — the service's one async path
    trajectory.py isolated hardware-free preview over shared Bluesky expansion
    writer_status.py the Tiled writer's heartbeat → one kit word (the "tiled writer" chip, /health); shown, never a gate
  web/
    app.py        create_app (the process) and create_scanner_router (the same as a router)
    pages.py      GET / — the page (make_templates from geecs_web_theme.web: `root` in every context)
    api.py        one route per verb, three lines each
    events.py     GET /api/events — SSE: status, progress, log (scan.log), console
  templates/console.html   the page; the kit's vocabulary, page-only classes from static/
  static/scanner.js        API + EventSource + capture/form → Preset
  static/sweep-composer.js trajectory editing → Sweep payload
  static/trajectory-view.js draws returned coordinates only
  static/scanner.css       page-only classes; never re-skins a kit class
  __main__.py     geecs-scanner: --experiment | --demo, --port 8300, --root-path, --portal-url
deploy/           the unit template + DEPLOYMENT.md
```

## Rules

- **Client-side expansion, nothing re-derived.** A scan is a
  `geecs_schemas.Preset`; `expand_preset` turns it into the plan call; the
  scanner never binds a variable to a device itself and never computes
  detectors. `md["geecs"]` is provenance only. The scan number is claimed
  worker-side and read from the start document.
- **Plan-name generic.** `POST /api/submit` takes a preset whose plan call
  names any allowed plan. There is no mode enum in the API — an optimize
  plan, when it exists on the worker, is one more name and one more form
  section, not an API change.
- **The catalog lists pseudo entries as scannable** (#879, the pseudo arc
  #904): a pseudo has no single `target`, the worker binds it as a
  namespace noun under its catalog name and `expand_preset` resolves it;
  the page never binds variable → device. `ScanVariableOut.scannable` /
  `reason` stay in the model for the next kind the worker cannot take.
- **Operator policy, not agent policy.** No shot cap, no "refuse if
  anything is queued" — those are GEECS-MCP's posture. Preflight questions
  must be acknowledged; that is the one gate. Ownership (who may stop whose
  scan) arrives with the operator registry (arc PR 5); until then `force`
  is recorded and changes nothing.
- **One client, one lock, blocking calls.** Every service method holds the
  lock around its client calls; the web layer runs them on the threadpool.
  `readback` is `async` on the app's loop (aioca is
  async): it takes no lock and reads no DB, because `/api/events` shares
  that loop — anything blocking there freezes every viewer's stream.
  The stream consumer threads are daemons that are never stopped (a zmq
  socket touched from another thread can abort the process). The numerical
  trajectory preview takes no client lock: it has its own process/concurrency
  budgets and never calls the manager.
- **Imports.** `geecs_bluesky.qs_client`, `geecs_bluesky.config_resolver`,
  `geecs_bluesky.plan_names`, `geecs_bluesky.trajectory` (hardware-free
  numerical expansion only), `geecs_schemas`, `geecs_web_theme`,
  `geecs_core.db` (the settables list), `geecs_core.pv_naming` + `aioca`
  (the readback — the scanner reads gateway PVs directly, like the
  preflight does through the client seam), and
  `geecs_bluesky.tiled_spool` (the writer's heartbeat model and reader —
  the shared side of the spool, never `tiled_writer`, the service loop).
  Never the
  portal, the logbook, GEECS-MCP, or the engine's
  `plans`/`devices`/`run_engine`/`namespace`. `tests/test_boundaries.py`
  pins this and the absence of facility literals in code.
- **Errors are the taxonomy.** Raise `ScannerError(kind, message, **extra)`;
  the web layer maps the kind. Never raise HTTPException in the service.
- **The demo is sample content**, chosen by a flag; nothing in it is a
  default for a deployment.

- **Idle-only items.** A move (`mv`), an action (`run_action`) and the two
  calibration plans are queue items that run by themselves the moment they
  are added, so the service refuses them unless the manager is idle **and
  nothing waits** — the check and the add happen under the one lock. The
  action preview flattens nested `run` steps over the schema models in
  `service/actions.py` (never the worker's compiler: `geecs_bluesky.plans`
  is off limits); an unresolvable plan cannot be run.
- **Writes go through the resolver.** "Save as preset" calls
  `ConfigsRepoResolver.write_preset` (atomic, refuses overwrite unless asked,
  never creates the experiment folder); the scanner never opens a file in
  the configs tree itself. Committing the new YAML is a human act.
- **scan.log is read where the start document says.** `scan_folder` in the
  start document is the run's folder on this host; `service/scanlog.py`
  reads `<folder>/scan.log` from an offset and never creates anything. The
  page's tail shows scan.log by default and the manager console behind a
  toggle — two narrations, never reconciled.
- **Portal links are URL only.** `--portal-url` (a site value) prefixes
  `/run/<uid>` and `/day/<iso>` links in the rail; nothing from the portal
  is imported.
- **The Tiled writer's word is shown, never enforced.** `service/writer_status.py`
  reads `heartbeat.json` under `GEECS_TILED_WRITER_STATE` (the unit sets
  it; `/var/lib/geecs-tiled-writer`, the same on every host) and reduces it
  to `ok` / `degraded` / `failed` for the "tiled writer" chip and
  `/health`'s `tiled_writer` (`scripts/fleet_status.sh` reads it there).
  With the spool a dead writer loses nothing, so no preflight question, no
  Start gate and no refusal ever comes from it (owner's ruling 2026-09-25;
  pinned in `tests/test_writer_status.py`). The thresholds are the
  measurement's (25–28 s per run): `pending` ≤ 1 fresh is ok, ≥ 3 or a
  `.failed` file is failed.

## Testing

`poetry install` then `pytest`. The suite runs over `DemoQueueClient` with
`period=0`, stepping the fake manager by hand — deterministic and fast.
`GET /api/events?once=1` is the one-round form of the stream for tests and
`curl`.

## The page

`console.html` is filled entirely by `scanner.js` from `/api/*` and kept
live over `/api/events`; Jinja renders only the experiment, identity and
version. Rules the logbook learned, pinned by `tests/test_page.py`:
`url_for(...).path` never the absolute URL; every literal `data-state`
(template or `setChip` in the script) is a kit word; every script passes
`node --check`; every class the page uses is styled by the kit, the theme
or `scanner.css`. The first three are `geecs_web_theme.testing`'s helpers
— the test file asserts over their findings and adds only what is the
scanner's own (the script's `K` table, `setChip` literals). Likewise the
forwarded-prefix middleware, the `/theme` mount and the templates factory
come from `geecs_web_theme.web` (the `web` extra): import them, never copy
them. The form builds a `Preset` and posts it — the page never
resolves a variable to a device.

## What is not here yet

The operator registry and ownership (arc PR 5: `operators.yaml`,
`geecs.operator` in theme-boot, the `denied` banner for a foreign running
item) and the Caddy front door (dropped 2026-09-13 — three ports, one
bookmark each). GEECS-Console itself is gone (PR 6, 2026-09-14; tag
`geecs-console-v0.32.1-final`).


## Optimize mode

The form selects an `OptimizerConfig` by ID; shared `prepare_optimizer_preset`
resolves defaults and required devices before both preflight and submission.
Saved presets retain the authored devices/kwargs; expansion happens only for
preflight/submission through the shared client seam.
Expanded required devices have `essential=True` and `save_images=True` even
when an authored table row says otherwise. ProgressCache reduces the worker's `optimization` stream; NaN
becomes null for JSON. Set to best takes a run UID, rejects stale/incomplete
runs, unsuccessful runs, offers older than 15 minutes, and offers invalidated
by a service-submitted move/action. It records the requesting operator and
submits recorded physical targets as one idle-only `mv`. The panel shows the
scan, completion time and exit status. Out-of-band gateway writes are not
observed by this cache; Set to best remains an explicit operator action. The scanner
never reconstructs pseudo offsets or imports the analysis runtime.

Optimizer listings expose unavailable names and reasons as well as usable names;
listing errors are displayed even when Optimize is disabled. Recorded physical
best targets remain visible after an offer is invalidated. Set to best confirms
the targets before queueing; queue acceptance is never described as completion.

## Inline Sweep composer

New scan starts unconfigured and selects Count / Sweep / Optimize. Sweep
expands inline into Axis Sweeps / Patterns. The browser builds typed inputs
and only draws coordinates returned by `POST /api/trajectory`; it never
implements spacing or pattern geometry. The service uses the shared Bluesky
trajectory module: raw request bodies are limited to 256 KiB before parsing,
with at most 250,000 expanded coordinates and two concurrent previews. Axis
sweeps with at most eight axes and X2X expand on the threadpool; all spirals
and larger axis sets use a disposable process with 256 MiB RSS and eight
seconds. Exact output counts do not bound square-spiral or high-axis Cycler
construction time. The service package imports lazily so the child does not
load the queue client or execute its module twice. These
are interactive computation budgets, not shot limits. Responses sample at
most 2,000 positions / 10,000 coordinates and disclose sampling; submitted
payloads always retain the full original trajectory. Relative previews are
offsets; execution captures the staged baseline. Stale requests cannot replace
newer form state. Superseded fetches are aborted, disconnected requests cancel
their child, and busy previews retry twice before showing a manual retry.
Preview is optional: Start still runs normal preflight validation, and Save
validates the nested Sweep schema without requiring numerical expansion.
The controls every scan type needs (the **Every scan** block) remain shared outside the trajectory tabs; background is
Count-only. A malformed preset clears the old trajectory, loads its own
capture fields, and disables both Start and Save with an explanation. The
points table is materialized only while its disclosure is open.

`sweep-composer.js` owns editing and numeric list parsing; `trajectory-view.js`
only renders server results and can be reused independently of the form.
Lists accept numeric comma/tab/space/newline values, preserving repeats. There
is no axis-count ceiling or evaluator. Presets seed the form; no preset is
automatically selected. Optimize retains its existing required-device and
Set to best controls.
