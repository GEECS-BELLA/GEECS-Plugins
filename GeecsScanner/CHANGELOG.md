# Changelog

All notable changes to `geecs-scanner` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.

## [0.10.0] - 2026-09-16

### Changed

- Adapt the existing range/grid form and demo preset to submit Sweep payloads; summarize Sweep queue items and retain historical stock item summaries. The full composer follows separately.

## [0.9.2] - 2026-09-16

### Fixed

- Show unavailable optimizer configs and listing failures; render exact physical best targets persistently and confirm before queuing a best move. Distinguish queue acceptance from move completion.

## [0.9.1] - 2026-09-16

### Fixed

- Allow Set to best only for successful recent runs, expire offers after fifteen minutes, invalidate after service-submitted moves/actions and record the requesting operator. Preserve authored presets on save, share optimizer expansion and disable Optimize when no compatible configs are available.

## [0.9.0] - 2026-09-16

### Changed

- Enable Optimize mode with config selection, strict acquisition, locked required devices, finite iteration budgets and live output/best values. Add a run-bound, idle-only Set to best action using recorded physical settings. Extend the demo and SSE bridge without numerical worker dependencies.

### Fixed

- Decode the worker’s Tiled-safe optimization column names through the shared import-light codec.

## [0.8.0] - 2026-09-15

### Changed

- **Pseudo scan variables are scannable from the picker** (#879, closes
  it): `GET /api/scan-variables` lists a `kind: pseudo` entry with
  `scannable: true` (no `target` — the worker binds it as a namespace
  noun under its catalog name, GeecsBluesky 0.90.0), preflight and
  submit expand it through `expand_preset` like any axis, and
  `POST /api/move` moves one (a manual bump: today's positions become the
  baseline, `0` moves nothing). The page changes nowhere else. The demo
  manager's device tree carries the demo catalog's pseudo noun so the
  preflight's reference check passes in `--demo`.

## [0.7.0] - 2026-09-15

Two gates that refused what the queue and the form are for (#900, #905).

### Fixed

- **A scan composed from scratch can Start** (#900). `recalc` no longer
  folds "a preset was loaded" into the form's validity and
  `updateStartGate` no longer asks for one: Start needs a valid form
  (shots, the axes) and an idle, reachable manager — a configs tree with
  no presets submits from the page, and the `preset <name>` provenance
  note simply stays empty. `buildPreset` already built the document from
  the form (`adhoc`, an empty device list scans with the readbacks only,
  per the schema); the demo's first-preset default is unchanged.
  **Judgment call:** *Save as preset…* now follows the same rule —
  enabled by a valid form, not a loaded preset — since compose-then-save
  is the workflow #900 describes.
- **A submit while a plan runs is queued behind it** (#905). The real
  client's add-then-start answered *RE Manager is busy* to `queue_start`
  and removed the item again; that refusal now reads as success when the
  manager's queue is started (the fix is `geecs-bluesky` 0.87.1's, the
  one client every queue front end shares), and the page's Start gate no
  longer closes on a running plan — the next scan queues behind it; a
  *paused* plan still closes it ("resume or stop it first"), since an
  item added then would be removed. The idle-only gate for a move, an
  action and the calibration plans is untouched.
  `tests/test_submit_while_running.py` runs the real `ZmqQueueClient`
  over a stub manager mid-plan through `POST /api/submit` → 200, the item
  stays queued.

## [0.6.0] - 2026-09-14

The New scan form after a day of scans (#895, #896): shorter, and it opens
on the mode choice.

### Changed

- **The preset picker leaves the headline row** (#896). It is a compact
  "Load preset…" select in the form's footer beside "Save as preset…" —
  load, the `preset <name>` provenance note, save and Start in one row,
  the header keeps its two segments, and the body opens on axis 1. The
  picker is a verb: it snaps back to "Load preset…" after a load, so the
  same preset can be reloaded over an edited form; the note carries which
  preset seeded the form. How a preset seeds the form (`formShape`,
  `fillFormFromPreset`, the demo's first-preset default) is unchanged.
- **The static explanatory hints are gone** (#895): the prose under the
  preset, variable, trigger-profile, description, calibration and drawer
  controls, the per-mode `mode-note` paragraph and its `NOTES` table, and
  the success-case prose of `presets-note`, `actions-note` and `mv-hint`.
  What stays is state or a unit: the point count and direction, the
  `.err` slots, `shots-hint` (`num` vs `shots_per_step`), `seconds` under
  the shot period, the device search's live count, and every
  load-failure / empty-listing note.
- The empty device table reads "No devices — add one." in the template too
  ("Pick a preset above." pointed at a row that no longer exists; the
  script already used that wording). The trigger-profile field keeps its
  environment-open gotcha as hover text (`title`). A preset whose load
  fails no longer leaves its name as the form's provenance.

## [0.5.1] - 2026-09-14

PR 6 of the web scanner arc — GEECS-Console deleted. No code change.

### Changed

- `README.md` / `CLAUDE.md`: the scanner is described as the operator front
  end (the PySide6 GEECS-Console it replaced is deleted; final state at the
  tag `geecs-console-v0.32.1-final`), and the deploy note no longer names
  the dropped Caddy front door — three ports, one bookmark each.

## [0.5.0] - 2026-09-14

PR 5b of the web scanner arc — the movable panel.

### Added

- **`GET /api/settables`** — every numeric settable of the experiment from
  the GEECS DB (`GeecsDb.get_experiment_device_variables` with GEECS-Core
  0.7.0's `alias`, ordered by its `numeric_settables`): **aliased
  variables first** (alphabetical by alias), then the rest by canonical
  `Device:Variable`; each row carries the alias beside the canonical name,
  never instead of it. Read once per process and kept — a DB roster change
  restarts the CA gateway anyway; a failed read is reported, not cached.
- **`GET /api/readback?variable=Device:Variable&units=`** — one live
  reading over the CA gateway's readback PV (`geecs_core.pv_naming`, one
  `aioca.caget` on the app's loop — the service's one async path, which
  takes no lock and reads no DB): value, the units the caller passes, the
  channel's stamp and age. The readback, not the `:SP` echo the Qt console
  showed as "set". The name is split by GEECS-Schemas 0.25.0's
  `split_device_variable`.
- **The devices · move panel** lists the settables (the option value is
  the canonical name; the catalog-only picker is gone) and shows the
  picked variable's readback with its age in the kit's live-value row,
  polled once a second while a variable is picked; a gateway that does not
  answer reads as `stale`.
- Demo backend: a fixed alias-first list and a readback that follows the
  fake manager's moves.

### Changed

- `geecs-core` is a direct dependency (it was a transitive of
  geecs-bluesky), the console's stated convention.

## [0.4.0] - 2026-09-13

PR 5a of the web scanner arc — layout and freshness, from Sam's notes on the
deployed page (2026-09-13).

### Changed

- **Presets are a dropdown in New scan**, first field of the form it seeds;
  the rail keeps its section links and the Recent list. The rail's preset
  picklist showed too much.
- **Actions are a dropdown** with the step preview and Arm / Run under it;
  the stacked picklist (one row per plan, step counts beside) is gone. A
  plan that cannot run stays pickable so the preview can say why.
- The trigger-profile hint says when an edit reaches the worker (its next
  environment open) — the one config kind the worker materialises at
  startup; everything else is read fresh per request (GeecsBluesky 0.86.1
  makes the scan-variable catalog follow the file too).

## [0.3.1] - 2026-09-13

### Changed

- Merge of `master` into `feature/web-scanner` (GeecsWebTheme 0.6.1, the
  tinycss2 CSS guards of #875): the "every class the page uses is styled"
  test reads the stylesheets through the theme's `classes_used` and
  `styled_classes` instead of a regex over `.name`; the theme's `testing`
  extra joins the dev group for it, and `poetry.lock` is relocked against
  the merged path dependencies. No runtime change.

## [0.3.0] - unreleased

The rest of the mock — PR 4 of the web scanner arc. Opens with the
shared-glue adoption the #871/#872 reviews asked for once
`geecs_web_theme.web` existed, then the panels day one left out.

### Added

- **Idle-only items** — `POST /api/move` (one `mv` queue item; the variable
  resolved exactly as a scan axis is, through `scan_variable_reference`,
  pseudo entries refused), `POST /api/actions/{name}/run` (`run_action`),
  `POST /api/calibration/check` and `/measure` (`check_shot_sync`,
  `measure_shot_offsets`; at least two devices; `measure` carries `shots`
  and `write`). All four refuse with 409 `policy_refusal` unless the manager
  is idle **and nothing waits** — the queue is started, so an item added
  behind a running or waiting scan would run by itself the moment it ends;
  the check and the add happen under the one lock.
- **Actions** — `GET /api/actions` (name, description, flattened step count,
  nested plan names, or the reason it cannot run) and `GET /api/actions/{name}`,
  the preview: every concrete step in execution order with the nested plan
  each was inlined from and the number of hardware writes. The flatten is
  the scanner's own walk over the schema models (`service/actions.py`) —
  the worker's compiler is off limits to a client.
- **Calibration** — `GET /api/calibration`: the stored `shot_offsets.yaml`
  summarized (reference, when, profile, per-device offsets, the largest).
- **Save as preset** — `POST /api/configs/presets/{name}` writes
  `presets/<name>.yaml` through `ConfigsRepoResolver.write_preset`
  (GeecsBluesky 0.86.0): the URL names the file, an existing preset is
  refused with 409 `exists` unless `overwrite` is set, the answer names the
  path so the operator knows what to commit.
- **The scan.log tail** — a `log` event type on `/api/events` and
  `GET /api/scanlog?offset=`: `service/scanlog.py` reads
  `<scan_folder>/scan.log` from the folder the start document names (the
  worker claimed it; this process reads and never creates), whole lines
  only, resumable by offset, a new run replayed from the top, an
  unreadable folder said once. `ProgressOut` gains `scan_folder` and `day`.
- **The page** — three panels under the queue as the mock drew them:
  *Devices · move* (variable + value, idle gate), *Actions* (picklist →
  preview → Arm → Run; arming is never remembered), *Calibration* (the
  stored offsets, shots + store-the-result fields, Check sync, Measure
  offsets… behind a dialog that says what will happen); the **Add device**
  drawer over `/api/devices` (bare device names, the table's rows marked)
  with a remove control per row; the **Save as preset** drawer showing the
  YAML it becomes; the tail's `.seg` now toggles **scan.log** (default) and
  the manager console; the rail's **Recent** links open the portal's run
  pages (`--portal-url`, a site value; `QueueRow.run_uids`) and its heading
  the portal's day.
- `--portal-url` on `geecs-scanner`; `GEECS_SCANNER_EXTRA_ARGS` in
  `site.env` is where a site sets it.

### Changed

- **The web glue is imported, not copied.** The `geecs-web-theme` path
  dependency gains `extras = ["web"]`; `web/app.py` uses
  `geecs_web_theme.web.ForwardedPrefixMiddleware` and `mount_theme`, and
  `web/pages.py` builds its environment with `make_templates(TEMPLATES_DIR)`
  (`root` in every context comes from the shared factory). Deleted: the
  scanner's `ForwardedPrefixMiddleware` + `_clean_prefix`, `_root` + the
  `Jinja2Templates` setup, and the try/except around the theme mount (the
  theme has been a hard dependency since 0.1.0). The scanner's named
  `/static` mount and `url_for(...).path` stay — they are the page's own.
- `tests/test_page.py`'s three template guards are one-line asserts over
  `geecs_web_theme.testing` (`bare_url_for_calls`, `unknown_data_states`,
  `inline_scripts` + `javascript_syntax_error`); the scanner-specific
  checks (the script's `K` table and `setChip` literals, the page's script
  file, "every class is styled") stay local. Proven to bite: a planted
  template with a bare `url_for`, a `data-state="FAILED"` and a broken
  inline script fails all three.

## [0.2.0] - 2026-09-13

The page — PR 3 of the web scanner arc. Day one of the console's job, as
the reviewed mock drew it, on the kit.

### Added

- **`GET /`** renders `templates/console.html`: topbar (experiment, manager
  and doc-stream chips, operator, density and theme pickers), rail
  (sections, presets from the configs tree, recent scans), and three
  panels — **Now** (`.chip.lg` state, scan number, plan line, `.meter`,
  `.live` shots / planned / document age / manager age, Pause · Resume ·
  Stop, the manager console tail behind a `.seg` whose scan.log half
  arrives with the run's folder), **New scan** (mode `seg`: No-scan · 1D ·
  Grid · Background · Optimize greyed; acquisition `seg`, strict default;
  axes with `aria-invalid` validation; shots, trigger profile, shot period,
  description; the preset's devices as a table with save-images and
  essential boxes; the estimate; Start), **Queue** (running / waiting /
  finished, sticky header, Clear queue…).
- **`static/scanner.js`** drives it: the JSON API for configs and the queue,
  `EventSource` on `/api/events` for status, progress and console lines
  (with `since` on reconnect), the form → `Preset` builder (a `count`,
  `scan` or `grid_scan` plan call from the fields; the preset's device
  group edited in place), the acknowledgement dialog over the preflight
  questions, and the **replace-the-waiting-item** dialog for the manager's
  one-deep queue (a 409 with `pending_items` → resubmit with
  `clear_pending`). Operator name kept under `geecs.operator` beside the
  theme's keys until the registry (arc PR 5).
- **`static/scanner.css`** — the page's own classes only; the kit is not
  re-skinned.
- `tests/test_page.py` — the logbook's three template guards (`url_for`
  takes `.path`, literal `data-state` values are kit words — the script's
  `setChip` literals too, every script parses under `node --check`), the
  proxy-prefix render, and "every class the page uses is styled".

### Changed

- The JSON pointer that was `GET /` moved to `GET /api`.
- `GET /api/events` sends `status` every round, not only on change, so the
  page can show how long ago the manager answered. Console frames carry
  `id: <epoch>:<seq>`; a browser reconnect resumes through `Last-Event-ID`,
  and a new epoch (the scanner restarted) replays from the start.
- `create_scanner_router` is the API + events only; the page is registered
  by `create_app`, which owns the named `/static` mount it needs.

### Fixed (from adversarial review, before merge)

- A preset whose plan the form cannot express (`list_scan`, a `rel_*`, a
  two-motor `scan`) rendered and would have submitted as a 1D `scan`. The
  form now says so and disables Start; a `background` flag survives a mode
  change.
- Every kit status word the script sets lives in one `K` table, pinned to
  `geecs_web_theme.STATES`; the guard also refuses a literal in `setChip`.
- The estimate no longer assumes 1 Hz: it quotes a time only when a shot
  period is given; the points hint shows the effective step when the typed
  step does not divide the range.
- The scanner's template, script and stylesheet joined GeecsWebTheme's
  literal-colour walk.

### Owed (recorded, not in this release)

- The forwarded-prefix middleware, the `root` context processor and the
  template guards are verbatim copies of the portal's and the logbook's;
  they move into GeecsWebTheme in arc PR 5, when the fourth surface makes
  three copies four (the brief's PR-5 row carries the same line).

## [0.1.0] - 2026-09-13

The service layer and the HTTP API — PR 2 of the web scanner arc
(`Planning/native_bluesky/10_web_scanner.md` (#869)). No page yet; that is 0.2.0.

### Added

- **`geecs_scanner.service.ScannerService`** — the verbs over an injected
  `geecs_bluesky.qs_client` client and configs resolver — the writes hold
  one lock, the reads none, so a two-minute graceful stop never freezes a
  status poll — every answer a Pydantic model: `status` (manager poll + readiness verdict),
  `queue` (running / waiting / finished, summarized), `list_configs`
  (`presets`, `trigger_profiles`, `scan_variables`, `actions`,
  `optimizer_configs` — the last listed now so the day an optimize plan
  exists the listing does), `scan_variables` (the catalog with `kind`;
  pseudo entries listed with `scannable: false` and the refusal text),
  `preset`, `devices`, `preflight`, `submit`, `pause`, `resume`, `stop`,
  `clear`, `progress`.
- **Submission policy**: preflight runs before every queue; a refusal is
  `invalid_request`; an unacknowledged question is `policy_refusal`
  carrying `needs_acknowledgement`; acknowledged checks are stamped
  `continued` into the `SubmissionRecord` in `md["geecs"]`, with the
  operator's name beside it. No shot cap, no queue-etiquette refusal —
  those are agent posture (GEECS-MCP), not an operator's.
- **`ProgressCache`** — the worker's pickled document stream and the
  manager's console text reduced to one picture (scan number, planned
  total from the start document with `max_iterations` as the adaptive
  fallback, shots from the **row streams** — `primary` for a strict run,
  the sampler's `shots` for a gated one, the same pair the s-file writer
  counts — exit from the stop document, the failed-move prefix as the
  paused reason, cleared by the next row) plus a ring of console lines
  with a cursor. Daemon consumer threads, never stopped.
- **`summaries.summarize_item`** — a stock plan item (`count`, `scan`,
  `grid_scan`, `list_scan`, the relative variants, `mv`, `run_action`,
  the calibration plans) read back into one line and a planned shot
  count; never raises on a shape it does not know. Ported from the
  console's `queue_panel`, rewritten for the stock verbs.
- **The web layer**: `create_app` / `create_scanner_router`, the portal's
  forwarded-prefix middleware, `/theme` mounted from GeecsWebTheme,
  `/health`, the JSON API (`no-cache` listings, errors as
  `{"error": {"kind", ...}}` mapped kind → status), and **`GET /api/events`**
  — Server-Sent Events with `status`, `progress` and `console` event types,
  a `since` cursor for reconnects and `once` for a single round.
- **`--demo`** — an in-memory manager (`DemoQueueClient`) that runs
  submitted items by itself, emits the same bluesky documents the worker
  would, pauses at step boundaries and stops gracefully; a `DemoResolver`
  with three real presets and a catalog carrying a pseudo entry; a
  `demo_preflight` that validates by the real expansion and asks one
  question. The test suite runs on it with the manager stepped by hand.
- `tests/test_boundaries.py` pins the import boundary (no portal, logbook,
  console, MCP or engine internals) and the absence of facility literals.
- `deploy/geecs-scanner.service` (template) + `deploy/DEPLOYMENT.md`; the
  unit joins `deploy/render_units.sh` and the fleet map (port 8300).

### Fixed (from adversarial review, before merge)

- Gated runs advanced no progress: only `primary` events counted, and a
  gated run's rows are the `shots` stream. Both row streams count now,
  pinned against `geecs_bluesky.callbacks.ROW_STREAMS`.
- One lock across reads and writes: a graceful stop (up to 120 s) froze
  every status poll and the SSE stream. The reads are lock-free.
- The demo disagreed with the worker three ways: a stopped run's stop
  document said `abort` (RunEngine.stop marks it `success`); pause was
  written into the progress picture (on the worker only the manager's
  `re_state` says paused); a second submission while one waited was
  accepted (the real client refuses with `pending_items` unless
  `clear_pending`). All three now match the worker.

### Not yet

The page (0.2.0). The scan.log tail (needs the run's folder; the stream
gains a `log` event type with it). `mv`, `run_action`, the calibration
verbs (0.3.0). The operator registry and ownership-gated stop (arc PR 5).
