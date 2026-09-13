# The web scanner: the console as the third web surface — the arc brief

**Status:** drafted 2026-09-13 from a three-way survey (GeecsBluesky on this
branch, GEECS-Console, the web foundation on master) and a clickable mock
Sam reviewed the same day ("really nice first mockup … no immediate major
things to change; once a working version is up, we can iterate").

**Why it is a merge-gate item.** Sam's ruling (2026-09-12): this branch cannot
merge to master until every piece is in place — the new backend **and** a web
front end that replaces GEECS-Console. Master keeps a working GUI until then.
The console is to be **deleted, not rewired**: its submit path already calls
verbs the rebuilt client removed (`submit_scan`, `move_variable`,
`submit_action`, `describe_action`), so on this branch Start, manual moves and
action execution are dead by design (`03` §10.5). This arc is the replacement,
and its closing PR deletes the console — which is what lifts the gate.

**What it builds on.** The web-surface-kit arc (#856/#857/#860/#863/#865,
GeecsWebTheme 0.2.2 + GeecsLogbook 0.9.0, on master 2026-09-13) was laid for
this: one shell, seven status words, five pane states, the overlay ladder,
density, and the FastAPI + Jinja + vanilla-JS pattern the portal and logbook
share. The scanner is the surface the kit was designed *for* — "the one with
the hardest requirements (live values, write actions, hazards, keyboard), so
it sets the vocabulary rather than inheriting a compromise" (GeecsWebTheme
CLAUDE.md). It needs the master → branch sync first (in flight 2026-09-13).

---

## 1. What exists, what does not (survey, 2026-09-13)

**The verbs exist and are branch-current.** `geecs_bluesky.qs_client`:
`status`, `submit_plan`, `submit_preset`, `request_pause`, `request_resume`,
`stop_scan`, `queue_items`, `history_items`, `running_item`, `clear_queue`,
`allowed_plan_names`, `allowed_device_names`, `readiness`; plus
`run_submit_preflight(preset, experiment)` → `PreflightReport` (refusal |
outcomes | questions), `build_submission_record`, `expand_preset` →
`QueueItem`, and `ConfigsRepoResolver` for presets / trigger profiles / scan
variables / actions / optimizer configs. Every call is a blocking 0MQ
round trip (2 s bounded), documented safe from a background thread.

**The plan surface is one tuple** — `GEECS_PLAN_NAMES` (22 names): 18 stock
`bluesky.plans` verbs bound with the GEECS kwargs (`trigger_profile`,
`shots_per_step`, `acquisition="strict"|"gated"`, `non_essential`,
`shot_period`), plus `mv`, `run_action`, `measure_shot_offsets`,
`check_shot_sync`. **Gated vs strict is a kwarg, not a plan.**

**The submission document is the `Preset`** (`geecs_schemas.preset`: devices
with `save_images`/`essential`, a `PlanCall{name,args,kwargs}`, trigger
profile, background). `ScanRequest` is orphaned — nothing in GeecsBluesky
imports it, its acquisition vocabulary still says `free_run`, and it carries
the dead save-set concept. **The scanner does not use `ScanRequest`.**

**What does not exist, and this arc must build:**

| gap | consequence |
|---|---|
| no HTTP anywhere near the queue (0MQ only, ports 60615/60625/5568) | the scanner is a **server-side** FastAPI process wrapping `qs_client`; a browser never talks to the manager |
| the document stream is **pickled Python** ("non-Python subscribers not offered today", qserver DEPLOYMENT.md) | a server-side `RemoteDispatcher` → JSON → **SSE** bridge; the repo's web tier has no live transport precedent (the portal polls) |
| blocking, synchronous client | every handler runs the client on a threadpool, behind one lock, one client per process |
| no optimization plan (Xopt core only; `OptimizationSpec` and `list_optimizer_configs` exist with nothing behind them) | Optimize is a **greyed** mode with a tooltip; §5 keeps the door open |
| no pseudo axes (`expand_preset` refuses `kind: pseudo`; `09_pseudo_transform.md`) | pseudo entries listed **disabled** in the variable picker; §5 |
| no per-item queue remove, no environment open on the client | clear-all is the recovery verb (Sam: "clear all is fine for now"); environment open stays the readiness unit's job |

**GEECS-MCP is inspiration, not canon** (Sam, 2026-09-13: "spun up super fast").
Its `_*_impl` functions are a transport-pure two-layer design worth copying
in *shape* — the JSON envelope, the error taxonomy, the acknowledge loop for
preflight questions, the ownership-gated stop — but its submit half is stale
against this branch and its identity is one string per process. Build the
service layer on `qs_client` directly; port the policy, not the code.

**From GEECS-Console, unchanged into FastAPI:** `services/health.py`
(`GatewayTiledDbHealth`), `services/configs.py`, `config_store.py`,
`presets.py`, `scan_variable_store.py`, `trigger_profile_store.py`,
`action_library_store.py`, `ops_paths.py` — twelve of seventeen services
modules are pure Python. `app/scan_monitor.py` is the reference for consuming
the two streams (adaptive totals from `max_iterations` included). Dropped:
`services/background.py` and everything Qt (~1,000 lines of thread-hop
machinery whose only reason is single-threaded widget painting), the scan
browser (the portal is a strict superset), the rep-rate field (bound to
nothing), the save-set widget (dead since GEECS-Schemas 0.21.0).

---

## 2. Rulings (Sam, 2026-09-13)

- **Free-run == gated.** The mode note says so; there is no `free_run` word.
- **Default acquisition is `strict`** for no-scan, 1D and grid.
- **Optimization is paused**, not dropped. Nothing in the API or the page may
  make it harder to add (§5).
- **Clear-all is enough.** No per-item remove verb in this arc.
- **Operator name: liked, not essential yet.** If adopted across logs and
  scans it gets **one** centralized mechanism, not two fields (§4, phase 5).
- **Ports:** the portal's "read-only" was history, not principle. Three
  surfaces run as **separate processes behind one front door** (§3.2).
- **The mock is the design.** Iterate on the working version, not the mock.
- **Interim data-taking:** HTU is down for a day or two; if the scanner is not
  up by then, the host checkouts go **back to master** for data-taking. No
  interim CLI is built for this arc.
- **Hardware acceptance needs no beam** — a strict `count` and a 1D magnet
  scan with cameras on dark exercise the whole path. Beam is available if a
  case ever needs it.
- **The tail shows both narrations** — scan.log and the manager's console
  text — behind a `.seg` toggle, scan.log the default. Their shot counts are
  never reconciled (GEECS-Console CLAUDE.md).
- **Kit additions (PR 1) go to master**, where the portal and logbook can use
  them now; one more master → branch sync follows.
- **Operators are a registry, not a free-text field**: pick from a list, with
  a built-in **guest** profile for visitors. No passwords — "our team is small
  and I trust people"; the posture the Qt console had.

---

## 3. The design

### 3.1 One package: `GeecsScanner`

`GeecsScanner/` (module `geecs_scanner`, unit `geecs-scanner`, port **8300**;
Tiled 8000, MCP 8100, portal 8200). Dependency cone = the MCP's:
`geecs-bluesky[qs-client]`, `geecs-schemas`, `geecs-web-theme`, `geecs-core`
(GeecsDb for health and completions), `geecs-data-utils` (today's scan
folder). **Never** the portal, the logbook, ScanAnalysis, or engine internals
— a peer client of the queueserver, same standing as the console had.

Two layers, one seam, like the logbook:

- `geecs_scanner/service/` — pure Python over `qs_client` + the resolver.
  Every function takes the client and returns a Pydantic model; no FastAPI,
  no JSON strings. This is the tested surface, and the layer a later OSPREY
  or MCP client can import.
- `geecs_scanner/web/` — `create_app(client, resolver, *, experiment,
  configs_root) -> FastAPI` **and** `create_scanner_router(...) ->
  APIRouter`, the logbook idiom, so the surface could be mounted elsewhere
  at zero cost even though it ships as its own process.

Kept from the portal verbatim: `_ForwardedPrefixMiddleware` + `{{ root }}`,
`/theme` mounted from `geecs_web_theme.static_dir()`, routes-per-concern
with a `Context` dataclass, `/health` with the version, `/openapi.json` on.
Kept from the logbook: the three template guards (`url_for(...).path`,
literal `data-state` ∈ `STATES`, every inline `<script>` passes `node
--check`), added to `_SURFACES` in GeecsWebTheme's literal-colour walk.

### 3.2 Separate processes, one front door

The portal is the box's memory-capped, freely-restarted process
(`MemoryHigh/Max`, `TimeoutStopSec=600`); the scanner owns the **abort
path**, which must never live inside something we kill on purpose, and holds
long-lived 0MQ subscriber threads that cannot be stopped without risking the
process. Different restart cadence, different dependency cone, different
unit. What the operator wants is nevertheless **one origin**: one bookmark,
one theme and density choice, one operator name — all of which are
same-origin `localStorage` under the theme's keys.

So: a reverse proxy on the worker host — **Caddy**: a web server whose only
job here is to listen on one port and send each path to the right process,
adding the `X-Forwarded-Prefix` header the portal already reads; one static
binary, a ten-line config, its own systemd unit, automatic TLS if ever
wanted; the fleet's first non-Python service, accepted for that reason — on
80/443 routing `/portal` → 8200, `/log` → 8200/log,
`/scan` → 8300. The forwarded-prefix middleware the portal already has exists
for exactly this, and OSPREY's panel integration already assumes a proxy.
SSE needs `flush_interval` / no buffering on that route — one line. The
services keep answering on their own ports for probes and `/fleet-status`.

### 3.3 The HTTP API

All JSON, `no-cache`, the portal's envelope conventions. Errors are the
taxonomy → HTTP: `invalid_request` 400, `policy_refusal` 409,
`manager_unreachable` 503, `not_found` 404, `task_timeout` 504.

| route | over | notes |
|---|---|---|
| `GET /api/status` | `client.status()` + `readiness_verdict` | `re_state`, `manager_state`, items, running uid, readiness word |
| `GET /api/queue` · `GET /api/history?limit=` | `queue_items` / `history_items` / `running_item` | rows summarized server-side by a **new** summarizer for stock-plan items (the console's `summarize_item` read `ScanRequest` documents and `geecs_*_plan` names the queue no longer carries; it cannot be ported — PR 2 wrote `summaries.summarize_item` over `count` / `scan` / `grid_scan` / `list_scan` / `mv` / `run_action` arguments) |
| `GET /api/configs/{kind}` | `ConfigsRepoResolver` | `presets`, `trigger_profiles`, `scan_variables` (with `kind`, alias first), `actions`, `optimizer_configs`; `GET /api/configs/presets/{name}` resolves one |
| `GET /api/operators` | `operators.yaml` in the experiment's configs tree | the operator picklist; a built-in `guest` entry is always present |
| `GET /api/devices` | `allowed_device_names` + namespace type | the "add device" list; a device the gateway does not serve does not appear |
| `POST /api/preflight` | `run_submit_preflight(preset, …)` | body = a `Preset`; returns refusal \| questions \| outcomes; submits nothing |
| `POST /api/submit` | `build_submission_record` + `client.submit_preset` | body = `{preset, acknowledged: [check…], operator}`; unacknowledged questions ⇒ 409 with `needs_acknowledgement`; acknowledged checks stamped `continued` in the `SubmissionRecord` that rides in `md["geecs"]` |
| `POST /api/pause` · `/api/resume` · `/api/stop` · `/api/clear` | the four verbs | `stop` and `pause` carry `{force}`; ownership (§4, phase 5) compares the caller's operator to the **operator stamped in the running item's metadata** — `submit` writes `md["geecs"]["operator"]`, and the manager's `running_item()` returns the item with its `kwargs.md`. Not the item's `user`: one client per process stamps `user=<the scanner's identity>` on every item (`ZmqQueueClient` sets it at `item_add`), so every browser operator would look like the same manager user. Until phase 5, `force` is recorded and changes nothing |
| `POST /api/move` · `POST /api/actions/{name}/run` · `GET /api/actions/{name}` | `submit_plan("mv", …)` / `submit_plan("run_action", …)` / the compiler's dry run | idle-only, refused 409 while a plan runs |
| `POST /api/calibration/check` · `/measure` | the two calibration plans | box-OFF only; `measure` takes `write` |
| `GET /api/events` | **SSE** | one stream: `status` (1 s poll), `progress` (from the doc bridge: scan number from the start doc, `planned_total`, `shots_done`, step), `log` (scan.log tail lines), `console` (manager text; `FAILED_MOVE_LOG_PREFIX` parsed into the paused reason) |

**Invariants the API must keep** (they are the whole point of `03` §4.D):
the client expands a preset; **nothing worker-side re-derives** detectors or
points. The scanner never resolves a variable to a device itself — it always
goes through `expand_preset` / the resolver. `md["geecs"]` is provenance only.
The scan number is claimed worker-side; the page reads it from the start doc.

### 3.4 The doc-stream bridge

One daemon thread per process, `bluesky.callbacks.zmq.RemoteDispatcher` on
`client.doc_addr`, reducing documents to a `Progress` model (`scan_number`,
`planned_total`, `shots_done`, `step`, `state`, `paused_reason`) — the
console's `scan_monitor.py` is the reference, adaptive totals included. A
second thread tails the manager console stream. Both write into one
process-wide cache the SSE handler reads; both are best-effort by design (a
dead stream degrades the `doc stream` health chip to `degraded`, never fails
a verb). The pickled wire format never leaves the process.

### 3.5 The page

Exactly the mock (artifact `d43b2a47-3bbc-4ad0-b002-3765b32b3a21`, 2026-09-13).
Shell: topbar (brand · experiment, health chips for qserver/gateway/tiled/doc
stream, operator, density, theme), rail (sections, presets, today's scans →
the portal's run pages), pane:

1. **Now** — the one panel the room watches: `chip.lg` state, plan line,
   determinate meter (shots, step, time left), live values with age (trigger
   box state, the scan axis readback, rate, one charge monitor, document
   age), Pause/Resume + Stop (rung 3 dialog), the `denied` banner when the
   running item belongs to someone else, the tail in a `.well` with a `.seg`
   toggle between **scan.log** (default) and the **manager console** text —
   two narrations, never reconciled.
2. **New scan** — mode `seg` (No-scan · 1D · Grid · Background · Optimize
   greyed), acquisition `seg` (strict default · gated), axis 1 (variable
   alias-first, start/stop/step with validation), axis 2 for grid, shots per
   step, trigger profile, shot period (strict only), description, the
   **devices table** (save images / essential / type / connected) = the
   preset made visible, "Add device…" drawer, "Save as preset…" drawer
   showing the YAML it becomes, footer estimate + Start.
3. **Queue** — running / waiting / today's finished with the `agent` chip on
   agent-submitted items, sticky header, Clear queue… (rung 3).
4. **Devices · move**, **Actions** (preview / arm / run; arming never
   persisted), **Calibration** (check sync / measure offsets, box-OFF only).

Dialogs: Stop, Take over (force), Clear queue — rung 3 as shipped; the
**preflight acknowledgement** is the one widened rung 3 (a list of questions,
each ticked; Submit disabled until all are). Keyboard: `P` pause/resume, `S`
stop, `N` new scan, `Esc` closes a drawer.

Mode → plan mapping, fixed:

| mode | plan | kwargs |
|---|---|---|
| No-scan | `count` | `num=shots_per_step` |
| 1D | `scan` (or `list_scan` when a values list is given) | axis, `shots_per_step` |
| Grid | `grid_scan` | two axes, `shots_per_step` |
| Background | `count` + `preset.background=true` | — |
| Optimize | *(none yet — §5)* | — |
| every mode | `acquisition`, `trigger_profile`, `shot_period` (strict), `non_essential` from the devices table | — |

### 3.6 Kit additions (GeecsWebTheme 0.3.0)

Found by building the mock; each is written in the kit's idiom (tokens only,
`.kit`-scoped, `data-state` vocabulary) and lands with a `kit.html` specimen
and its test in the same commit:

- `.live` — a reading with `.k` label, `.v` value (+ unit), `.age`;
  `data-age="stale"` past a threshold turns it warn-coloured and says
  "stale". The pane-level `stale` doctrine at value granularity.
- `.meter` — determinate progress: track, fill, two-ended label; takes
  `data-state` for the terminal colour.
- `.field[data-invalid="true"]` + `.err` slot + `.req` marker + disabled
  input styling. The kit has focus but no invalid/required.
- `dialog.ack` — the wider rung 3 that admits one form: a list of tickable
  questions. Still one decision, still no scrolling prose.
- `paused` joins `STATES` (warn colour, no pulse). A paused scan has no word
  today; "stopped by operator" renders `failed`-coloured with the text
  "stopped" — the attribute carries severity, the text carries the word.
- `denied` gets a rule (dashed edge, `--surface-2`); today it is
  named-but-neutral. Ownership refusal is its first real use.
- `.chip.lg` for the one state the room watches; `.tscroll.sticky` for
  live tables.

Not added: plots (the portal's), tabs, toasts. If one turns out necessary it
is its own small PR against `kit.html`, not an inline style.

---

## 4. Sequencing — six PRs, each with its own acceptance

All target `feature/native-bluesky-plans` **except PR 1**, which targets
master: GeecsWebTheme has no Bluesky dependency, its additions are additive,
and the portal and logbook on master can use them at once. One more
master → branch sync brings them here before PR 3.

| # | PR | contents | acceptance |
|---|---|---|---|
| 0 | *(prerequisite, in flight)* | master → branch sync: the kit arrives | `feature/web-surface-kit` content present on the branch; CI green |
| 1 | **GeecsWebTheme 0.3.0** — the kit additions (**targets master**; the next sync carries it here) | §3.6 + `kit.html` specimens + tests (`STATES` gains `paused`; the literal-colour walk gains the scanner's paths) | 127+ tests green; `kit.html` shows every addition; contrast test passes for `paused` |
| 2 | **GeecsScanner 0.1.0** — service layer + HTTP API | package, `service/`, `web/api`, the doc-stream bridge, SSE, `StubQueueClient`-backed tests; unit template, `render_units.sh` line, `site.env` keys, fleet-map row, DEPLOYMENT.md | **headless**: every route exercised against the stub in tests; on the box, `curl` against the real worker: status, configs, preflight of a real preset, `/api/events` streaming during a queued `count` |
| 3 | **GeecsScanner 0.2.0** — the page, day one | Now, New scan (count / scan / grid_scan), Queue, health chips, the four dialogs, keyboard; template guards | **hardware**: Sam submits a strict `count` and a strict 1D `scan` from a browser on the box; pause/resume/stop each observed once; the s-file matches the console-era shape |
| 4 | **GeecsScanner 0.3.0** — the rest of the mock | Devices · move (`mv`), Actions (preview/arm/run), Calibration, Add-device drawer, Save-as-preset (writes YAML through the resolver root), Today rail → portal links | each verb observed once on hardware; a saved preset round-trips through `GET /api/configs/presets/{name}` and submits |
| 5 | **Front door + operators** | Caddy on the worker host (`/portal`, `/log`, `/scan`), unit + site.env; an **operator registry** — `operators.yaml` in the experiment's configs tree (name, initials, optional role), plus the built-in `guest` — served by `/api/operators`; the chosen operator stored as `geecs.operator` in `theme-boot.js` beside `geecs.theme`, so the scanner's picker **and** the logbook's author field (GeecsLogbook patch: a picklist over the same registry, free text only for guest) read one value; ownership compare in `/api/stop` and `/api/pause` against the operator in the running item's `md["geecs"]` (the manager's `user` is the scanner's one identity); **and the shared web plumbing moves into GeecsWebTheme** — the forwarded-prefix middleware, the `root` context processor and the three template guards, copied verbatim by the portal, the logbook and the scanner (reviews of #871/#872) — as a `fastapi` helper and a `testing` module, once this fourth change makes three copies four | one origin serves all three; theme/density/operator follow between them; the logbook and scanner offer the same names; a foreign running item shows `denied` and needs `force` |
| 6 | **Delete GEECS-Console** — the closing PR | the package, `console-windows` CI job + its repo variable, docs pages, fleet-map rows, root `CLAUDE.md` row and dependency-graph entry; `git tag` a milestone first | CI green without the job; `docs/` builds; **the branch is now eligible for master** per the 2026-09-12 gate |

Each PR gets the `/land` ritual (scope, version + CHANGELOG, tests as CI runs
them, fresh-context adversarial review dispositioned, CI watch); PRs 3 and 4
carry a hardware-verification section that is *not* "not applicable". The
mock is the design of record for 3 and 4; iterate on the deployed page, not
on the mock.

**Not in this arc:** plots (portal), config editors (the `/configs` pattern),
the scan browser (portal), rewiring GEECS-MCP (disposable; it can later
become a client of `geecs_scanner.service` if wanted), a per-item queue
remove verb, auth beyond LAN + operator name (`#660` CurveZMQ stays open —
the front door is where TLS lands when it does).

---

## 5. Doors kept open — pseudo positioners and optimization

Both are owed on this branch (`09_pseudo_transform.md`; `03` §8 "optimization
is broken until it is re-glued to the native scan path in its own phase").
The scanner must not make either harder. Concretely:

**Pseudo scan variables.**
- The variable picker is fed by `GET /api/configs/scan_variables`, which
  returns the catalog **with `kind`**. Pseudo entries render disabled with
  the refusal text until the pseudo arc lands; then the flag flips and the
  page changes nowhere else.
- The page's axis model is `(variable, start, stop, step | values)` and the
  **page never binds a variable to a device**. `expand_preset` does. When
  the derived-signal noun exists, the same preset expands to it.
- Per-component confirm, if the pseudo arc wants one, is a preflight
  *question* — the acknowledgement dialog is generic over questions, so it
  needs no new UI.

**Optimization.**
- The API is **plan-name generic**: `POST /api/submit` takes a `Preset`
  whose `PlanCall` names any allowed plan. There is no mode enum in the
  API. An optimize plan becomes one more name in `GEECS_PLAN_NAMES` (and the
  operator group regex) with its own kwargs (`optimizer_config`,
  `max_iterations`), and the page gets one more form section keyed by plan
  name — the Optimize slot in the mode `seg` is already there.
- `GET /api/configs/optimizer_configs` ships in PR 2 even though nothing
  submits one yet, so the listing exists the day the plan does.
- **Progress reads `planned_total` from the start doc**, not from
  `steps × shots`: an adaptive plan reports `max_iterations` (the console's
  `scan_monitor.py` already handles this; the bridge keeps it). The meter's
  right-hand label degrades to "of ≤ N" for adaptive totals.
- The Now panel's live values are a list from the server, not fixed slots,
  so an objective readout can appear when an optimizer runs.
- Kept from the mock: nothing in the page assumes a scan has axes.

---

## 6. Open questions

None blocking. One to settle during PR 3 by looking at the deployed page:

1. Whether the Now panel's live-value slots are chosen server-side (the
   scan axis + the preset's essential scalars) or pinned by the operator.
   Server-side first; pinning is a later drawer if wanted.
2. ~~Which tail to show~~ — settled 2026-09-13: both, behind a toggle,
   scan.log default; the SSE stream carries them as separate event types.
