# The EPICS estate: repos, families, the web estate

Status: **proposal, provisional until built** (2026-10-25, after one
three-lens adversarial round; dispositions in §9). For the maintainer and the
LabVIEW developer. Nothing here is a contract; every "recommend" is a starting
point. Migration, host cutovers and gated runs are out of scope: the question
is what shape the estate should have so it is easy to grow, understand and
maintain, and how to reach a judgeable MVP without putting GEECS-Plugins'
master into an intermediate state.

Ruled by the maintainer, 2026-10-24/25: operator GUIs are **web**, modern and
dynamic; Phoebus is not part of the operator path; the cost of a custom front
end is accepted; browser code goes to a **new repository**; the MVP is the
full, organised codebase with one or two surfaces to open and click, judged
before anything lands in master.

## 1. What exists today

| Repo | Role today | Size |
|---|---|---|
| `GEECS-BELLA/geecs-epics` (private) | IOC instances, hosts, simulators, bench harness, docs, `CLAUDE.md` + `AGENTS.md` (#41), the `/epics-port` skill | 42 PRs since 10-21; `services/` still carries 21 bench scripts (~3k lines) and 15 evidence files |
| `ioc-motion`, `ioc-timing`, `ioc-plc`, `ioc-picoscope` (private) | generic IOC images, one per family, from epics-containers' `ioc-template`, each with a `CLAUDE.md` | built on the IOC host by hand (`:local`); CI publishes only on tag pushes; picoscope has no CI |
| `GEECS-Plugins` (public) | the Python services: scanner, portal, logbook, panels, MCP; the kinds and the registry schema; the shared web theme | per surface: about 2k lines of browser code over 3k to 7k of Python |
| `GEECS-Plugins-Configs` (public, deployed on the share) | `native_devices.yaml`, presets, scan variables, action plans | 2 native cameras |
| `GEECS-General-GUIs` (LabVIEW) | 49 device-type GUIs under `Device GUIs/`; the composed panels; `General-Controler` (the room's home: alarms, device enable status, last 10 commands, active preset); `GUI_Preset_Manager`; `GUI_DeviceVariablePlotter` and `LoggerPlotter` | the pattern the new estate keeps |

Facts that drive the proposal:

- **Two kinds of GUI, as in LabVIEW**, plus a third the first draft forgot:
  the room's home screen. One generic GUI per device type, composed panels,
  and `General-Controler`. All three get a layer.
- **The panels' Python is device knowledge, not boilerplate.** `GeecsPanels`
  shares its CA primitives, but its four feeds hold the rules that took real
  work (Appendix A): freshness judged by another PV's counter gated by a
  compute flag, worst-first state words over a dozen fields, drive-limit and
  lock gates that fail closed on unread context, server-kept history,
  decimated traces. About 2.4k lines. A generic value stream cannot carry
  them; the foundation keeps them server-side, deduplicated, not deleted.
- **The services are already API-first**, more than they look (verified):
  the scanner has 32 of 34 routes under `/api` plus SSE; the logbook 14 of
  22; the portal 15 of 25 with the run page composed server-side from a
  ~40-key context; the panels' 13 routes exist but are not namespaced.
- **Repo visibility is a constraint**: geecs-epics and the image repos are
  private; GEECS-Plugins and the configs repo are public and installed
  everywhere. A file a public package loads cannot live in a private image
  repo without a transport nobody has.

## 2. The web estate

### 2.1 Repositories, and the cut between them

**Browser code out; Python services stay.** A new repository, `geecs-web`,
holds everything rendered in a browser. `GEECS-Plugins` keeps every backend
and loses its templates surface by surface; each service ends as an API with
Pydantic models, a committed OpenAPI document and a JSON Schema for its SSE
events. The old Jinja pages and the new app talk to the same backends, so a
page migrates by being deleted, never ported.

```
geecs-web/                     React + TypeScript + Vite; one repo, several apps
  tokens/      the kit's colour tokens (pinned copy of theme.css, CI diff, until the file moves)
  ui/          PV-bound components: Readback, Setpoint, Enum, Bit, Command, Trace, Image, Journal
  api/         typed clients generated from each service's exported OpenAPI + SSE schemas,
               pinned to one GEECS-Plugins commit; regenerated in CI, fail on diff
  data/        hooks: device streams and the raw PV stream (SSE), queue state, Tiled; TanStack Query
  apps/panels  device pages from family + instance data, composed panels, the experiment home,
               the camera viewer, the PV inspector
  apps/scanner the scan console (phase 3)
  apps/...     logbook, portal, config editor, in that order, each on its own kill rule
  build.json   version, commit, and the service versions the clients were generated against
```

Stack: React, TypeScript, Vite; components styled **through the kit's tokens,
never a library's theme** (the one kit rule that survived three surfaces; the
no-literal-colour test becomes a stylelint rule on day one); Plotly where the
portal proved it, uPlot where rate matters; Playwright screenshots in CI,
since the maintainer judges by looking. Figma, if used, shapes the components
and tokens; it is a spec, never a code generator.

**Deployment, named as a deliverable.** A front door on the services box: a
Caddy (or nginx) unit rendered from `site.env` by `deploy/render_units.sh`
with one new key, the front-door host, every service under a path prefix
(the forwarded-prefix middleware already supports this), the bundle served by
the proxy from a directory a release artifact populates, never a clone.
`/fleet-status` reads `build.json` as the bundle's version, and the shell
compares the service versions in it with each `/health` at load and shows
the kit's `degraded` state on mismatch. The two absolute inter-service URLs
in `site.env` (the portal's logbook URL, the MCP's three) become paths or
stay absolute, stated per key. TLS: plain HTTP on the lab LAN for the MVP;
a lab CA is a later decision (Windows control-room browsers warn on
self-signed). The panels' "no deployment machinery until asked" rule is
revoked by this document: the PV bridge becomes a worker-host unit with its
own service account and `StateDirectory` in phase 1.

### 2.2 The layers

| Layer | What | Format | Home | LabVIEW analogue |
|---|---|---|---|---|
| 0 | **Family data**: one file per **record template** (motor record, AD camera chain, scope contract, `fastps`, `easydriver`, `dg645`, `modbus-plc`): the records by role (readback, setpoint with limits, enum, bit, command, array, image), labels, units, which are operator-level and which expert, which roles need a hazard confirm, which are archived | one YAML (or JSON) per template | **package data in `GEECS-Schemas`** (`geecs_schemas/epics/families/`), with its Pydantic model and loader: public, installed wherever the bridge, the scanner, the archiver and the MCP contract tests run. Each image repo's CI checks that every record its templates define is named by the family file: the drift alarm points from the image to the file | the driver's variable list |
| 0b | **Instance data**: per device, the family it instantiates, role overrides and bit labels where a family has no shared record set (the two supplies), the per-device `allow` list and operator note, the archive opt-ins | one file per experiment in the configs repo, `<experiment>/panel_devices.yaml` (today's `--bench` file, promoted) | the configs repo; read by the bridge only. A device may be listed here while the GEECS DB still owns it (a bench device); the worker never reads this file, so `native_devices.yaml` stays what the worker binds | the device's GEECS DB row |
| 1 | **Device-type page**: one page per device, rendered from family + instance data over that device's stream | spec, rendered | `geecs-web` apps/panels | `Device GUIs/GUI_<type>` |
| 2 | **Composed panels** (§2.4), and the **experiment home**: the registry roster with IOC liveness, the alarm list reduced from the device streams' severities, the bridge's put journal (who, what, when, accepted or refused), the active-snapshot verdict | one YAML per panel, `<experiment>/panels/*.yaml` in the configs repo, validated by a `GEECS-Schemas` model, served by the bridge at `GET /api/panels` | the configs repo | the composed panels; `General-Controler` |
| 3 | **Bespoke views where the interaction is the product**: the camera viewer, the scope traces, the shot-control state view | components | `geecs-web` | `GUI_Cameras`; the shot-control GUIs |

### 2.3 The PV bridge

A browser cannot speak Channel Access. The bridge is the Python service that
holds the CA connections and translates. It exists inside `GeecsPanels`
today, one feed per kind; it stays in `GEECS-Plugins` (it imports the
registry schema, the family loader and the EPICS environment bootstrap) and
becomes:

- **Device streams.** `GET /api/devices/{name}/events` is an `EventSource`
  (the browser reconnects by itself; `id:` cursors; full snapshot on connect,
  changed fields per tick after). It is produced by **one `Feed` class
  parameterised by the family file** for the monitored set and the roles,
  with **per-template Python for the derived rules**: freshness by the
  producing plugin's counter and compute flags, state words, the lock state,
  the server-kept history sampled on the counter tick, arrays read on
  request and decimated. Today's four feeds deduplicated into that one class
  plus per-template rule modules, not deleted. Panels consume device streams.
- **A raw PV stream** for the inspector and for composed-panel widgets that
  name a bare PV: `POST /api/subscriptions` creates a handle (a PV set;
  monitors ref-counted and shared across handles; evicted after a grace
  period with no listener), `GET /api/stream/{id}` is its `EventSource`,
  one frame per tick with changed PVs only.
- **Puts.** `POST /api/put` with device, field, value and the operator's
  identity. Legality has **three gates**, all kept: the IOC (access security,
  record limits, the refusal pattern of Appendix A rule 11); the bridge's
  per-device, context-reading allow-list (`check_write(field, value,
  snapshot)`: 403 for not-permitted-on-this-device before 422, fail closed on
  unread limits or severities); and the spec, which decides only what a page
  *offers*. A refused put is the record's alarm, never a success. **Every put
  is journaled** (operator, device, field, value, result, time); the journal
  is a route and the experiment home's "last commands" list.
- **Writes from a browser**, decided here, not later:
  1. *Identity.* The scanner's free-text operator field becomes a required,
     per-browser identity sent with every put and journaled. It is not
     authentication; it is who-did-what, which the room never had.
  2. *The IOC sees one account.* The bridge writes as its own service
     account; the IOC's access files grant that account and the worker's.
     The per-user rules the DG645 and PLC bench ports built were bench
     scaffolding and are retired with them; per-operator policy lives in the
     bridge, from instance data. CA client names are spoofable on the lab
     network anyway, so this loses no security, only the illusion of it.
  3. *Hazard confirm.* A role marked `confirm: hazard` in the family or
     instance data (the DG645 trigger source and inhibit, PLC coils, a
     supply's output, a motor's home, SINGLESHOT) opens a rung-3 `<dialog>`
     naming device, field and value before the put.
  4. *A stale page cannot write.* The bridge refuses a put whose device
     stream is older than a threshold or disconnected; the page greys its
     controls the moment its stream drops (Appendix A rule 13 covers dirty
     fields; this covers a page left open overnight).
  5. *Two tabs.* A dirty field whose readback moved shows it; motion fields
     show "moving by someone else"; setpoints are last-write-wins.
- `GET /api/devices` (registry + instance data), `GET /api/families`,
  `GET /api/panels`, `GET /api/frame/{device}` (one upstream per camera, one
  frame in flight), `GET /api/pv/{name}` (the inspector), `GET /api/journal`.
- Appendix A lists the rules the bridge keeps. Its size is the existing 2.4k
  lines reorganised plus the routes above, not the 500 the first draft said.

### 2.4 Composed panels: what the spec may and may not say

So the spec never becomes a language:

- **Layout**: named sections, each a grid; a `device` widget embeds a whole
  layer-1 device page and a `component` widget a layer-3 component (the
  camera viewer) inside a composed panel (`geecs_schemas.epics.panels`).
  The LabVIEW composed panels mostly embedded device GUIs.
- **Derived values are IOC records** (calc, mbbi; the pneumatic-actuator
  note in `services/plctest1/README.md` is the pattern), never spec expressions.
- **Multi-step buttons are an `action` widget naming an action plan**, run
  through the scanner's API, idle-only, never spec logic.
- **Conditional visibility** has one declarative form, `show_when: {pv,
  equals}`, and nothing more.
- **Per-experiment overrides of a family page** exist and are small:
  `<experiment>/panels/overrides/<family>.yaml` may hide roles, relabel and
  reorder; it may not add fields or logic.
- **Trends**: a `Trend` widget over device streams (§2.5), PVs picked on the
  page, two Y axes, last N minutes, scan marks.

The MVP's composed pages (§6): the **camera grid** (N embedded viewers, each
with its saturation chip and stale state; it stresses the bridge's fan-out,
`embed` and the layout model at once) and **one page that writes**: a motor,
a supply and a snapshot restore, through the bridge with identity and
confirm. The every-supply page is a third if time allows.

### 2.5 Trends, history, alerts

- **Live trends are the bridge's, not Grafana's.** The LabVIEW
  `DeviceVariablePlotter` is a live subscription plotter at the device's
  rate; the archiver, by design (#1114's deadband), discards exactly the
  jitter an operator watches, and today it archives nothing native: the
  archive set is derived from the GEECS DB, and a migrated device is
  disabled there. Two consequences, both phase 1: the `Trend` component over
  the device stream's server-kept history (minutes, at rate), and
  **`geecs-archiver` learns the registry and the family files' archive roles**
  (the per-kind rule `NATIVE_DEVICES.md` §10 already proposes), or the first
  migrated supply has no history at all.
- **Grafana over the Archiver Appliance** is the engineer's tool for longer
  than an hour, and a deploy slice of its own: a unit or compose rendered
  from `site.env` beside the appliance, image and plugin version pinned
  (`archiverappliance-datasource` 2.2.0, 2026-10-22, one external
  maintainer on a Grafana ↔ plugin ↔ appliance upgrade triangle: a known
  cost), its admin password and an embed token in the secrets home (§5),
  anonymous viewer access on the lab LAN or embed tokens, decided at deploy.
  Dashboards provisioned from JSON in git. **Scan marks** come from an
  archived PV, the shot-control profile state the gateway already serves,
  not from a new consumer of the document stream.
- **Alarms**: the experiment home lists them from the device streams'
  severities today; Grafana alert rules notify; acknowledge and annunciate
  are designed when a supply first faults in production.

### 2.6 Persistence: settings, snapshots, presets

Three layers; one is new. Last state across an IOC restart is autosave,
already on every instance; **named per-IOC setting sets stay autosave
configMenu**, EPICS-native, no code. History is the archiver. What neither
covers is the LabVIEW Preset Manager's job: a **beamline-level snapshot
across many devices and IOCs**. That becomes a snapshot document, and the
restore semantics are specified now because they are where it is dangerous:

- A snapshot is a named set of setpoints with the readbacks at the time,
  author, note, optional golden mark, scoped by a composed panel's device set
  or a family's writable roles. Schema in `GEECS-Schemas`.
- **Restore is a diff view with a tick per row**, never whole-scope by
  default; unreachable devices are grouped, skipped and reported, never a
  failure of the whole; **motion rows** (a setpoint role on a motor field)
  sit behind the hazard confirm with Stop visible, applied sequentially and
  waited on (`DMOV`); rows whose family marks a write as cycle-disturbing
  (the DG645 delay) carry the same confirm; "stop on first refusal" or
  "continue" is chosen before the run; each row ends accepted or refused by
  its record's alarm.
- **Compare**: snapshot against live, snapshot against snapshot; the
  **active-snapshot verdict** (every row within tolerance of live) is what
  the experiment home shows as "which preset are we in".
- **One authoritative copy**: the bridge's `StateDirectory`. The
  `{experiment}/snapshots/` mirror on the data tree is a derivative under the
  logbook's own-tree guard (never under `scans/`); golden snapshots are
  promoted into the configs repo by PR and are what a scan-time action plan
  reads, live after the share checkout's pull like every config.
- geecs-epics **decision 3 is amended** in the same arc: named setups are
  autosave configMenu per IOC and the snapshot document across IOCs; Phoebus
  Save & Restore is struck. Decision 6's wording gains "family data and panel
  policy are GEECS-side".

## 3. Families: the unit is the record template

| Template | Image | Kind | Family data source today | State |
|---|---|---|---|---|
| AD camera chain | upstream `ioc-adaravis` | `aravis_camera` (built) | the chain contract; `camera.py` | two instances |
| motor record | `ioc-motion` | `epics_motor` (#1142) | `motor.py` (#1143); stock fields | ESP302 proven; **the template that settles the file's shape** |
| `dg645` | `ioc-timing` | setpoint device (unbuilt) | upstream `drvDG645.db` | DG645 proven |
| `modbus-plc` | `ioc-plc` | setpoint device | `plctest1.substitutions` + DB aliases (#39) | U_PLC proven |
| `fastps`, `easydriver` | upstream `ioc-streamdevice` + protos in `services/` | setpoint device | `power.py` (#1145), per-device role maps → instance data | both proven |
| scope contract | `ioc-picoscope` (vendor SDK, host-built) | array detector (unbuilt) | the scope contract; `scope.py` | 5242D proven |
| **shot control** | a `dg645` instance | profile states (`ShotControl`, owned by the worker during a run: the run-lock pattern) | `shot_control.py` | layer-3 state view; SINGLESHOT behind the hazard confirm; ported last |
| EPS / interlocks | — | — | — | stays LabVIEW; out of scope for this estate, written here so it is not forgotten |
| later: HV, spectrometers, WFS | | | | |

Family ≠ kind ≠ image: three templates share the setpoint-device kind, and
the two supplies share an image and a kind but not a record set. The kind's
suffix table (`NATIVE_DEVICES.md` §10) is derived per template. No
`ioc-supply` repo: the family files no longer need an image repo to live in,
and the StreamDevice protos stay where the instances are until a second
instance of a supply exists, when they move to a shared folder in geecs-epics.

## 4. The repos, as the sysadmins would want them

| Repo | Holds | Does not hold |
|---|---|---|
| `geecs-web` (new) | tokens, components, generated clients pinned to a GEECS-Plugins commit, data hooks, the apps, `build.json`, screenshots CI, its own `CLAUDE.md` | Python, PV access, IOC config, specs |
| `GEECS-Plugins` | the Python services as APIs (scanner, portal, logbook, the PV bridge, MCP), the kinds, the registry schema, **the family files**, `theme.css` until it moves, the front door in `deploy/` | templates and static pages (removed surface by surface) |
| `GEECS-Plugins-Configs` | `native_devices.yaml`, `panel_devices.yaml`, `panels/*.yaml`, golden snapshots | anything else EPICS |
| `ioc-<family>` | Dockerfile, `ibek-support-geecs/<module>/`, README "what init sends", `CLAUDE.md`, the CI check against its family file | instances, hosts, bench scripts, family data |
| `geecs-epics` | `services/<ioc>/config/`, `hosts/`, `sim/`, `bootstrap/`, `docs/`, skills, `scripts/` | drivers, family data, panel specs, bench harness, evidence files |

Still to do in geecs-epics after the cleaning pass (#39 to #42): the
per-device bench harnesses (~3k lines) consolidated into one generic guard
(a `connected` watcher with a safe-off hook) or moved to `bench/<ioc>/` as
dated records; evidence files to PR comments; PVA bound to loopback on
dgtest1, fpstest1 and esptest1 as on plctest1; dgtest1's watcher reading
severity; an image pin for picotest1; the compose header's bare `up -d`
line; the intermittent trigger-box sim test.

**Images.** CI publishes only on tag pushes and picoscope has none; compose
pins commits while CI publishes tags. Proposed: the three non-vendor images
move to tag releases with **public packages** on the registry (Apache and
synApps code only; the repos may stay private), compose pins the published
tag, picoscope stays the one `:local` exception with its source commit in
compose. Public packages avoid the estate's first registry secret on hosts
that have no GitHub credentials.

## 5. The maintainability kit, transferred

| In GEECS-Plugins | geecs-epics now | Proposed |
|---|---|---|
| `CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING.md` | `CLAUDE.md`, `AGENTS.md` (#41); per `ioc-*` repo too | a `CONTRIBUTING.md` only when a second human commits |
| `scripts/check.sh` + CI | `check_registry.py` only; no CI | `check.sh`: every `ioc.yaml` against its image's ibek schema, `compose config` per host, sim tests, the registry check, the port register derived from compose; GitHub Actions |
| pre-commit | none | yaml, shellcheck, ruff on `sim/` and `scripts/` |
| skills | `/epics-port` | `/ioc-fleet` (`fleet-status` analogue: host vs compose, tags vs repos, checkout vs main), `/sim-bench`, `/deploy-host`; `/land` as is |
| `doc_audit.py` | none | catalogue vs `services/` vs image repos: every instance a row, every row an image |
| versions at deploy | `:local` by commit comment | registry tags; geecs-epics by the deployed commit |
| `site.env` | `bootstrap/site.env` | the same; Grafana and the front door rendered from it |

`geecs-web` gets the kit in its own idiom: `CLAUDE.md`, lint and typecheck,
unit tests, Playwright screenshots on every PR, the token diff, stylelint's
no-literal-colour rule, generated clients regenerated in CI from the pinned
GEECS-Plugins commit and failing on diff. In `GEECS-Plugins`:
`scripts/export_openapi.py` emits each service's OpenAPI document and its
SSE event JSON Schemas; `unit-tests.yml` fails when a route changed and the
committed document did not. Composed-panel specs are validated by
`GEECS-Schemas` against the installed family files, so the check runs where
the schemas are. `scripts/images.py` in geecs-epics reports drift across the
image repos (template version, `ibek-support` pin, base tag, the tag compose
uses; four repos already disagree on the `ibek-support` pin).

**The estate's first secrets** (a registry pull token if packages stay
private, Grafana's admin password and embed token) need a home the one-home
rule does not have today: a root-only `/etc/geecs/secrets.env` beside
`site.env`, the `DATA_CREDENTIALS` pattern, read by the units and never
rendered into files.

## 6. Reaching the MVP without touching master

1. **`geecs-web` merges freely to its own main.** It has no users.
2. **The existing stack merges first.** #1142 to #1145 land on master
   before anything starts (the maintainer's call, 2026-10-25); their
   templates are the first thing the kill rule deletes. `feat/web-estate`
   is cut from that master, so no hunk lives in two lines.
3. **The integration branch is small and lives only as long as the push.**
   Phase 0 and 1 only (the OpenAPI export, the bridge, the front door
   template). The work is an agent-parallel push of a few days (§6a), so the
   branch should live days; if it is still open after two weeks it is
   merged or killed, not synced. The last long-lived branch in this repo
   (`feature/native-bluesky-plans`) cost a 24-conflict sync PR; the
   `feat/pva-capture` branch lived two days. This one must look like the second.
4. **The MVP runs from the branch** on the dev box or the services box
   against the simulator host, as the worker ran detached at a branch tip before.
5. **The gate**: `geecs-web` complete as a codebase, and surfaces to open and
   click: the panels app with the camera viewer, the camera grid, the
   experiment home and **one composed page that writes** (a motor, a supply,
   a snapshot restore, with identity and confirm); the scanner app as the
   second surface. The plotter is not counted: it stays on its own brief
   as the LabVIEW developer's onboarding project and moves to `geecs-web`
   with the portal in phase 4.
6. **After a positive verdict**, the branch's additive routes land on master
   as ordinary PRs (they are additive; nothing on master breaks), and
   **each surface's deletion is its own PR** under the kill rule: when the
   new version is preferred by looking, the old templates, its `_SURFACES`
   entries, import-linter contracts, docs pages and CHANGELOG row go in one
   bounded PR. There is no bulk promotion PR; the repo's own size rules forbid it.
7. **After a negative verdict**, the branch and the repo are deleted and
   master never knew.

### 6a. The push, as lanes

The MVP is reachable as one agent-parallel push on the EPICS arc's lane
model (interfaces frozen first, three to five builders, the maintainer
merging `geecs-web` and the branch on the geecs-epics rule). What makes it
short is that every piece exists in some form: the feeds, the kit, the
scanner's API, the sim host, the panel bodies of #1143 to #1145.

| When | Lane | Deliverable | Depends on |
|---|---|---|---|
| hour 0–2 | S | `GEECS-Schemas`: family-file model + loader, the motor, camera and `fastps`/`easydriver` files transcribed from `motor.py`, `camera.py`, `power.py`; instance-data and panel-spec models; the device-stream event schema frozen | nothing |
| day 1 | B | the bridge: one `Feed` over the family file + per-template rules (the 2.4k lines reorganised), device and raw streams, puts with identity and journal, routes, unit | S |
| day 1 | W0 | `geecs-web` scaffold: tokens, CI, screenshots, clients from the exported documents | `export_openapi.py` (lane F) |
| day 1 | W1 | components and the device-page renderer against the frozen event schema; the camera viewer ported | S, W0 |
| day 1 | F | `export_openapi.py` + SSE schemas + CI diff; the front-door template and unit | nothing |
| day 2 | W2 | composed panels: the camera grid, the experiment home with the journal, the writing page (motor, supply), snapshot v0 (save, diff, per-row restore) | B, W1 |
| day 2 | R | the archiver's registry pass; the `Trend` component over device history | S |
| day 3 | W3 | the scanner app over its API and SSE | W0 |
| day 3 | gate | everything on the sim host behind the front door, judged by looking | all |

**Front end first, against frozen contracts.** `ui/` and the apps are
developed as a self-contained front-end package: a gallery page (the kit's
`kit.html` in the new idiom) shows every component in every state, and a
**fixture server replays recorded device-stream frames** (a minute per
device from the sim host; this is decision 13's PV-level replayer, now with
a consumer), so the web lanes never wait on the bridge lane and a reviewer
can judge a page with nothing else running. Only three contracts are
frozen first and owned by the schemas lane: the family-file model, the
device-stream event schema, and the put request and result. Everything
behind them (the bridge's internals, the front door, the archiver pass) can
move without touching a component. The package is what the LabVIEW
developer, or another BELLA group, would reuse.

A clickable panels foundation (camera grid, motor and supply pages, the
inspector) is a day-one outcome; the gate as written is about three days
of wall time, bounded by the maintainer's merge and review throughput
rather than by agent capacity, as the EPICS arc showed. The branch's
lifetime is that push.

## 7. Phases and the GEECS-Plugins rework

| Phase | geecs-web | GEECS-Plugins (branch until the verdict, then master) | Deleted, each in its own PR |
|---|---|---|---|
| 0 scaffolding | repo, CI, tokens, generated clients, screenshots | `export_openapi.py` (+ SSE schemas) and its CI diff; the front-door template and unit in `deploy/`; the secrets home | nothing |
| 1 foundation on panels | components, device-stream and raw-stream hooks, device pages from family + instance data, composed panels incl. the experiment home and the camera grid, the camera viewer, the PV inspector, the writing page; acceptance on the sim host | the bridge: one parameterised `Feed` + per-template rules (the 2.4k lines reorganised), device and raw streams, puts with identity and journal, the routes of §2.3, a unit with a service account and `StateDirectory`; family files + loader + instance-data and panel-spec models in `GEECS-Schemas`; the archiver's registry pass | GeecsPanels templates and static |
| 2 scanner | the scanner app over the existing API and SSE | type the two or three routes still returning dicts | scanner templates |
| 3 logbook | the logbook app | JSON for the day and month views | logbook templates |
| 4 portal + plotter + editor | the portal app, the plotter moved in, the editor app | the portal's run page decomposed into routes (the largest slice, 1k to 2k lines); the notebook-snippet endpoint; a JSON editor API in `ScanAnalysis` | portal templates and the 971-line script; `GeecsPlotter/` if it landed as Jinja; the editor's templates |
| 5 retirement | component docs carry the kit's rules (status words, overlay ladder, containers by role) | `theme.css` moves to `geecs-web`; repo map, graph, import-linter, `check.sh` updated per deletion | `GeecsWebTheme`'s kit layer and its surface-walking tests |

About 12k lines of browser code replaced over time; about 19k lines of
service Python kept minus template glue. `GEECS-MCP`, `GeecsBluesky` and the
archiver's appliance deploy are untouched; the MCP contract tests keep
mounting the apps in process and benefit from the stricter APIs.

## 8. Decisions this asks for

1. `geecs-web` as the repository name; React/TypeScript/Vite.
2. The branch rule of §6.2–3: the panel stack merged first, a
   `feat/web-estate` for phases 0–1 that lives for the push (two weeks at
   most), deletions one PR per surface.
3. Family files as `GEECS-Schemas` package data (public); no `ioc-supply`.
4. Instance data (`panel_devices.yaml`) and composed specs in the configs
   repo, read by the bridge only.
5. The browser-write model of §2.3: one bridge account at the IOC,
   per-operator policy in the bridge, a required operator identity, the
   journal, hazard confirms from the family data.
6. Decision 3 and 6 amendments in geecs-epics in the same arc.
7. Public packages for the three non-vendor images, tag releases, compose
   pins tags.
8. TLS: plain HTTP on the lab LAN for the MVP.
9. Bench harness: `bench/<ioc>/` as dated records, or delete.
10. The second surface at the gate: the scanner (recommended); the plotter
    stays on its brief and is not counted.

## 9. Review dispositions (round one, 2026-10-25)

Three fresh-context reviewers (architecture, operations, the operator and
LabVIEW parity), 24 findings. **Accepted in full**: the experiment home,
alarm list and put journal (O1); live trends over the bridge and the
archiver learning the registry (O2); restore semantics and decision 3 (O3,
A5); the browser-write model (O4, A2); composed-spec rules and the camera
grid as the MVP page (O5, O7); the shot-control and EPS rows (O6); family
files in `GEECS-Schemas`, no `ioc-supply`, family = template, motor record
first (A1, A8); instance data as a layer and three legality gates (A2);
device streams from a parameterised `Feed`, the bridge's real size (A3);
subscription handles, `EventSource` with cursors, coalescing, eviction (A4);
composed specs in the configs repo (A6); pinned-commit client generation,
SSE schemas, the editor API in ScanAnalysis (A7); the wrong branch precedent,
the short branch and per-surface deletions (P1, P7); the #1142 ordering
(P2); the front door as a deliverable with `build.json` (P3); tag releases
and public packages (P4); Grafana as its own slice, scan marks from an
archived PV (P5); one authoritative snapshot copy (P6); the version lattice
and the secrets home (P8). **Accepted in part**: O8 asked to keep the
plotter off the new stack entirely; it stays on its brief and is not
counted toward the gate, but moves to `geecs-web` in phase 4 rather than
remaining a permanent Jinja surface. A5 asked to weigh autosave configMenu
as the snapshot store; it is kept as the per-IOC store, and the cross-IOC
snapshot document stands because configMenu cannot span IOCs.

## Appendix A. What the bridge must keep

Rules the panels learned in their first three days, each pinned by a test on
`feat/panels-power-supply` (the coordinator's list, 2026-10-25). The bridge's
per-template rule modules and the components carry them forward; a rule
without a test in the new repo is not carried.

1. Freshness is judged by the producing plugin's `ArrayCounter_RBV`, never by
   a value changing; a value whose `Compute*` flag is off is `off`, not fresh
   (`test_feed.py`; `NATIVE_DEVICES.md` §9 on `ComputeStatistics` / `ComputeCentroid`).
2. One upstream per camera, one frame in flight, latest wins, viewers share
   (`test_concurrent_viewers_share_one_upstream_request`).
3. Arrays (profiles, histogram, traces) are read on request, never monitored
   (`test_plots.py`).
4. Positions are image pixels, not JPEG pixels: ffmpegServer rescales
   (`test_a_click_is_the_pixel_under_it`, `test_roi_positions_map_to_image_pixels`).
5. Saturation is judged on the raw frame, `STAT0`, before background and ROI.
6. The run lock: an unreadable lock refuses acquisition writes; a locked
   camera answers 423 naming the run; only the override route releases, and
   logs who (`test_lock.py`).
7. The stream URL is rehosted to the CA address (the IOC names itself by a
   hostname a laptop cannot resolve).
8. Writes are allow-listed server-side with context, `check_write(field,
   value, snapshot)`; 403 for not-permitted-on-this-device before 422.
9. Disconnected is not stale: a dropped field reads `None` and the state is
   `unknown`; a known fault outranks a dropped field (`test_motor.py`, `test_supply.py`).
10. NaN travels as `null` or the SSE stream dies
    (`test_a_nan_reading_travels_as_null_and_the_stream_lives`).
11. Setpoint refusal lives in the IOC: an `ao` clamps beyond `DRVH` and
    writes; refusal is `DRVL`/`DRVH` plus `HIHI`/`LOLO` at the ceiling ± one
    step with INVALID severity and IVOA, set at load, never by `dbpf`. The
    page's own gate fails closed on unread limits, bounds or severities
    (`services/fpstest1/README.md`, `test_supply.py`).
12. Motor record: soft-limit refusal is stock ophyd-async before any put and
    LVIO at the IOC; put-callback completes at motion end for motion fields
    and at once for settings; the post-move confirm reads uncached; operator
    level ASL0 also covers SET/OFF/CNEN/JOG, so per-device allow lists matter.
13. An edited field stays dirty until sent or Escape; a stream push must not
    resend an old value (#1144, 80c48b466).
14. A bench or registry name must be one URL segment.
15. Access security trusts the client-reported user over CA and PVA; PVXS
    serves records over PVA whatever the environment says, so bind PVA to
    loopback where only CA should write (#37).
16. Network: the gateway re-dials a down device at most every 30 s, so
    `connected` lags; CA beacons do not cross the VPN, so clients that
    searched while a server was down back off slowly; PVA through Docker
    Desktop needs the TCP name server.
17. A supply panel shows ringing: the FAST-PS current peaks 0.75 A on a 0.5 A
    step regardless of slew; slew 10 A/s reverts at power cycle (geecs-epics #38).
18. A DG645 delay write can abort the delay cycle in progress on every output
    of the box (`services/dgtest1/opi/README.md`): a cycle-disturbing write,
    behind the hazard confirm.

Delete this file in the last per-surface deletion PR (`Planning/README.md`):
by then `geecs-web/CLAUDE.md`, `GeecsPanels/CLAUDE.md` and
`geecs-epics/docs/architecture.md` carry the picture.
