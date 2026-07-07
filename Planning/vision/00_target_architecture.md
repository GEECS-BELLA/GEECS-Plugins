# Target architecture — the guiding star

Status: **vision document** (2026-07-07). This is what the system looks like
when the EPICS/Bluesky commitment is complete — a direction to steer by, not
a step-by-step plan. Nothing here is a promise or a schedule. When a design
decision comes up, the question this doc answers is: *does the change move us
toward or away from this picture?*

The one-sentence thesis:

> **GEECS devices are IOCs. Scans are plans. Configs are schemas. Front-ends
> are clients.**

Everything below is that sentence applied to each part of the system.

---

## 1. The layer cake

```
┌─────────────────────────────────────────────────────────────────┐
│ FRONT-ENDS (replaceable clients)                                │
│   Scanner GUI (thin: editors + run console)   Phoebus displays  │
│   notebooks / scripts          (future: web console, qserver)   │
├─────────────────────────────────────────────────────────────────┤
│ ENGINE  (geecs_bluesky — one engine, no bridge)                 │
│   GeecsSession = run discipline (numbering, paths, Tiled,       │
│   shot control, roles) · plans (step/free-run/single-shot/      │
│   optimize) · CA device family · typed event stream ·           │
│   OperatorChannel (dialogs)                                     │
├─────────────────────────────────────────────────────────────────┤
│ SCHEMAS (versioned Pydantic models; YAML is just serialization) │
│   ScanRequest · SaveSet · ScanVariables · TriggerProfile ·      │
│   ActionPlan · Diagnostics/Groups · OptimizationSpec            │
├─────────────────────────────────────────────────────────────────┤
│ DATA   (Tiled catalog + NetApp files + event schema v1)         │
│   runs in Tiled · native files joined by acq_timestamp          │
│   (geecs_data_utils.native_files) · analysis results as         │
│   derived Tiled runs · legacy s-files as an export              │
├─────────────────────────────────────────────────────────────────┤
│ ACCESS LAYER (GeecsCAGateway — the ONLY GEECS-protocol speaker) │
│   caproto CA server · PV_CONTRACT.md is the API · CONNECTED     │
│   liveness · :SP setpoints ride blocking sets · systemd service │
│   → Archiver Appliance (curated PVs) · → image sidecar (view)   │
├─────────────────────────────────────────────────────────────────┤
│ GEECS (LabVIEW devices, MySQL device DB, native file saving)    │
└─────────────────────────────────────────────────────────────────┘
```

Two rules make the cake hold its shape:

- **Nothing above the access layer speaks GEECS TCP/UDP.** If a feature needs
  a new fact from a device, the gateway learns to serve it as a PV (or PV
  metadata) and the contract documents it. This week's liveness work is the
  template: the scanner didn't grow a TCP client to ask "are you alive?" —
  it read the `CONNECTED` PV the gateway already served.
- **Nothing below the front-ends imports a front-end.** The engine emits
  typed events and asks questions through an injected channel; it never knows
  whether a Qt dialog, a web page, or a log line answers.

## 2. One engine, no bridge

`BlueskyScanner` exists to serve two masters during the transition. In the
target picture it dissolves:

- **`GeecsSession` is the engine.** The GUI (or any client) builds a
  `ScanRequest` (§4.1) and calls `session.run(request)`. What remains of the
  scanner is a small `SessionRunner` utility — a thread, an abort flag, and
  event fan-out; ~200 lines with no scan knowledge of its own.
- **The event vocabulary moves down.** `ScanEvent`, `ScanStepEvent`,
  `DialogRequest` etc. are engine concepts that today live in
  `geecs_scanner` (a front-end package) for historical reasons, forcing the
  engine to import its own vocabulary defensively. They move to the engine
  (or a small neutral package); front-ends import them from below. The
  try/except import dance disappears.
- **Operator interaction is one seam.** Today's pre-flight dialogs, refire
  policies, and future "are you sure" moments all flow through a single
  injected `OperatorChannel`: `ask(question, options, default, timeout) →
  answer`. The GUI binds it to Qt dialogs; headless binds it to
  default-and-log; a web console binds it to a websocket. The engine never
  contains dialog plumbing again — it contains *questions*.
- **Pre-flight is a pipeline, not methods.** Liveness (CONNECTED), trigger
  running (free-run), disk reachability, config sanity — declarative checks
  that run before the claim, each yielding pass / ask-operator / abort. New
  checks are additions to a list, not new branches in a 1500-line class.

## 3. What the commitment deletes

The deletion is the unlock, not just a cleanup. When the legacy engine goes:

- `ScanManager`, `DataLogger`, `DeviceManager`, `ScanStepExecutor`,
  `FileMover`, `TriggerController`, `DeviceCommandExecutor` and their tests —
  the entire parallel engine.
- **GEECS-PythonAPI retires.** Its device layer (TCP subscriptions,
  `ScanDevice`) served the legacy engine; its DB layer already moved to the
  gateway. What survives is only whatever config-ini reading hasn't already
  migrated to `geecs_data_utils`.
- The GUI's dual-backend seams (`RunControl`'s two constructors,
  `GEECS_USE_BLUESKY`) — one engine, no toggle.
- The event types' defensive imports, the bridge's duck-typed protocol
  (`device_requirements`, `on_finish`, …) — replaced by declared interfaces
  once there is exactly one consumer chain.

**Gate, not date:** deletion happens when a written parity checklist is done
(setup/closeout actions on the Bluesky path, background scan mode decision,
MultiScanner/preset compatibility, ECS dump story) **and** some agreed period
of routine production scanning never reached for the legacy toggle. Evidence-
gated, like everything else this month.

## 4. The schema redesign

The current config landscape grew one YAML dialect at a time: save elements,
scan devices + composite variables, timing configs, action library, presets,
optimizer configs, diagnostics. The diagnostics/groups redesign (unified
`image:`/`scan:` schema, Pydantic-validated, one loader) is the model for
what the rest become. Principles:

- **Pydantic-first.** Every config is a versioned model (`schema_version`)
  in one home; YAML is serialization. Loaders validate on read; GUI editors
  and scripts share the same models; migration converters live next to the
  schema they migrate.
- **Declare intent, derive mechanics.** Configs written for the legacy
  engine encode *how* (force-appended variables, synchronous flags, per-state
  write matrices). Bluesky-era configs declare *what*, and the engine derives
  the rest — the auto-provisioning and role-classification work already
  showed the derivations are computable.
- **Device facts live below the configs.** Limits, units, tolerances, and
  choices belong to the GEECS DB and should surface as **PV metadata from the
  gateway** (EGU, DRVL/DRVH, enum strings — the CA-native way). Client YAML
  then stops repeating facts the control system already knows, and Phoebus
  gets them for free.

### 4.1 ScanRequest — the one submission object

Everything a client submits is one model. Presets are saved ScanRequests;
MultiScanner is a queue of them (and maps naturally onto queueserver later);
`ScanInfo` and run metadata are projections of it.

```yaml
schema_version: 1
mode: step            # step | noscan | optimize
variable: jet_z       # name from ScanVariables (absent for noscan)
positions: {start: 4.0, end: 6.0, step: 0.5}
shots_per_step: 10
acquisition: free_run # free_run | strict
save_set: undulator_baseline      # name of a SaveSet
trigger_profile: htu_laser_off    # name of a TriggerProfile
actions: {setup: [pre_scan_ebeam], closeout: []}   # named ActionPlans
description: "jet z scan with probe"
# optimize mode adds an optimization: block (VOCS, objective, generator)
```

### 4.2 SaveSet (today: save elements)

A device entry declares what to record; roles and bookkeeping variables are
derived (first sync device = free-run reference unless overridden;
`acq_timestamp` is always implicit — the device layer already enforces this).

```yaml
# before (legacy save element)              # after (SaveSet entry)
UC_Amp4_IR_input:                           - device: UC_Amp4_IR_input
  synchronous: true                           images: true
  save_nonscalar_data: true                   scalars: [MaxCounts, centroidx]
  variable_list: [acq_timestamp,            - device: U_HP_Daq
    MaxCounts, centroidx]                     scalars: [ch1]
                                              role: snapshot   # only overrides
```

Device-name spelling is validated against the DB at load (the case-drift
lesson, 2026-07-06), not at first CA timeout.

### 4.3 ScanVariables (today: scan_devices.yaml + composite_variables.yaml)

Scan variables become declared positioners rendered by a factory to
`CaSettable` / `CaMotor` / `CaCompositeSettable` (the composite = a pseudo-
positioner with forward/inverse expressions — same math as today's numexpr,
now living where motion abstractions belong). Limits/units come from gateway
PV metadata; YAML adds only what GEECS doesn't know:

```yaml
jet_z:
  target: U_ESP_JetXYZ:Position.Axis 3
  kind: motor            # blocking move + readback tolerance
e_beam_energy:           # composite
  kind: pseudo
  targets: [U_EMQ1:Current, U_EMQ2:Current]
  forward: "..."         # expressions, as today
  inverse: "..."
```

### 4.4 TriggerProfile (today: timing/shot-control configs)

The state *names* (OFF / STANDBY / SCAN / SINGLESHOT / ARMED) proved to be
the right abstraction — they survive. What changes:

- A profile is a validated model; the laser-on/off duality becomes explicit
  **profile variants** (`armed: external` vs `armed: internal`) instead of
  parallel look-alike YAML files.
- **Trigger state becomes EPICS-visible**: the gateway (or a thin timing
  helper on top of it) exposes a `TRIG:STATE` enum PV that applies a
  profile's writes atomically. Then Phoebus shows and sets the machine
  trigger state with a button, the engine's `ShotController` becomes a
  one-PV client, and "what state is the trigger in?" has one answer for every
  tool — instead of each client privately replaying a write-matrix.

### 4.5 ActionPlans (today: action library + setup/closeout)

In a Bluesky world, an action sequence *is* a plan. The action YAML (steps of
set / wait / read-and-check) compiles to plan stubs executed by the same
RunEngine — which means actions get abort, logging, event emission, refire-
style policies, and OperatorChannel questions for free, instead of the
legacy ActionManager's parallel executor. `setup:`/`closeout:` in a
ScanRequest are just named ActionPlans run inside the scan plan's preamble/
finalizer (the hook seams already exist in `orchestration.py`).

### 4.6 Diagnostics, analysis, optimization

Already pointed the right way; the vision just finishes the thought:

- Diagnostics/groups stay as designed (they are the schema template).
- Optimization stays "a scan with a suggester" (`optimize:` block in
  ScanRequest); evaluator/analyzer wiring keeps flowing through the
  diagnostics schema.
- Analysis results land in Tiled as derived runs (the parked "analysis
  artifacts in Tiled" design); `LiveTaskRunner` eventually subscribes to the
  run catalog instead of watching s-files. s-files and legacy scalar exports
  remain **exports**, produced for compatibility, never load-bearing.

## 5. Front-ends in the target picture

- **Scanner GUI thins to three jobs**: schema-driven editors (SaveSets,
  ScanVariables, TriggerProfiles, ActionPlans — forms over Pydantic models),
  a run console (event-stream consumer: progress, dialogs via
  OperatorChannel, log tail), and submission (build a ScanRequest). No engine
  logic. Whether it stays PyQt or ever becomes a web console is then a
  low-stakes choice, because it's *replaceable*.
- **Phoebus** owns synoptics, trigger state (§4.4), device health
  (`CONNECTED` walls), and archived-PV browsing.
- **Notebooks/scripts** use `GeecsSession` directly — already true today,
  kept sacred: every feature must work headless first.
- **queueserver** is the natural future home for MultiScanner-style queues
  of ScanRequests; explicitly *not* current-phase work.

## 6. Observability & ops (the surrounding envelope)

- Gateway runs as a systemd service with smoke tests (landed 2026-07-07).
- Archiver Appliance archives a curated PV subset (gateway health, device
  CONNECTED, trigger state, key readbacks) — the Citadel replacement.
- The image sidecar (NTNDArray live view) stays view-only; native GEECS
  file saving remains the authoritative image path until proven otherwise.
- One `config.ini` for infrastructure addresses (DB, Tiled, `[epics]`);
  everything experiment-shaped lives in the configs repo as schemas.

## 7. What this doc is not

- Not a schedule. Sequencing lives in the planning docs and is gated on
  evidence (parity checklist + production confidence before deletion;
  deletion before schema redesign; schema redesign before GUI rebuild).
- Not a rewrite plan. Every schema ships with a converter from the current
  YAML; the diagnostics migration is the precedent for doing this without a
  flag day.
- Not finished thinking. Open questions worth keeping visible:
  - Where exactly do the schema models live (`geecs_data_utils` vs a new
    lean `geecs_schemas` package)? Data-utils is already accused of being
    "scattered"; a schemas package may be the cleaner home.
  - Does `TRIG:STATE` live in the gateway proper or a tiny timing IOC beside
    it? (Gateway purism says it's a 1:1 GEECS mirror; pragmatism says one
    enum PV is cheap. Decide when building it.)
  - How much PV metadata (limits/units) can the gateway serve from the DB
    without the DB schema itself becoming the bottleneck?
  - ECS dumps and background scans: parity items or retired concepts? The
    parity checklist must answer, not assume.
