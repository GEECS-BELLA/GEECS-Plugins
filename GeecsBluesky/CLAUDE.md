# GeecsBluesky — Developer Context for Claude

Bridges the GEECS hardware control system to the
[Bluesky](https://blueskyproject.io/) experiment orchestration ecosystem.
The primary product is the **queueserver worker** (`qserver/` — a
bluesky-queueserver RE Manager whose startup profile serves
`geecs_scan_request_plan`, the one plan every ScanRequest runs through)
plus the headless `GeecsSession`.  GEECS-Console and every other client
talk to the RE Manager over its queue/status API — the in-process
`BlueskyScanner` GUI bridge was **deleted** (W5, issue #649, 2026-08-21)
after the console became a manager client (W6, #648).  Hardware-verified
for both acquisition modes (free-run and strict) including DG645 shot
control; first GUI-launched scans ran in production on 2026-07-06, first
queue-launched scans 2026-08-21 (Scans 001–004).

**The one submission shape is `geecs_schemas.ScanRequest`.**  The legacy
duck-typed `exec_config` path was deleted root-and-stem (G3, executed
early 2026-07-16 by owner decision).
GEECS-Scanner-GUI itself was **deleted** (2026-08-20, geecs-core arc)
after its `optimization` module relocated into this package; the legacy
scanner line's final state is preserved at the tag `legacy-scanner-final`
(the M6 cutover merged the vision line into `master` the same day).

## Two acquisition modes (the core architecture)

Scans run in one of two modes, declared by the request
(`ScanRequest.acquisition`: `free_run` or `strict`; the old
`GEECS_BLUESKY_ACQUISITION_MODE` env override died with the exec_config
path — a request declares intent).  Both write the **same versioned event
schema** (`EVENT_SCHEMA.md`); consumers branch on `geecs_event_schema`, never on
the mode.  `EVENT_SCHEMA.md` is the canonical data contract.

- **`free_run_time_sync`** — the external trigger free-runs at the machine rep
  rate.  The first synchronous device is the **reference** (pacemaker): its
  `acq_timestamp` advance creates one event row; every other device fills that
  row's columns, each labeled with a derived `shot_id` / `shot_offset` /
  `valid` so late/slow devices are tolerated and realignable downstream.
- **`strict_shot_control`** — every device must be present on each shot.  With
  a reachable shot-control device and an `ARMED` state in the shot-control
  config it does true plan-owned single-shot (arm → confirm trigger quiescent
  → fire one shot → await all).  Strict mode aborts when those requirements are
  not met; use `free_run_time_sync` for free-running trigger acquisition.

NOSCAN ("statistics collection") is just a motorless step scan (one no-move
bin), so it honours the same mode dispatch.

## Package Layout

```
geecs_bluesky/
  session.py                # GeecsSession — headless scans (RE + Tiled + discipline)
                            #   + session.run(ScanRequest) — the schema front door;
                            #   writes scan.log when it claimed the scan number
                            #   + move_variable — manual scan-variable move
                            #   (plain/confirm/pseudo, fresh movable per
                            #   call, refused mid-scan)
                            #   + run_action / describe_action — on-demand
                            #   ActionPlan execution & dry-run (G-actions v1)
  preflight.py              # Pre-flight checks as a pipeline (pass/ask/abort);
                            #   UnservedVariablesCheck (devices-config level,
                            #   pre-device-build), run pre-claim.  Headless
                            #   engine-side (an Ask takes its on_default);
                            #   the console runs the checks pre-submit and
                            #   renders Ask as a modal (its submit_preflight
                            #   is the live consumer — OperatorQuestion +
                            #   ANSWER_* live here since W5)
  config_resolver.py        # ConfigResolver protocol + ConfigsRepoResolver:
                            #   ScanRequest names → schema models (new-schema
                            #   YAML directly, else legacy-convert)
  forward_expr.py           # compile_forward — AST-whitelist compiler for
                            #   pseudo-variable forward formulas (arithmetic +
                            #   math functions; scanned value = composite_var
                            #   or its alias x; corpus pinned by tests;
                            #   skeleton = geecs_schemas.restricted_expr,
                            #   shared with the gateway's derived channels)
  scan_request_runner.py    # run a geecs_schemas.ScanRequest:
                            #   SaveSet→devices_config and
                            #   TriggerProfile→ShotControlWrites (ordered,
                            #   multi-device) adapters, save-set union,
                            #   action slot assembly (§4.4b layers) + compile
                            #   + signal prefetch, multi-axis grid execution
  plans/
    orchestration.py        # build_step_scan_plan — THE one scan recipe (both front
                            #   doors); setup/per_step/closeout action hooks +
                            #   finalize nesting (save-off → disarm → closeout)
    action_compiler.py      # compile_action_plan — ActionPlan → plan stubs
                            #   (legacy ActionManager semantics pinned; signals
                            #   from an injected SettableFactory) +
                            #   flatten_action_steps (the pure resolve/flatten
                            #   walk behind describe_action and fail-fast
                            #   nested-name validation)
    step_scan.py            # geecs_step_scan — step scan (motor optional OR a
                            #   motor list = multi-axis grid; per_step hook)
    free_run_step_scan.py   # geecs_free_run_step_scan — reference-paced + t0-sync + tail flush
    optimize.py             # geecs_adaptive_scan — optimization as a scan (iteration = bin)
    single_shot.py          # geecs_single_shot + geecs_confirm_quiescent
    t0_sync.py              # geecs_t0_sync — coordinated per-device t0 capture
    run_wrapper.py          # geecs_run_wrapper + claim_scan_number (numbering + save + md)
    scan_request_plan.py    # geecs_scan_request_plan — "run this ScanRequest"
                            #   as ONE plan (queueserver round 1, issue #633):
                            #   the run_scan_request prologue relocated into
                            #   the plan preamble (validate → resolve →
                            #   construct+connect in-plan → claim → the same
                            #   inner plan); set_plan_session installs the
                            #   worker default session; optimize-mode
                            #   requests run in-plan too (set_optimization_
                            #   loader registers the worker's loader — see
                            #   optimization/worker_loader.py — refused
                            #   loudly, not mid-scan, when unregistered);
                            #   s-file export is NOT here (worker stop-doc
                            #   callback seam)
  devices/
    ca/                     # THE device family: CA-backed via GeecsCAGateway PVs (`ca` extra)
      triggerable.py        # CaAcqTimestampReadable (persistent CA monitor) + CaTriggerable
      generic_detector.py   # CaGenericDetector — shot-id columns + native saving
      timestamped_readable.py # CaTimestampedReadable — free-run contributor
      snapshot.py           # CaSnapshotReadable — async readback
      settable.py           # CaSettable — put :SP, read streamed readback
      motor.py              # CaMotor — blocking :SP put + readback-tolerance poll
      pseudo.py             # CaPseudoMovable — pseudo (composite) variable:
                            #   one number fanned out to N targets' :SP
                            #   (absolute / relative-with-staged-baselines)
      confirm.py            # CaConfirmSettable — set X, confirm on a
                            #   different variable Y (ScanVariable.confirm)
      action_signals.py     # CaActionSignalFactory — the production
                            #   SettableFactory for compiled action plans
                            #   (cached :SP settables + str readbacks)
      gateway_put.py        # GatewaySetpointPut — THE one gateway :SP put
                            #   primitive (addressing rule ca://-vs-bare,
                            #   wire conventions, timeout, mock); every
                            #   setpoint pathway delegates (issue #490)
    shot_id.py              # ShotIdTracker + ShotIdSupport mixin (schema-v1 columns)
    nonscalar_save.py       # NonScalarSaveSupport mixin — save-path column + asset docs
    contributor.py          # FreeRunContributorSupport — reference-relative labeling
    scan_context.py         # ScanContext — bin_number / shot_index_in_bin / scan_event_index
  analysis/                 # Post-run analysis contracts: models (AnalysisResult,
                            #   FeatureRow, provenance), derived analysis runs
                            #   published to Tiled, ImageAnalyzerAdapter, camera
                            #   end-to-end analysis over archived Tiled runs
  assets/                   # External asset helpers for native GEECS files
                            #   (handlers, readback, registry)
  epics_env.py              # Applies [epics] ca_addr_list from the shared config
                            #   before aioca import (called by geecs_bluesky/__init__)
  scan_log.py               # shared per-scan scan.log handler (scan_log() ctx
                            #   manager) — bridge delegates; GeecsSession
                            #   attaches it when it claimed the scan number.
                            #   Root-logger capture (0.51.0): scan.log records
                            #   the whole process story (bluesky/ophyd_async/
                            #   geecs_data_utils included), with a pre-claim
                            #   buffer started at submission so the file opens
                            #   with connects/telemetry drops; httpx +
                            #   mysql.connector INFO chatter filtered out
  shot_controller.py        # ShotController — arm/disarm/quiesce/fire plan stubs (gateway :SP)
  optimize.py               # suggester protocol, RandomSuggester, XoptSuggester, BinData
  optimization/             # the config-driven Xopt/evaluator stack (relocated
                            #   from geecs_scanner.optimization 2026-08-20):
                            #   BaseOptimizer, evaluators, generator factory,
                            #   SessionOptimizationBridge, config models,
                            #   _legacy_models_* (ActionSequence/SaveDeviceConfig
                            #   — the legacy engine re-imports them via shims).
                            #   Heavy deps ride the `optimize` extra; tests
                            #   skip whole without it
    worker_loader.py         # the queueserver worker's optimization_loader:
                            #   OptimizationSpec -> SessionOptimizationBridge
                            #   (the ONE loader implementation — its former
                            #   console twin died with the cutover); also
                            #   warm_up_optimization_stack(), called once at
                            #   worker startup (qserver/startup/startup.py)
                            #   to pre-import the heavy stack off-thread
  tiled_integration.py      # subscribe_tiled + descriptor patch + safe callback
  data_paths.py             # local ↔ device-server path mapping, asset roots
  scanner_configs.py        # configs-repo resolution + shot-control YAML loading
  models/
    shot_control.py         # ShotControlConfig / ShotControlState — validated YAML
  exceptions.py             # scan-level errors; wire-level ones re-exported from geecs-core
  utils.py                  # safe_name()

The GEECS access-layer core (``transport/``, ``db/``, ``pv_naming``,
``FakeGeecsServer``, wire-level exceptions) lives in **GEECS-Core**
(``geecs_core``) — this package depends on it for library use (``GeecsDb``
metadata, naming, exceptions) and consumes the CA gateway purely as a
service (its PVs) for all device I/O; no gateway code is imported.

EVENT_SCHEMA.md — the canonical event-schema v1 data contract (read it).

## The scan service — key design points

### The queueserver worker (the one service surface)

`qserver/` holds the worker: `launch_re_manager.sh` (Redis + the
bluesky-0MQ-proxy document stream + `start-re-manager --keep-re`),
`startup/startup.py` (builds the headless `GeecsSession`, exposes its
`RE`, registers `geecs_scan_request_plan` + `geecs_run_action_plan` and
the `function_execute` manual verbs `geecs_move_variable` /
`geecs_describe_action`, subscribes Tiled + the s-file stop-doc callback,
registers the optimization loader), `user_group_permissions.yaml`, and
`deploy/` (systemd unit + runbook).  Read `qserver/README.md` first —
its Troubleshooting section is the empirical contract (permissions file,
`--keep-re`, manager-restart-after-install, failed-items-requeue-at-front,
CLI parses Python literals not JSON).

Clients submit `ScanRequest.model_dump(mode="json")` dicts as queue items;
GEECS-Console's client lives in `geecs_console/services/queue_client.py`.
Operator pause/resume/stop are the manager's own verbs (decision 4):
deferred pause lands at the next plan checkpoint and resume replays
nothing; stop-from-paused finalizes gracefully with partial data (both
live-verified 2026-08-21).  **Expected pause latency — do not "fix"
(live-investigated 2026-07-16):** 1–2 shots complete after the pause
request — the in-flight shot always finishes (checkpoints deliberately
never split a shot) and the in-flight GEECS blocking set is always waited
out; this is the architectural floor, and getting under it means
reintroducing the hard-pause replay trap (a HARD pause replays from the
last checkpoint and re-executes GEECS sets — scoping note in the
Planning doc).

On a failed axis move the queue plan pauses instead of raising
(`failed_move_policy="pause"`, decision 4 — the reason line is the
`FAILED_MOVE_LOG_PREFIX` ERROR record; resume retries the move, stop ends
the scan gracefully).  Headless `session.scan` keeps the `raise` default:
with no operator to answer, a pause would hang.

The deleted bridge's other verbs re-homed as follows (W6, #648): manual
actions → `geecs_run_action_plan` queue items (idle-only by queue
semantics); manual moves → `geecs_move_variable` via `function_execute`
(foreground function execution requires an idle manager; both queue
plans additionally refuse while the session's manual-move lock is held —
background function execution bypasses the manager's idle gate);
pre-flight questions → client-side pre-submit (decision 3, provenance in
`ScanRequest.submission`); GUI progress → the ZMQ document stream; the
pause-window action flow (G-actions v2) was **dropped** (decision 2) —
its `action_direct`/`PauseSupervisor`/`OperatorChannel`/`events`
machinery is deleted (W5; `ShotControlPauseQuiescer` in
`plans/pause_semantics.py` is the live pause-quiesce equivalent).

### Shot control — `ShotControlConfig` + named states

Shot control is a validated `ShotControlConfig` (`models/shot_control.py`),
coerced from the `{device, variables: {var: {state: value}}}` YAML via
`ShotControlConfig.from_information` (empty/`{}` → `None`, no shot control).
States are `ShotControlState`: `OFF`, `SCAN`, `STANDBY`, `SINGLESHOT`, `ARMED`.
`values_for_state(state)` returns the `{var: value}` writes for a state, skipping
empty-string no-ops (matching legacy `TriggerController`).

`ShotController` (`shot_controller.py`) drives the shot-control device(s)
through named states as plan stubs, via `CaPutSetter`s writing the gateway
`:SP` PVs (put-completion rides GEECS's blocking set). Two construction
paths: the single-device `ShotControlConfig` + one setter per variable
(state writes issued concurrently — byte-identical to the pre-M3b
behavior; headless callers may still hand a `ShotControlConfig` to
`GeecsSession.shot_control`), and
`ShotController.from_writes(ShotControlWrites)` — generalized per-state
**ordered** `(device, variable, value)` lists, possibly spanning several
devices (TriggerProfile semantics: writes replayed top to bottom, each
completing before the next; one cached `CaPutSetter` per distinct target).
`trigger_writes_from_profile` (scan_request_runner) adapts a TriggerProfile
into that shape — the runner attaches it per request (`trigger_profile`
named ⇒ writes, else `shot_control(None)`); `GeecsSession.shot_control`
accepts either generation.  The plans use it:

- `arm()` → `SCAN`, `disarm()` → `STANDBY` (per-step bracketing on the
  free-running modes; jet on during shots, off during moves)
- `quiesce()` → `OFF` (stops the free-run — used before free-run t0 sync;
  `STANDBY` keeps the trigger free-running on real hardware, so it cannot quiesce)
- `arm_single_shot(detectors)` → `ARMED` then `geecs_confirm_quiescent`, and
  `fire_shot()` → `SINGLESHOT` (strict plan-owned single-shot)

How they compose per mode (native saving is **windowed** to the
trigger-stopped part of the scan — Gate-2 hardware finding: an eager save-on
let free-running frames be saved as orphan images joining no event row):
```
free-run:  quiesce[OFF] → save-on → t0_sync → per step: mv → arm[SCAN] → N×(ref-paced read) → disarm[STANDBY] → end: quiesce[OFF] → tail flush
strict:    setup once: arm[ARMED] → confirm quiescent → save-on → per shot: trigger→fire[SINGLESHOT]→await→read
```
(`geecs_run_wrapper(defer_save_on=True)` + the step plans' `enable_saving`
hook yielding `save_enable_plan`; ScanRequest setup actions run before the
save-on point by construction.  The end-of-scan quiesce closes the tail:
STANDBY passes external edges, so without it frames kept landing between
the last disarm and the finalize save-off.)

A `bpp.finalize_wrapper` around the plan guarantees the disarm (→ `STANDBY`)
runs even on mid-scan abort; the finalize nesting is quiesce[OFF]
(free-run abort parity, inside the plan; skipped when the end-of-scan
quiesce already ran) → save-off → disarm → closeout, so saving always
stops while the trigger cannot pass edges, and hardware is restored to the
legacy free-running STANDBY end state last.  **Accepted window, do not
"fix"**: between-step STANDBY frames in multi-step free-run scans — the
per-step disarm during moves is deliberate legacy parity (jet off during
moves); frames there join by timestamp and orphans are ignorable — never
turn this into per-step save toggling.

`ARMED` is **config-specific**: it sets data-taking output (jet amplitude /
delay) + the single-shot trigger source — *external* single-shot when the laser
is on, *internal* (`Single shot`) when off.  The Python is agnostic; the
difference lives entirely in the per-config YAML.

### Acquisition-mode dispatch

`ScanRequest.acquisition` selects the mode (`free_run` or `strict`); the
runner's `_build_request_detectors` (`scan_request_runner.py`) assigns each
save device a role from it: free-run → first sync device is `reference`,
later sync devices are `contributor` (`CaTimestampedReadable`), async are
`snapshot`; strict → all sync are `triggered` (`CaGenericDetector`).
Explicit `role:` overrides in the save set are honoured by
`save_set_to_devices_config` (reference moves first; conflicting explicit
roles across merged sets raise).  STEP and NOSCAN share one plan body
(NOSCAN = `motor=None`, one no-move bin).

### Tiled integration

`GeecsSession` (with `tiled=True` — the worker startup profile's setting)
reads `[tiled] uri` and `[tiled] api_key` from
`~/.config/geecs_python_api/config.ini` and subscribes a `TiledWriter` to the
`RunEngine`.  All event documents (start, descriptor, event, stop) are written to
the Tiled catalog at `http://192.168.6.14:8000`.  Silently skips if the server is
unreachable or `tiled[client]` is not installed.

### Threading model

Under the queueserver the RE Manager's worker process owns the RunEngine
(`--keep-re`: one module-level `RE` alive across queue items); headless
`GeecsSession` runs it in a background thread (`bluesky-scan`).  The RE's
internal `asyncio` event loop is persistent — devices are connected into
it and remain connected across the scan.  `RunEngine(context_managers=[])` disables SIGINT
handling, which fails when the RE is not on the main thread.

Device connect/disconnect uses `asyncio.run_coroutine_threadsafe(...).result(timeout=...)`.

## Device Layer

**Devices are CA-backed only** (`devices/ca/*`): stock ophyd-async
`epics_signal_r/rw` against the GeecsCAGateway PVs
(`[experiment:]device:variable` — all-lowercase components, setpoints at `…:SP`). Requires the `ca` extra
(`aioca`) and a running gateway (≥0.3.0 for control-surface and long-string
path PVs). The gateway is consumed as a **CA service, never as a Python
import** (the gateway imports our transport core, so an import the other way
would be circular). The direct UDP/TCP device backend was deleted after the CA
backend reached verified live parity (Scans 007–015, 2026-07-03/04); a stale
`GEECS_BLUESKY_DEVICE_BACKEND` env var set to anything but `ca` now raises.

- **`CaAcqTimestampReadable`** — readable signals + a persistent CA monitor on
  `acq_timestamp` feeding a local cache/queue (the CA analogue of the old TCP
  shot cache). Non-positive timestamps are ignored: `0.0` is the gateway
  channel's pre-acquisition placeholder, so "never acquired" reads as `None`.
  Also carries a non-readable `connected_status` child on the gateway's
  per-device `CONNECTED` PV (created outside `add_children_as_readables()`,
  so never in event rows) — the authoritative liveness signal consumed by
  the scanner pre-flight and the strict refire gate; only the exact
  `"Disconnected"` reading means down (fail-open otherwise).
- **`CaTriggerable`** — `trigger()` completes when `acq_timestamp` advances.
  The stale-drain and baseline happen **synchronously in `trigger()`** so a
  shot fired immediately after `bps.trigger` (strict single-shot) can't be
  missed — pinned by a mock race test.
- **`CaGenericDetector`** — the triggered detector: dynamic float signals +
  schema-v1 companion columns (`ShotIdSupport`) + native file saving
  (`NonScalarSaveSupport`; `localsavingpath`/`save` write the gateway `:SP`).
- **`CaTimestampedReadable`** — free-run contributor: non-blocking reads with
  reference-relative `shot_offset`/`valid` labeling (the shared
  `FreeRunContributorSupport` mixin).
- **`CaSnapshotReadable`** — async readback sampled once per event row.
- **`CaSettable` / `CaMotor`** — puts ride GEECS's native blocking convergence
  through the gateway `:SP`; the motor adds a readback-tolerance poll with
  `move_timeout` as the CA put budget. **Known gap:** no `stop()` — GEECS has
  no universal abort variable (some device types have one, implemented
  inconsistently), so an RE abort cancels the wait but the hardware finishes
  its move. If a specific device's abort variable matters, an optional
  `stop_variable` hook on `CaMotor` is the intended future shape.
- **`CaPseudoMovable`** — the pseudo (composite) scan variable
  (`devices/ca/pseudo.py`, 0.47.0): `set(u)` evaluates each component's
  compiled `forward` formula (`forward_expr.compile_forward` — AST
  whitelist, `composite_var`/`x` as the scanned value; legacy
  `composite_variables.yaml` corpus pinned) and puts every target's `:SP`
  concurrently through `GatewaySetpointPut` — completion when the slowest
  target's GEECS exe response lands; setpoint semantics per component (v1,
  legacy `ScanDevice` parity — per-component `kind: motor` is the intended
  additive upgrade).  `relative` mode captures per-target baselines at
  `stage()` (lazily on first `set()` for unstaged callers, i.e. optimize),
  drops them at `unstage()`; `absolute` mode is stateless.  **A scan
  restores a relative pseudo axis to its captured baselines at the
  end** (0.50.0, owner request): `restore_baselines_plan()` runs as a
  finalize in `build_step_scan_plan` — success and abort, after
  closeout, before unstage — with exact per-target puts; absolute
  pseudos and plain axes are deliberately not restored.  **The recorded
  event-row column is the demanded pseudo value** (soft readback child,
  header = the catalog friendly name) — include target devices in the save
  set when their measured positions matter.  Built by
  `GeecsSession.pseudo_movable` via `build_movable`'s dispatch on
  `PseudoMovableTarget` (both grid-axis and optimize paths); spec + formula
  sources recorded in run metadata under `pseudo_variables`.
- **`CaConfirmSettable`** — the topology-C device (`devices/ca/confirm.py`):
  writes `variable` but confirms on a *different* variable's readback
  (`ScanVariable.confirm`) — the EMQ triplet's `Current_Limit.ChN` (a
  software limit) vs its measured `Current.ChN`. Analog match by tolerance
  (default 0.05, sized from a live no-beam characterization — jitter 0.01 A,
  <1 s lag, ~3-frame settle; dispatch is on the declared `datatype`, not
  parseability — a `str` confirm target matches by exact equality even when
  the label looks numeric) or discrete match by exact equality (future
  `CaShutter`). `GeecsSession.confirm_settable(...)` builds it;
  `resolve_movable_target` returns the entry's `confirm` target alongside
  `(device, variable, kind)`, and `build_movable` dispatches on it (confirm
  wins over `kind`) in both the grid-axis and optimize-mode movable
  construction paths of `scan_request_runner`. **The recorded event-row
  column is the written variable, not the confirming one** (same "motor
  column" shape as `CaSettable`/`CaMotor`) — include the confirming variable
  in the save set separately when the measured value itself matters, not
  just the pass/fail of confirmation. Any code that moves a confirm-settable
  outside a plan (e.g. optimize `on_finish`) must go through its `set()`,
  never the raw `:SP` signal — `GeecsSession._move_movables` does.

Shot IDs (`ShotIdTracker`): a device's `shot_id` is its physical
trigger-opportunity number, derived **incrementally** from its own
`acq_timestamp` (`shot_id += round(Δt × rep_rate)`) so rep-rate error never
accumulates. Cross-device matching is `shot_id` **equality**; files join to
events by device `acq_timestamp`, never by `shot_id`.

Hermetic testing uses ophyd-async mock backends (`tests/ca_mock_helpers.py`):
`set_mock_value` on `acq_timestamp` is a shot, `start_pacer` on the RE loop is
the free-running trigger, `follow_setpoint` stands in for GEECS convergence.

## Read path: staging & shot coherence (0.32.0)

**The read-path contract: every per-row read device is staged.**
`build_step_scan_plan` wraps the composed plan in `bpp.stage_wrapper` over
detectors + telemetry + motors, so every readable child signal gets a caching
CA monitor for the scan's duration and per-shot `read()`s are served from
memory — zero network round trips per row.  Before this, every signal of
every device was one uncached CA get per row, serialized across devices by
the RunEngine (measured live 2026-07-13: 87 telemetry devices × ~7 ms VPN
RTT ≈ 0.7 s/row — scans at a 1 Hz trigger ran at exactly 0.5 Hz).  Pinned by
`tests/test_read_path_staging.py` (zero backend gets per row; stage/unstage
bracket the run, including the abort path).

Why cached reads are *correct*, not just fast — the coherence chain, each
link verified against installed sources (caproto 1.3.0, aioca 2.1,
ophyd-async 0.19.3):

1. The gateway posts a frame's data variables before its timestamp-ladder
   variables (PV_CONTRACT.md §3, pinned by
   `test_callback_posts_timestamp_variables_last`), sequentially awaited —
   never gathered.
2. caproto serializes all subscription updates FIFO through one context
   queue → one per-circuit queue → one TCP socket; aioca delivers via a
   single FIFO `call_soon_threadsafe` hop; the ophyd-async cache updates
   synchronously in that callback.
3. Therefore when `trigger()` completes (the `acq_timestamp` advance
   arrived), every staged data cache already holds that frame's values —
   or newer, never older.  This is strictly stronger than an uncached
   post-trigger get (which samples the same gateway cache, with the same
   next-frame race, plus a round trip).

Conditions the guarantee stands on (do not silently violate):

- One CA context per process (ordering is per-circuit; aioca's module-level
  context — the default — satisfies this).
- `OPHYD_ASYNC_EPICS_CA_KEEP_ALL_UPDATES` stays at its `True` default —
  client-side coalescing would break "data before timestamp" delivery.
- The client must not backlog past caproto's per-subscription drop-oldest
  quota (~1000 updates) — the stated boundary where coherence can break.
- Connection loss surfaces as `alarm_severity`/`CONNECTED`, never assumed
  away — a dead monitor serves its last reading silently.

Sharp edges: ophyd-async staging is a **bool, not a refcount** — the
orchestration layer stages exactly once; nested plans must never re-stage
these devices.  A staged reading's `timestamp` is the CA server (device
ladder) time of the last delivery, not read time.  The first post-stage
read blocks until each monitor's initial update (stage early, not
mid-step).  `CaAcqTimestampReadable`'s persistent `subscribe_reading`
shares the same refcounted-by-listeners cache and survives unstage —
deliberate and safe.

Rate-derived bounds (1 Hz-era constants scaled for the 5 Hz system limit):
the contributor grace wait is capped at half a trigger period
(`_effective_grace_wait_s`), the t0-sync window is capped at
`0.4 / rep_rate_hz` (recorded value lands in the start doc), the telemetry
per-signal read budget is 2 s (`CaTelemetryReadable._read_timeout_s` — one
hung PV costs at most that, not ophyd's 10 s default), and the shot queue
holds 128 updates.  **The t0-sync design floor**: the window must exceed
inter-machine clock skew (~50 ms); at 5 Hz that leaves 50–80 ms of margin,
and rates meaningfully beyond 5 Hz need a redesigned seeding stage — see
`Planning/device_read_path/00_overview.md`.

## Transport Layer — moved to GEECS-Core

The GEECS wire-protocol transport (`GeecsUdpClient`, `GeecsTcpSubscriber`)
no longer lives in this package: it lives in `GEECS-Core/geecs_core/transport/`
(extracted 2026-08-20 from its interim home in GeecsCAGateway), alongside the
DB layer and PV naming.  This package touches GEECS devices **only** through
the gateway's CA PVs; it imports geecs-core's library modules (`GeecsDb`,
`pv_naming`, wire-level exceptions) and never the transport, the gateway, or
the server.  See `GEECS-Core/DESIGN.md` and `GeecsCAGateway/README.md` for
the protocol details that used to be documented here.

**Images: two paths, deliberately separate.** Per-shot *scan data* stays on
the file path — the LabVIEW device writes files, this package's asset
registry/handlers reference them, Tiled serves them post-hoc. *Live* frames
are NTNDArray PVs over pvAccess served by `GeecsPvaGateway/` (distributed,
per camera server, same `pv_naming` namespace). ophyd-async speaks PVA
natively, so a live-image signal is a stock EPICS signal when a use case
wants one — never a bespoke transport, and never 2 MB frames through the
document stream.

## Test Infrastructure

`FakeGeecsServer` / `FakeGeecsDevice` (the in-process UDP/TCP server that
speaks the real GEECS wire protocol) also lives in GEECS-Core
(`geecs_core.testing`).  This package's hermetic tests are built on
ophyd-async **mock backends** instead (`tests/ca_mock_helpers.py`) — see the
Device Layer section above.

### Hardware integration test

`tests/test_scan_request_hardware.py` (integration-marked, skipped in CI)
runs a real ScanRequest end to end against the live gateway over the lab
network / VPN — save set + trigger profile resolved from the configs repo,
every name parameterizable via `GEECS_HW_*` env vars (see its module
docstring for the invocation and the table).  The old exec_config-driven
`test_bluesky_scanner.py` hardware script was deleted with the legacy
path (G3).

## Configuration

All runtime config reads from `~/.config/geecs_python_api/config.ini`:

```ini
[Paths]
geecs_data = /path/to/user data   # must point to dir containing Configurations.INI

[tiled]
uri = http://192.168.6.14:8000
api_key = <key>
```

`GeecsDb` reads `Configurations.INI` (in `geecs_data`) for MySQL credentials.

## Engine consolidation — current state (post-W5)

The bridge-era event vocabulary (`events.py`), the `OperatorChannel`
dialog transport, `pause_supervisor.py`, and `plans/action_direct.py`
were deleted with the GUI bridge (W5, #649) — the manager status /
document stream / console-output stream are the client-facing vocabulary
now.

Pre-flight is a pipeline (`preflight.run_preflight`); new checks are list
entries.  One engine-side check exists: the config-level
`UnservedVariablesCheck`, which the **runner and the queue plan** run
over the resolved devices config *before any detector is built* — a
save-set variable outside the gateway's served set (`get='yes'` ∪
settable of enabled devices; `GeecsCAGateway/DEPLOYMENT.md`) has no PV,
so building its detector used to die in a 20 s ophyd
`NotConnectedError` (live incident 2026-07-15: `UC_TopView`
`2ndmomW0x`/`2ndmomW0y`).  Engine-side the check is headless
(continue-and-drop with a WARNING); the console asks the same question
pre-submit (decision 3) and stamps the answer into
`ScanRequest.submission`.  Dropped variables — a fully-unserved device
is dropped whole — are recorded in run metadata
(`dropped_unserved_variables` / `dropped_unserved_devices`).  The served
set comes from the failure-tolerant `db_runtime.GeecsDbServedSetProvider`,
whose DB failure reads as *unknown* (check skipped with one warning),
never as *empty*.  (The old device-level `GatewayLivenessCheck` /
`FreeRunStalenessCheck` were bridge-hook-only and died with it; the
console's pre-submit CONNECTED/staleness reads are their successors.)

`ScanRequest` execution (`scan_request_runner` / `GeecsSession.run`) runs
the full schema surface as of 0.23.0 (M3b): **actions execute**
(request-level setup/per_step/closeout, SaveSet entry rituals de-duplicated
by name, ExperimentDefaults plans — assembled in §4.4b nesting order:
defaults → entries → request on setup, exact mirror on closeout; the
assembled order is recorded in run metadata as `action_plans`), **multi-axis
grids execute** (outer product, first axis outermost; only changed axes
re-moved; every axis readback in every event row; `scan_axes`/`grid_shape`
metadata), and **multi-device trigger profiles execute** (ordered write
lists via `ShotControlWrites`).  Action plans compile via
`plans/action_compiler.py` against the session's `CaActionSignalFactory`;
every signal is prefetched/connected pre-claim (a lazy connect inside the
RE loop would deadlock).  Names still resolve fail-fast pre-claim.
Remaining validated-then-refused v1 gaps: `all_scalars`, and optimize
without an injected objective/suggester.  (Pseudo scan variables execute
as of 0.47.0 — see `CaPseudoMovable` in the Device Layer section; a bad
`forward` expression still fails validation pre-claim.)
Actions on an optimize-mode request are **not** refused — optimize has no
action hooks yet, so the actions (request, experiment defaults, and
save-set rituals) are skipped, logged (WARNING), and recorded in run
metadata under `skipped_action_plans` (refusing would block every
optimization the moment an experiment defines default bracket actions;
unknown names still fail fast).  **Both execution paths run through the
one engine definition**: the queue plan (`geecs_scan_request_plan`) and
the headless runner (`run_scan_request`, via `GeecsSession.run`) share
the runner's module-level prologue functions, so actions, entry rituals,
multi-axis grids, db_scalars, and telemetry execute identically.  The
runner's bridge-era hooks (`preflight`, `on_scan_start`,
`operator_channel`, `pause_supervisor`) died with the bridge (W5);
`should_abort` survives as the one external-stop probe (pre-claim
init-stage checkpoints + the in-plan gate).  **Optimize-mode requests**:
the worker's startup-registered `optimization_loader`
(`optimization/worker_loader.py`, decision 5) — or a headless caller's
injected objective/suggester — builds the stack; the loader's one
argument is the request's resolved `OptimizationSpec`, and the returned
bridge's `bind` threads in as the `optimization_binder` hook.  Because
the binder's analyzers need the real `ScanTag`, the **runner claims the
scan itself just before binding** (after every fail-fast resolution and
device connect — the one path where the claim is not inside the session
call) and passes the pre-claimed number/folder to `session.optimize`,
owning the `scan.log` attach; the optional `finish()` (legacy
`xopt_dump.yaml`) runs after a successful run.  The config-level
unserved-variables check runs pre-claim on every mode.  **Optimizer
`device_requirements` auto-provisioning (0.38.0)** — reversing the
deliberate #520 deferral after a field incident (2026-07-15:
`TopViewMax` optimize runs produced NaN objectives on every iteration
because the evaluator's auto-generated requirements were ignored and
`UC_TopView` never saved).  The runner reads the loader-returned
bridge's `device_requirements` duck-typed (like `finish`) and hands the
opaque mapping to `run_scan_request(device_requirements=...)`;
`merge_optimizer_device_requirements` unions it into the effective
devices config with `merge_save_sets` semantics (variable lists deduped,
`save_nonscalar_data` ORs; an already-configured device keeps its
save-set `synchronous`/role semantics, new devices append after the
save-set ones — pinned parity with the deleted legacy merge, including
case-insensitive device-name matching).  Provisioned additions run
through the same unserved-variables pre-flight and are recorded in run
metadata as `provisioned_device_requirements`.  A zero-save-sets
optimize request is now valid when the optimizer provisions its own
diagnostics; an empty *effective* device set still refuses pre-claim
with a clear `GeecsConfigurationError`.  The
dependency direction (no `geecs_scanner` import anywhere in this package)
is pinned by an AST-level test in the scan-request seam suite.
Experiment defaults
(`experiment_defaults.yaml`) fill request fields left unset — never
overriding explicit values — and every applied default is recorded into
the run metadata for provenance (closeout defaults append *after* the
scan's own since geecs-schemas 0.2.0 — mirrored teardown).

**M4 step 0 (0.25.0) — multiple save sets union.**  `ScanRequest` now carries
`save_sets: list[str]` (was the single `save_set`); a bare string still
validates (coerced to a one-element list by a schema before-validator).
`run_scan_request` (and the optimize path) resolve **each** named save set and
union them into one effective `SaveSet` (`merge_save_sets`) before deriving the
recorded device set, so operators mix and match named diagnostic groups per
scan.  Per-device union rule (documented on `merge_save_sets` and in the
`scan_request_runner` module docstring): `scalars` union
order-preserving/deduped, `images`/`db_scalars`/`all_scalars` OR together (True
wins), the single non-`None` `role` used — **conflicting explicit roles across
the sets raise** (role sets the pacemaker/contributor/snapshot semantics, so
overlapping sets must not disagree) — entry-level `setup`/`closeout` ritual name
lists union (deduped).  Entry rituals are collected across *all* named sets,
deduped by plan name so a shared ritual runs once
(`resolve_save_sets_and_rituals`).  Everything downstream operates on the
merged set: `save_set_to_devices_config`, the reserved-boundary warning, and —
crucially — **telemetry exclusion** (`select_telemetry_variables` gets the
merged set, so Tier-2 telemetry excludes devices in *any* named set).  Run
metadata records the list under `save_sets`.  The queue plan shares the
same resolution functions, so the union — entry rituals included —
applies identically on both paths.

**M3c (0.24.0) — the DB-integration runtime tier, GET-SIDE ONLY.**  Two
get-side capabilities are live, all gated by schema flags that already
existed (the schema fields are untouched — only descriptions changed); the
pure resolution logic lives in `geecs_bluesky/db_runtime.py`, the one place
touching `GeecsDb` is its failure-tolerant `GeecsDbScalarPolicy` (a DB lookup
that fails degrades to empty policy + a warning — a scan never aborts because
the DB blipped):

- **db_scalars (Tier 1 recorded scalars).**  A `SaveSetEntry`'s recorded
  scalars = its DB `get='yes'` variables ∪ its explicit `scalars`
  (`db_scalars=True`, default); `all_scalars=True` unions *every* DB variable;
  `db_scalars=False` (the legacy-converter pin) = explicit-only.
  `save_set_to_devices_config(save_set, scalar_policy)` threads it; with no
  policy (no DB / off-network) only the explicit list is recorded — M3b
  behavior, strictly additive.
- **Background telemetry (Tier 2).**  Every live device with a `get='yes'`
  variable not in *any* named save set (the merged set — see M4 above) → soft
  `CaTelemetryReadable` columns
  (`telemetry_<device>-…`): read-only, never waited on (a failed read is a
  dtype-appropriate null cell — NaN for numeric, `""` for string — and a
  dead-at-start device is dropped with a log line via `session.telemetry`
  returning `None`).  Telemetry is **dtype-tolerant, per-variable**: signal
  type is inferred from the PV (`epics_signal_r(datatype=None, …)`), so
  numerics stay float (downstream analysis unaffected) while enum/string/path
  variables (e.g. `U_VisaPlungers` `DigitalOutput.Channel N`) are logged as
  their label string.  **No telemetry variable — and no telemetry device — is
  dropped for a *type* reason** (one awkward non-numeric channel must never
  take the device's other columns down; do NOT regress this back to a forced
  `datatype=float`).  A device is dropped only when genuinely unreachable.
  The rule: if we `get` it, we log it.  Gated on
  `ScanRequest.background_telemetry` else the experiment default; selection
  recorded (`background_telemetry`).  **Softness vs synchronicity are
  mutually exclusive — telemetry must never gate a shot; do not make it
  participate in shot completion.**

**Set-side (DB scan start/end writes) is intentionally DISABLED / reserved.**
The `set='yes'` boundary writes are *not* applied by the engine.  Live DB
inspection showed they would race the shot controller / TriggerProfile on the
DG645 — `U_DG645_ShotControl`'s `set='yes'` rows are `Trigger.Source` and
`Amplitude.Ch AB`, the very variables the ShotController already drives — and
the remaining `set='yes'` rows are almost all `save` / `localsavingpath`,
which the scanner owns through its save-windowing.  So triggering is set up
via the TriggerProfile / shot controller and camera saving via the scanner's
own windowing, never via DB boundary writes.  The reserved schema fields
(`SaveSetEntry.at_scan_start` / `at_scan_end`,
`ExperimentDefaults.apply_db_scan_defaults`) are kept for a possible future
re-enable; a config that still sets them draws one `logger.warning`
(`warn_if_reserved_boundary_overrides`) and is otherwise inert (no boundary
write, no setup/closeout chaining, no `db_scan_writes` metadata).  The
gateway's `GeecsDb.get_scan_boundary_writes` remains a reserved read-only
library query, not consumed by the engine.

Optimize mode resolves db_scalars but does not run telemetry yet (no
scan-boundary hook on `GeecsSession.optimize`) — recorded as
`db_scan_runtime` in metadata; the set-side is disabled everywhere.  Adding a
new analyzer/writer still must not create scan folders (cross-package
invariant); M3c is scanner-side but touches no scan-folder creation.

## Known Gaps

The acquisition-modes architecture is complete and hardware-verified (both
modes, including single-shot; GUI-launched scans verified live 2026-07-06).
Remaining items are features/tuning, not architecture — see
`Planning/acquisition_modes/00_overview.md` "Deferred".

- **Strict single-shot needs an `ARMED` state** in the shot-control YAML to
  engage (the production experiment configs have one).  Without `ARMED` or a
  reachable shot-control device, strict aborts before acquisition
  (`GeecsConfigurationError`); use `free_run_time_sync` for free-running
  trigger acquisition.
- **Requested rep-rate throttling is unbuilt** — firing strict single
  shots slower than the external rate (gas-jet economy), or free-run
  subsampling every Nth reference shot.  Carried over from the retired
  pre-queueserver roadmap; not tracked anywhere else.
- **The bridge-era GUI event stream is gone** (W5 — lifecycle/step/dialog
  events; clients observe the manager status, document, and console-output
  streams instead).  Liveness remains CONNECTED-based (the gateway serves
  every DB device's data PVs whether or not the device is up, so
  CA-connect success never implied liveness) — read pre-submit by the
  console's preflight; its staleness sample window still deserves a lab
  session of tuning against real rep rates.
- **Scalar s-files are exported from Tiled best-effort** after a scan when the
  Tiled client extra is installed and the run can be read back.  Legacy TDMS
  output is not produced.
- **Background mode is a metadata flag, not a distinct behaviour** —
  the legacy `Background` scan mode executes as a noscan with
  `ScanRequest.background` set (the schema's own definition), recorded
  as `Background = true` in `ScanInfoScanNNN.ini`; the console's
  BACKGROUND mode submits it.  Optimization runs as a scan via
  `GeecsSession.optimize` (adaptive scan: iteration = bin, same schema/data
  tree as any scan — see `plans/optimize.py`), both headless (suggester +
  objective in hand) and through the queue: the worker's startup-registered
  `optimization_loader` (`optimization/worker_loader.py` builds it from
  `geecs_bluesky.optimization` — the Xopt/evaluator stack relocated INTO
  this package 2026-08-20, heavy deps behind the `optimize` extra) runs
  the config-driven Xopt 3.1 / evaluator / ScanAnalysis stack against
  the session's bin rows (loader argument: the request's resolved
  `OptimizationSpec`; see the engine-consolidation section).  The evaluator seam is
  `EvaluatorDataSource` in `geecs_bluesky.optimization.base_evaluator`;
  the package stays free of any geecs_scanner import (pinned by an
  AST-level test, which now also guards the relocated stack).
- **Action sequences run per request.**  `GeecsSession.run(request)` and
  the queue plan execute setup/per_step/closeout ActionPlans; legacy save
  elements' actions are executed when the element is resolved as a save
  set through a ScanRequest — the converter extracts them into entry
  rituals.  On-demand execution: `GeecsSession.run_action` /
  `describe_action` headless, `geecs_run_action_plan` /
  `geecs_describe_action` through the manager (idle-only; the
  pause/decide/resume during-scan flow was dropped with the queueserver
  migration, decision 2).
- **Scan-folder creation invariant:** `claim_scan_number`
  (`plans/run_wrapper.py`) is the one place (outside the GUI's `ScanDataManager`)
  allowed to create a `scans/ScanNNN/` folder.  It logs a warning and returns
  `(None, None)` if `geecs_data_utils` is unavailable or the NetApp is not
  mounted.  Analysis-side code must still never create missing scan folders.
