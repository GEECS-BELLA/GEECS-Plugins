# GeecsBluesky — Developer Context for Claude

Bridges the GEECS hardware control system to the
[Bluesky](https://blueskyproject.io/) experiment orchestration ecosystem.
The primary product is `BlueskyScanner` — a RunEngine-backed scan executor
designed to become a `ScanManager` replacement.  It runs from the
`GEECS-Scanner-GUI` (`use_bluesky=True`, or the `GEECS_USE_BLUESKY` env var
for a stock GUI session) and has been hardware-verified for both acquisition
modes (free-run and strict) including DG645 shot control; first GUI-launched
scans ran in production on 2026-07-06.

## Two acquisition modes (the core architecture)

Scans run in one of two modes, selected by the
`GEECS_BLUESKY_ACQUISITION_MODE` env var (`free_run_time_sync` or
`strict_shot_control`; default strict).  Both write the **same versioned event
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
  events.py                 # THE typed event vocabulary: ScanEvent hierarchy,
                            #   ScanState, DialogRequest — moved down from
                            #   geecs_scanner (vision §2); geecs_scanner's
                            #   scan_events.py / dialog_request.py are re-export
                            #   shims of these same class objects
  operator_channel.py       # OperatorChannel seam: ask(OperatorQuestion) →
                            #   "continue"/"abort"/default; EventStreamOperator
                            #   (GUI dialog path) / NullOperator (headless)
  preflight.py              # Pre-flight checks as a pipeline (pass/ask/abort);
                            #   GatewayLivenessCheck + FreeRunStalenessCheck,
                            #   run pre-claim, questions via OperatorChannel
  config_resolver.py        # ConfigResolver protocol + ConfigsRepoResolver:
                            #   ScanRequest names → schema models (new-schema
                            #   YAML directly, else legacy-convert)
  scan_request_runner.py    # run a geecs_schemas.ScanRequest:
                            #   SaveSet→devices_config and
                            #   TriggerProfile→ShotControlWrites (ordered,
                            #   multi-device) adapters, save-set union,
                            #   action slot assembly (§4.4b layers) + compile
                            #   + signal prefetch, multi-axis grid execution
  scanner_bridge/
    bluesky_scanner.py      # BlueskyScanner — ScanManager-compatible GUI bridge
                            #   (reinitialize also accepts a ScanRequest)
  plans/
    orchestration.py        # build_step_scan_plan — THE one scan recipe (both front
                            #   doors); setup/per_step/closeout action hooks +
                            #   finalize nesting (save-off → disarm → closeout)
    action_compiler.py      # compile_action_plan — ActionPlan → plan stubs
                            #   (legacy ActionManager semantics pinned; signals
                            #   from an injected SettableFactory)
    step_scan.py            # geecs_step_scan — step scan (motor optional OR a
                            #   motor list = multi-axis grid; per_step hook)
    free_run_step_scan.py   # geecs_free_run_step_scan — reference-paced + t0-sync + tail flush
    optimize.py             # geecs_adaptive_scan — optimization as a scan (iteration = bin)
    single_shot.py          # geecs_single_shot + geecs_confirm_quiescent
    t0_sync.py              # geecs_t0_sync — coordinated per-device t0 capture
    run_wrapper.py          # geecs_run_wrapper + claim_scan_number (numbering + save + md)
  devices/
    ca/                     # THE device family: CA-backed via GeecsCAGateway PVs (`ca` extra)
      triggerable.py        # CaAcqTimestampReadable (persistent CA monitor) + CaTriggerable
      generic_detector.py   # CaGenericDetector — shot-id columns + native saving
      timestamped_readable.py # CaTimestampedReadable — free-run contributor
      snapshot.py           # CaSnapshotReadable — async readback
      settable.py           # CaSettable — put :SP, read streamed readback
      motor.py              # CaMotor — blocking :SP put + readback-tolerance poll
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
                            #   attaches it when it claimed the scan number
  shot_controller.py        # ShotController — arm/disarm/quiesce/fire plan stubs (gateway :SP)
  optimize.py               # suggester protocol, RandomSuggester, XoptSuggester, BinData
  tiled_integration.py      # subscribe_tiled + descriptor patch + safe callback
  data_paths.py             # local ↔ device-server path mapping, asset roots
  scanner_configs.py        # configs-repo resolution + shot-control YAML loading
  models/
    shot_control.py         # ShotControlConfig / ShotControlState — validated YAML
  exceptions.py             # scan-level errors; wire-level ones re-exported from the gateway
  utils.py                  # safe_name()

The GEECS access-layer core (``transport/``, ``db/``, ``pv_naming``,
``FakeGeecsServer``, wire-level exceptions) lives in **GeecsCAGateway** — this
package depends on it for library use (``GeecsDb`` metadata, naming,
exceptions) and consumes its CA service for all device I/O.

EVENT_SCHEMA.md — the canonical event-schema v1 data contract (read it).

## BlueskyScanner — Key Design Points

### Public API (matches ScanManager)

```python
scanner = BlueskyScanner(
    experiment_dir="Undulator",
    shot_control_information=shot_ctrl_yaml_dict,  # optional
)
scanner.reinitialize(exec_config)   # stores config; no hardware yet
scanner.start_scan_thread()         # launches scan in background thread
scanner.is_scanning_active()        # → bool
scanner.estimate_current_completion()  # → 0.0–1.0
scanner.stop_scanning_thread()      # RE.abort() + thread join
```

`RunControl` in `GEECS-Scanner-GUI` switches between `ScanManager` and
`BlueskyScanner` via an explicit `use_bluesky=True` or, for a stock GUI
session, the `GEECS_USE_BLUESKY` env var (resolved by
`geecs_scanner.engine.backend_selection`; explicit argument wins, default
legacy).  That path loads the selected shot-control YAML and passes it as
`shot_control_information`, and passes the `on_event` callback:
BlueskyScanner emits `ScanLifecycleEvent`s through it via `_set_state`,
shot-level `ScanStepEvent`s per event document via `_on_document` (so the
GUI progress bar works in Bluesky mode), and pre-flight `ScanDialogEvent`s
from the gateway-liveness check (both modes: each sync device's
`connected_status` — the gateway `CONNECTED` PV — is read pre-claim; free-run
adds a staleness stage for the trigger-must-be-free-running requirement)
(all defensive imports — headless
installs without geecs_scanner just skip emission).  `DeviceCommandEvent`
translation is deliberately skipped (no consumer).  Still not done in
Bluesky mode: `ActionControl` / setup-closeout actions on the legacy
`exec_config` path (ScanRequest submissions execute actions via the
delegated runner — see the engine-consolidation section).  Acquisition
mode is chosen by
the `GEECS_BLUESKY_ACQUISITION_MODE` env var — there is no GUI toggle for it
(intentional; bluesky is still being derisked).

### exec_config duck-typing

`reinitialize(exec_config)` accepts any object with:
- `.scan_config` — object with `scan_mode`, `device_var`, `start`, `end`, `step`,
  `wait_time`, `additional_description`
- `.options` — object with `rep_rate_hz`
- `.save_config` — object with `.Devices` dict (device name → config dict or
  Pydantic model with `variable_list`, `save_nonscalar_data`)

In production this is `ScanExecutionConfig`.  In the hardware integration test it
is a `SimpleNamespace` to avoid cross-package imports.

`SaveDeviceConfig.setup_action` and `SaveDeviceConfig.closeout_action` exist in
the current scanner model, but `BlueskyScanner` does not execute them yet.

### shots_per_step derivation

`ScanOptions` has no explicit shots field.  `shots_per_step` is derived as:
```python
max(1, round(rep_rate_hz * wait_time))
```

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
paths: the legacy single-device `ShotControlConfig` + one setter per
variable (state writes issued concurrently — byte-identical to the
pre-M3b behavior, and untouched for the scanner path), and
`ShotController.from_writes(ShotControlWrites)` — generalized per-state
**ordered** `(device, variable, value)` lists, possibly spanning several
devices (TriggerProfile semantics: writes replayed top to bottom, each
completing before the next; one cached `CaPutSetter` per distinct target).
`trigger_writes_from_profile` (scan_request_runner) adapts a TriggerProfile
into that shape; `GeecsSession.shot_control` accepts either generation.
Both `BlueskyScanner` and `GeecsSession` use it:

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

`BlueskyScanner._resolve_acquisition_mode` reads `options.acquisition_mode` with
a `GEECS_BLUESKY_ACQUISITION_MODE` env override (env wins), default strict.
`_classify_device_roles` assigns each save device a role from the mode: free-run
→ first sync device is `reference`, later sync devices are `contributor`
(`CaTimestampedReadable`), async are `snapshot`; strict → all sync are
`triggered` (`CaGenericDetector`).  STANDARD and NOSCAN share one
`_run_step_scan` body (NOSCAN = `motor=None`, one no-move bin).

### Tiled integration

`BlueskyScanner.__init__` reads `[tiled] uri` and `[tiled] api_key` from
`~/.config/geecs_python_api/config.ini` and subscribes a `TiledWriter` to the
`RunEngine`.  All event documents (start, descriptor, event, stop) are written to
the Tiled catalog at `http://192.168.6.14:8000`.  Silently skips if the server is
unreachable or `tiled[client]` is not installed.

### Threading model

The `RunEngine` runs in a background thread (`bluesky-scan`).  The RE's internal
`asyncio` event loop is persistent — devices are connected into it and remain
connected across the scan.  `RunEngine(context_managers=[])` disables SIGINT
handling, which fails when the RE is not on the main thread.

Device connect/disconnect uses `asyncio.run_coroutine_threadsafe(...).result(timeout=...)`.

## Device Layer

**Devices are CA-backed only** (`devices/ca/*`): stock ophyd-async
`epics_signal_r/rw` against the GeecsCAGateway PVs
(`[Experiment:]Device:Variable`, setpoints at `…:SP`). Requires the `ca` extra
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

## Transport Layer — moved to GeecsCAGateway

The GEECS wire-protocol transport (`GeecsUdpClient`, `GeecsTcpSubscriber`)
no longer lives in this package: it moved to
`GeecsCAGateway/geecs_ca_gateway/transport/`, alongside the DB layer and PV
naming.  This package touches GEECS devices **only** through the gateway's
CA PVs; it imports the gateway's library modules (`GeecsDb`, `pv_naming`,
wire-level exceptions) and never the transport or the server.  See
`GeecsCAGateway/README.md` and `GeecsCAGateway/DESIGN.md` for the protocol
details that used to be documented here.

## Test Infrastructure

`FakeGeecsServer` / `FakeGeecsDevice` (the in-process UDP/TCP server that
speaks the real GEECS wire protocol) also moved to GeecsCAGateway
(`geecs_ca_gateway.testing`).  This package's hermetic tests are built on
ophyd-async **mock backends** instead (`tests/ca_mock_helpers.py`) — see the
Device Layer section above.

### Hardware integration test

`test_bluesky_scanner.py` (top-level, run with `poetry run python
test_bluesky_scanner.py`) requires lab network access.  Tests three scenarios
against real hardware: NOSCAN, STANDARD step scan, NOSCAN with DG645 shot control.
The opt-in pytest case
`test_bluesky_scanner_full_output_hardware_integration` requires
`GEECS_BLUESKY_FULL_OUTPUT_TEST=1` and `poetry install --extras tiled`; it runs a
real STANDARD scan with native camera saving enabled and verifies the scan
folder, `ScanInfo`, `scan.log`, legacy scalar files, analysis s-file, saved
camera images, and event save-path metadata.

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

## Engine consolidation (0.22.0) — shim state

The event vocabulary (`ScanEvent` hierarchy, `ScanState`, `DialogRequest`)
lives in `geecs_bluesky/events.py`; `geecs_scanner.engine.scan_events` and
`geecs_scanner.engine.dialog_request` are **re-export shims** of the same
class objects (the legacy `DEVICE_COMMAND_ERRORS` tuple and
`escalate_device_error` stay in the shim — they need geecs_python_api).
The scanner's old defensive try/except imports are gone; the remaining
`is None` guards on the module-level names exist purely as test seams
(hermetic tests monkeypatch them to simulate a consumer-less install).

Operator interaction is one seam: `operator_channel.OperatorChannel`
(`EventStreamOperator` = today's GUI dialog behavior, `NullOperator` =
headless default-and-log).  Pre-flight is a pipeline
(`preflight.run_preflight`); new checks are list entries.

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
Remaining validated-then-refused v1 gaps: pseudo scan variables,
`all_scalars`, and optimize without an injected objective/suggester.
Actions on an optimize-mode request are **not** refused — optimize has no
action hooks yet, so the actions (request, experiment defaults, and
save-set rituals) are skipped, logged (WARNING), and recorded in run
metadata under `skipped_action_plans` (refusing would block every
optimization the moment an experiment defines default bracket actions;
unknown names still fail fast).  **As of 0.28.0 (M4 step i) the GUI bridge
delegates ScanRequest execution to `run_scan_request`**:
`reinitialize(ScanRequest)` validates every name fail-fast (discarding the
results) and stores the **original pre-defaults** request; the scan thread
runs it through the one engine definition, so actions, entry rituals,
multi-axis grids, db_scalars, and telemetry all execute through the bridge
too.  The bridge contributes its two seams via runner hooks:
`preflight(detectors, strict)` (operator-dialog pipeline, pre-claim,
`None` aborts; dropped devices are left connected — the runner's `finally`
owns disconnection) and `on_scan_start(total_steps, total_shots)` (GUI
progress totals).  The bridge never pre-claims on this path —
`session.scan` claims and self-attaches `scan.log`.  Optimize-mode
requests are still refused at reinitialize until GUI-submission step (iii)
wires the `optimization_loader` into the delegated path.  Experiment defaults
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
metadata records the list under `save_sets`.  The GUI bridge delegates to
`run_scan_request` (0.28.0), so the union — entry rituals included —
applies identically through the bridge.

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
  policy (GUI bridge / off-network) only the explicit list is recorded — M3b
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

## Known Gaps (as of 0.21.0)

The acquisition-modes architecture is complete and hardware-verified (both
modes, including single-shot; GUI-launched scans verified live 2026-07-06).
Remaining items are features/tuning, not architecture — see
`Planning/acquisition_modes/00_overview.md` "Deferred".

- **Strict single-shot needs an `ARMED` state** in the shot-control YAML to
  engage (the production experiment configs have one).  Without `ARMED` or a
  reachable shot-control device, strict aborts before acquisition
  (`GeecsConfigurationError`); use `free_run_time_sync` for free-running
  trigger acquisition.
- **`DeviceCommandEvent`s are not translated** (deliberate — the GUI does
  not render them; see `Planning/gui_stewardship/00_overview.md` §5).
  Lifecycle, step/progress, and pre-flight dialog events are emitted as of
  0.21.0.  Pre-flight liveness is CONNECTED-based (the gateway serves every
  DB device's data PVs whether or not the device is up, so CA-connect
  success never implied liveness); the 10 s staleness threshold now only
  matters for the free-run trigger-off check and still needs a lab session
  of tuning against real rep rates.
- **Scalar s-files are exported from Tiled best-effort** after a scan when the
  Tiled client extra is installed and the run can be read back.  Legacy TDMS
  output is not produced.
- **Background scan mode not implemented.**  Optimization runs as a scan via
  `GeecsSession.optimize` (adaptive scan: iteration = bin, same schema/data
  tree as any scan — see `plans/optimize.py`), both headless (suggester +
  objective in hand) and from the GUI: `BlueskyScanner` handles OPTIMIZATION
  scan mode through a GUI-injected `optimization_loader`
  (`geecs_scanner.optimization.session_bridge`), which runs the config-driven
  Xopt 3.1 / evaluator / ScanAnalysis stack against the session's bin rows.
  The evaluator seam is `EvaluatorDataSource` in
  `geecs_scanner.optimization.base_evaluator`; this package stays free of any
  geecs_scanner import (dependency direction).
- **Pre/post-scan action sequences run on the ScanRequest path only.**
  `GeecsSession.run(request)` executes setup/per_step/closeout ActionPlans
  (0.23.0), and since 0.28.0 the GUI bridge's delegated ScanRequest path
  does too; the legacy `exec_config` path still skips
  `setup_action` / `closeout_action` (legacy elements' actions *are*
  executed when the element is resolved as a save set through a
  ScanRequest — the converter extracts them into entry rituals).
- **Scan-folder creation invariant:** `claim_scan_number`
  (`plans/run_wrapper.py`) is the one place (outside the GUI's `ScanDataManager`)
  allowed to create a `scans/ScanNNN/` folder.  It logs a warning and returns
  `(None, None)` if `geecs_data_utils` is unavailable or the NetApp is not
  mounted.  Analysis-side code must still never create missing scan folders.
