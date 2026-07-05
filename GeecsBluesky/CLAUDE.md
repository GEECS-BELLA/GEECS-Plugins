# GeecsBluesky — Developer Context for Claude

Bridges the GEECS hardware control system to the
[Bluesky](https://blueskyproject.io/) experiment orchestration ecosystem.
The primary product is `BlueskyScanner` — a RunEngine-backed scan executor
designed to become a `ScanManager` replacement.  It runs from the
`GEECS-Scanner-GUI` (`use_bluesky=True`) and has been hardware-verified for both
acquisition modes (free-run and strict) including DG645 shot control.

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
  scanner_bridge/
    bluesky_scanner.py      # BlueskyScanner — ScanManager-compatible GUI bridge
  plans/
    orchestration.py        # build_step_scan_plan — THE one scan recipe (both front doors)
    step_scan.py            # geecs_step_scan — step scan (motor optional; hooks)
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
    shot_id.py              # ShotIdTracker + ShotIdSupport mixin (schema-v1 columns)
    nonscalar_save.py       # NonScalarSaveSupport mixin — save-path column + asset docs
    contributor.py          # FreeRunContributorSupport — reference-relative labeling
    scan_context.py         # ScanContext — bin_number / shot_index_in_bin / scan_event_index
  shot_controller.py        # ShotController — arm/disarm/quiesce/fire plan stubs (gateway :SP)
  optimize.py               # suggester protocol, RandomSuggester, XoptSuggester, BinData
  tiled_integration.py      # subscribe_tiled + descriptor patch + safe callback
  data_paths.py             # local ↔ device-server path mapping, asset roots
  scanner_configs.py        # configs-repo resolution + shot-control YAML loading
  models/
    shot_control.py         # ShotControlConfig / ShotControlState — validated YAML
  exceptions.py             # scan-level errors; wire-level ones re-exported from the gateway
  utils.py                  # safe_name() / build_signal_attrs()

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
`BlueskyScanner` via `use_bluesky=True`.  That path now loads the selected
shot-control YAML and passes it as `shot_control_information`, and passes the
`on_event` callback (BlueskyScanner emits `ScanLifecycleEvent`s through it via
`_set_state`).  Still not done in Bluesky mode: `ActionControl` /
setup-closeout actions, and translating per-shot Bluesky documents into the
richer `ScanStepEvent` / `DeviceCommandEvent` stream (only lifecycle events are
emitted).  Acquisition mode is chosen by the `GEECS_BLUESKY_ACQUISITION_MODE`
env var — there is no GUI toggle (intentional; bluesky is experimental).

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

`ShotController` (`shot_controller.py`) drives the shot-control device through
its named states as plan stubs, via `CaPutSetter`s writing the gateway `:SP`
PVs (put-completion rides GEECS's blocking set). Both `BlueskyScanner` and
`GeecsSession` use it:

- `arm()` → `SCAN`, `disarm()` → `STANDBY` (per-step bracketing on the
  free-running modes; jet on during shots, off during moves)
- `quiesce()` → `OFF` (stops the free-run — used before free-run t0 sync;
  `STANDBY` keeps the trigger free-running on real hardware, so it cannot quiesce)
- `arm_single_shot(detectors)` → `ARMED` then `geecs_confirm_quiescent`, and
  `fire_shot()` → `SINGLESHOT` (strict plan-owned single-shot)

How they compose per mode:
```
free-run:  quiesce[OFF] → t0_sync → per step: mv → arm[SCAN] → N×(ref-paced read) → disarm[STANDBY] → tail flush
strict:    setup once: arm[ARMED] → confirm quiescent → per shot: trigger→fire[SINGLESHOT]→await→read
```

A `bpp.finalize_wrapper` around the plan guarantees the disarm (→ `STANDBY`)
runs even on mid-scan abort.

`ARMED` is **config-specific**: it sets data-taking output (jet amplitude /
delay) + the single-shot trigger source — *external* single-shot when the laser
is on, *internal* (`Single shot`) when off.  The Python is agnostic; the
difference lives entirely in the per-config YAML.

### Acquisition-mode dispatch

`BlueskyScanner._resolve_acquisition_mode` reads `options.acquisition_mode` with
a `GEECS_BLUESKY_ACQUISITION_MODE` env override (env wins), default strict.
`_classify_device_roles` assigns each save device a role from the mode: free-run
→ first sync device is `reference`, later sync devices are `contributor`
(`GeecsTimestampedReadable`), async are `snapshot`; strict → all sync are
`triggered` (`GeecsGenericDetector`).  STANDARD and NOSCAN share one
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

Shot IDs (`ShotIdTracker`): a device's `shot_id` is its physical
trigger-opportunity number, derived **incrementally** from its own
`acq_timestamp` (`shot_id += round(Δt × rep_rate)`) so rep-rate error never
accumulates. Cross-device matching is `shot_id` **equality**; files join to
events by device `acq_timestamp`, never by `shot_id`.

Hermetic testing uses ophyd-async mock backends (`tests/ca_mock_helpers.py`):
`set_mock_value` on `acq_timestamp` is a shot, `start_pacer` on the RE loop is
the free-running trigger, `follow_setpoint` stands in for GEECS convergence.

## Transport Layer (gateway-facing — devices do not use it)

### GeecsUdpClient

Two-stage protocol: command sent to port N, ACK received on port N, EXE response
received on port N+1.  ACK format: `"get{var}>>>>accepted"`.  EXE success:
`"no error,"`.  Holds an `asyncio.Lock` for serialisation.

Local IP detected at connect time via a no-op UDP socket against the OS routing
table — handles VPN/PPP lab links.

### GeecsTcpSubscriber

Framed TCP: 4-byte big-endian length prefix + JSON payload.  Pushed by device at
~5 Hz.  Feeds a shared `_shot_cache` dict; `GeecsTriggerable` watches for
`acq_timestamp` advances via an `asyncio.Queue`.

## Test Infrastructure

### FakeGeecsServer / FakeGeecsDevice

An in-process UDP/TCP server that speaks the real GEECS wire protocol.  Use as an
async context manager:

```python
device = FakeGeecsDevice("U_DG645", variables={"Trigger.Source": "External rising edges"})
async with FakeGeecsServer(device) as srv:
    udp = GeecsUdpClient(srv.host, srv.port, device_name="U_DG645")
    await udp.connect()
    val = await udp.get("Trigger.Source")
```

`device.fire_shot()` advances `acq_timestamp` and broadcasts a TCP push — used to
simulate hardware shot events in tests.

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

## Known Gaps (as of 0.8.0)

The acquisition-modes architecture is complete and hardware-verified (both
modes, including single-shot).  Remaining items are features/tuning, not
architecture — see `Planning/acquisition_modes/00_overview.md` "Deferred".

- **Strict single-shot needs an `ARMED` state** in the shot-control YAML to
  engage.  The experiment configs gained one on the `geecs-plugins-configs`
  branch `add-bluesky-armed-shot-control`.  Without `ARMED` or a reachable
  shot-control device, strict aborts before acquisition.
- **Only lifecycle `ScanEvent`s are emitted** via `on_event` (through
  `_set_state`).  Per-shot/step Bluesky documents are not translated into the
  richer `ScanStepEvent` / `DeviceCommandEvent` stream (and may not need to be).
- **Scalar s-files are exported from Tiled best-effort** after a scan when the
  Tiled client extra is installed and the run can be read back.  Legacy TDMS
  output is not produced.
- **Background scan mode not implemented.**  Optimization runs headless via
  `GeecsSession.optimize` (adaptive scan: iteration = bin, same schema/data
  tree as any scan; exploratory — see `plans/optimize.py`).  The legacy GUI
  optimizer surface is untouched and separate.
- **Pre/post-scan action sequences not implemented** (`setup_action` /
  `closeout_action`); the legacy scanner runs these through `ActionManager`.
- **Scan-folder creation invariant:** `claim_scan_number`
  (`plans/run_wrapper.py`) is the one place (outside the GUI's `ScanDataManager`)
  allowed to create a `scans/ScanNNN/` folder.  It logs a warning and returns
  `(None, None)` if `geecs_data_utils` is unavailable or the NetApp is not
  mounted.  Analysis-side code must still never create missing scan folders.
