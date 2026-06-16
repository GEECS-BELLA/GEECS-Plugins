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
the mode.  The full design and rationale live in
`Planning/acquisition_modes/` (start at `00_overview.md`); `EVENT_SCHEMA.md` is
the canonical data contract.

- **`free_run_time_sync`** — the external trigger free-runs at the machine rep
  rate.  The first synchronous device is the **reference** (pacemaker): its
  `acq_timestamp` advance creates one event row; every other device fills that
  row's columns, each labeled with a derived `shot_id` / `shot_offset` /
  `valid` so late/slow devices are tolerated and realignable downstream.
- **`strict_shot_control`** — every device must be present on each shot.  With
  an `ARMED` state in the shot-control config it does true plan-owned
  single-shot (arm → confirm trigger quiescent → fire one shot → await all);
  without `ARMED` it falls back to `SCAN` + `trigger_and_read` on the
  free-running trigger.

NOSCAN ("statistics collection") is just a motorless step scan (one no-move
bin), so it honours the same mode dispatch.

## Why This Exists

The legacy `ScanManager` in `GEECS-Scanner-GUI/` is a monolith that mixes device
I/O, file writing, state management, and threading.  `BlueskyScanner` replaces it
with a Bluesky plan that is:

- **Testable without hardware** — `FakeGeecsServer` runs an in-process UDP/TCP
  device; all unit tests use it
- **Observable** — every shot emits a Bluesky event document; Tiled persists them
- **Composable** — plans are plain generators; the per-shot stubs
  (`geecs_single_shot`, `geecs_t0_sync`, …) and `geecs_run_wrapper` are the
  reusable unit, so a custom notebook plan inherits the same schema, scan
  numbering, and save-path discipline as a GUI scan

## Package Layout

```
geecs_bluesky/
  scanner_bridge/
    bluesky_scanner.py      # BlueskyScanner — ScanManager-compatible API; mode dispatch
  plans/
    step_scan.py            # geecs_step_scan — step scan (motor optional; arm/disarm/fire/setup hooks)
    free_run_step_scan.py   # geecs_free_run_step_scan — reference-paced + t0-sync + tail flush
    single_shot.py          # geecs_single_shot (fire→await→read) + geecs_confirm_quiescent
    t0_sync.py              # geecs_t0_sync — coordinated per-device t0 capture
    run_wrapper.py          # geecs_run_wrapper + claim_scan_number (numbering + save + md)
  devices/
    geecs_device.py         # GeecsDevice — StandardReadable base; shared UDP lifecycle
    settable.py             # GeecsSettable — Movable base; UDP set + polling convergence
    motor.py                # GeecsMotor — axis device; from_db_axis() factory
    generic_detector.py     # GeecsGenericDetector — dynamic vars + shot-id companion cols
    timestamped_readable.py # GeecsTimestampedReadable — free-run contributor (no blocking trigger)
    snapshot.py             # GeecsSnapshotReadable — async readback, sampled per row
    triggerable.py          # GeecsTriggerable — acq_timestamp-gated trigger
    shot_id.py              # ShotIdTracker (incremental shot ids) + ShotIdSupport mixin
    nonscalar_save.py       # NonScalarSaveSupport mixin — localsavingpath/save + save-path column
    scan_context.py         # ScanContext — bin_number / shot_index_in_bin / scan_event_index
    camera.py               # GeecsCameraBase — thin camera subclass
  models/
    shot_control.py         # ShotControlConfig / ShotControlState — validated shot-control YAML
  transport/
    udp_client.py           # GeecsUdpClient — asyncio UDP; two-stage ACK/EXE protocol
    tcp_subscriber.py       # GeecsTcpSubscriber — framed TCP push at ~5 Hz
  backends/                 # ophyd-async SignalBackend wired to UDP/TCP
  db/
    geecs_db.py             # GeecsDb — MySQL lookup: device name → (host, port)
  testing/
    fake_device_server.py   # FakeGeecsServer / FakeGeecsDevice — in-process test server
  signals.py                # geecs_signal_rw / geecs_signal_r helpers
  exceptions.py             # GeecsDeviceNotFoundError, GeecsT0SyncError, GeecsQuiescenceTimeoutError, …
  utils.py                  # safe_name() — device name → valid Python identifier

EVENT_SCHEMA.md             # canonical event-schema v1 data contract (read this)
```

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

`_UdpSetter` is a minimal Bluesky `Movable` (`.set(value) → AsyncStatus`) wrapping
one GEECS variable over the shared `GeecsUdpClient`.  It omits `.parent`, so
`_set_trigger_state` uses `bps.abs_set` + `bps.wait`, not `bps.mv`.  The scanner
exposes plan-stub callables built from it:

- `_arm_trigger` → `SCAN`, `_disarm_trigger` → `STANDBY` (per-step bracketing on
  the free-running modes; jet on during shots, off during moves)
- `_quiesce_trigger` → `OFF` (stops the free-run — used before free-run t0 sync;
  `STANDBY` keeps the trigger free-running on real hardware, so it cannot quiesce)
- `_arm_single_shot` → `ARMED` then `geecs_confirm_quiescent`, and
  `_fire_single_shot` → `SINGLESHOT` (strict plan-owned single-shot)

How they compose per mode:
```
free-run:  quiesce[OFF] → t0_sync → per step: mv → arm[SCAN] → N×(ref-paced read) → disarm[STANDBY] → tail flush
strict SS: setup once: arm[ARMED] → confirm quiescent → per shot: trigger→fire[SINGLESHOT]→await→read
strict TR: per step: mv → arm[SCAN] → N×trigger_and_read → disarm[STANDBY]   (no ARMED state)
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

### GeecsDevice

Base class (`StandardReadable`).  One shared `GeecsUdpClient` per device instance;
all signals share it.  The UDP client's `asyncio.Lock` serialises concurrent
get/set calls.

### GeecsTriggerable

`trigger()` waits for `acq_timestamp` to advance (event-driven via `asyncio.Queue`
populated by the TCP subscriber).  No polling.  Robust to device restarts because
it tracks the timestamp value, not a shot counter.  It drains stale frames and
baselines `acq_timestamp` **synchronously at call time**, so a shot fired right
after `bps.trigger` (the strict single-shot pattern) can't be missed.

### GeecsMotor

`set(pos)` sends a UDP move command then polls `Position` until within tolerance
(`move_timeout=30 s`).  Factory: `GeecsMotor.from_db_axis(device, variable, name=...)`.

### Shot IDs (`ShotIdTracker` / `ShotIdSupport`)

A device's `shot_id` is its physical trigger-opportunity number, derived
**incrementally** from its own `acq_timestamp` (`shot_id += round(Δt × rep_rate)`)
so rep-rate error never accumulates.  `ShotIdSupport` (mixin) adds
`configure_shot_id` / `seed_shot_id` / `last_acq_timestamp` and emits the
companion columns.  Cross-device matching is `shot_id` **equality** (each device
quantizes its own elapsed time against its own t0), not a timestamp tolerance —
the implicit window is the ±½-period rounding.  Per-device t0s are captured
together by `geecs_t0_sync` (free-run) or self-seeded on first read (strict).

### GeecsGenericDetector

Dynamically creates one signal per variable in `variable_list`, and (via
`ShotIdSupport`) emits the schema-v1 companion columns every read:
`<det>-acq_timestamp`, `<det>-t0_acq_timestamp`, `<det>-shot_id`,
`<det>-shot_offset`, `<det>-valid` — stable keys (NaN/`False` when underivable).
Native file saving comes from the shared `NonScalarSaveSupport` mixin
(`save_nonscalar_data=True` → `localsavingpath` / `save` signals +
`<det>-nonscalar_save_path` column); `geecs_run_wrapper` sets the save paths on
before the run and off in a finalize.  File names remain hardware-native — join
files to events by device `acq_timestamp`, never by `shot_id` (a matching/
diagnostic value, not a file key).

### GeecsTimestampedReadable

The free-run **contributor**: read like a snapshot (no blocking `trigger()`, so
it never gates the row), but it carries the same companion columns relative to
the reference.  `set_reference(ref)` anchors it (held via `ophyd_async.Reference`
to avoid child-adoption); `read()` peeks the reference's `shot_id`, optionally
grace-waits ~one TCP push period for a late frame to catch up, then labels its
own data with `shot_offset` / `valid`.  Supports `save_nonscalar_data` too.

## Transport Layer

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
  branch `add-bluesky-armed-shot-control`.  Without `ARMED`, strict uses the
  free-running `trigger_and_read` fallback.
- **Only lifecycle `ScanEvent`s are emitted** via `on_event` (through
  `_set_state`).  Per-shot/step Bluesky documents are not translated into the
  richer `ScanStepEvent` / `DeviceCommandEvent` stream (and may not need to be).
- **Shot-control bracketing is not yet extracted** for notebook reuse — the
  arm/disarm/quiesce/fire callables live in `BlueskyScanner` (built from
  `_UdpSetter` + `ShotControlConfig`).  Plans, devices, `geecs_run_wrapper`, and
  the schema are reusable; a future `ShotController` helper would give notebooks
  full parity (jet gating / single-shot firing).
- **Scalar s-files / TDMS output not produced** — Bluesky writes to Tiled only.
  `ScanAnalysis` still reads s-files; a Tiled→s-file exporter is deferred until
  the free-run event shape has survived real use.
- **Optimization and Background scan modes not implemented.**  For optimization,
  start from the unified `BaseEvaluator` / `BaseOptimizer` surface in
  `GEECS-Scanner-GUI`, not the removed `MultiDeviceScanEvaluator` /
  `ScalarLogEvaluator` split.
- **Pre/post-scan action sequences not implemented** (`setup_action` /
  `closeout_action`); the legacy scanner runs these through `ActionManager`.
- **Scan-folder creation invariant:** `claim_scan_number`
  (`plans/run_wrapper.py`) is the one place (outside the GUI's `ScanDataManager`)
  allowed to create a `scans/ScanNNN/` folder.  It logs a warning and returns
  `(None, None)` if `geecs_data_utils` is unavailable or the NetApp is not
  mounted.  Analysis-side code must still never create missing scan folders.
