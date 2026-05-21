# GeecsBluesky — Developer Context for Claude

Bridges the GEECS hardware control system to the
[Bluesky](https://blueskyproject.io/) experiment orchestration ecosystem.
The primary product is `BlueskyScanner` — a drop-in replacement for
`ScanManager` that routes all scan execution through a Bluesky `RunEngine`.

## Why This Exists

The legacy `ScanManager` in `GEECS-Scanner-GUI/` is a monolith that mixes device
I/O, file writing, state management, and threading.  `BlueskyScanner` replaces it
with a Bluesky plan that is:

- **Testable without hardware** — `FakeGeecsServer` runs an in-process UDP/TCP
  device; all unit tests use it
- **Observable** — every shot emits a Bluesky event document; Tiled persists them
- **Composable** — plans (`geecs_step_scan`) are plain generators; wrappers
  (`bpp.finalize_wrapper`) guarantee cleanup on abort

## Package Layout

```
geecs_bluesky/
  scanner_bridge/
    bluesky_scanner.py      # BlueskyScanner — ScanManager-compatible public API
  plans/
    step_scan.py            # geecs_step_scan — step scan plan with arm/disarm
  devices/
    geecs_device.py         # GeecsDevice — StandardReadable base; shared UDP lifecycle
    settable.py             # GeecsSettable — Movable base; UDP set + polling convergence
    motor.py                # GeecsMotor — axis device; from_db_axis() factory
    generic_detector.py     # GeecsGenericDetector — dynamic variable list + save signals
    triggerable.py          # GeecsTriggerable — acq_timestamp-gated trigger
    camera.py               # GeecsCameraBase — thin camera subclass
  transport/
    udp_client.py           # GeecsUdpClient — asyncio UDP; two-stage ACK/EXE protocol
    tcp_subscriber.py       # GeecsTcpSubscriber — framed TCP push at ~5 Hz
  backends/                 # ophyd-async SignalBackend wired to UDP/TCP
  db/
    geecs_db.py             # GeecsDb — MySQL lookup: device name → (host, port)
  testing/
    fake_device_server.py   # FakeGeecsServer / FakeGeecsDevice — in-process test server
  signals.py                # geecs_signal_rw / geecs_signal_r helpers
  exceptions.py             # GeecsDeviceNotFoundError and friends
  utils.py                  # safe_name() — device name → valid Python identifier
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
`BlueskyScanner` via a `use_bluesky=True` flag.  The API surface is intentionally
identical so no GUI code changes are needed.

### exec_config duck-typing

`reinitialize(exec_config)` accepts any object with:
- `.scan_config` — object with `scan_mode`, `device_var`, `start`, `end`, `step`,
  `wait_time`, `additional_description`
- `.options` — object with `rep_rate_hz`
- `.save_config` — object with `.Devices` dict (device name → config dict or
  Pydantic model with `variable_list`, `save_nonscalar_data`)

In production this is `ScanExecutionConfig`.  In the hardware integration test it
is a `SimpleNamespace` to avoid cross-package imports.

### shots_per_step derivation

`ScanOptions` has no explicit shots field.  `shots_per_step` is derived as:
```python
max(1, round(rep_rate_hz * wait_time))
```

### Shot control — per-step arm/disarm

The DG645 fires autonomously at the machine rep rate.  Software controls which
state the outputs are in — `SCAN` (live) vs `STANDBY` (blocked) — so the trigger
is only active while shots are being collected, not during motor moves.

Flow per step:
```
bps.mv(motor, pos) → arm_trigger() [→ SCAN] → N × trigger_and_read → disarm_trigger() [→ STANDBY]
```

`_UdpSetter` is a minimal Bluesky `Movable` (has `.set(value) → AsyncStatus`) that
wraps a single GEECS variable over a shared `GeecsUdpClient`.  It intentionally
omits `.parent` (an ophyd-specific attribute) so `bps.abs_set` + `bps.wait` must
be used instead of `bps.mv` when setting shot control variables.

`_set_trigger_state(state)` drives all configured variables to a named state.
Empty-string values in the YAML are skipped — they mean "no-op for this state"
(matching legacy `TriggerController` behaviour).

A `bpp.finalize_wrapper` around the scan plan guarantees `disarm_trigger()` runs
even on mid-step abort.

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
it tracks the timestamp value, not a shot counter.

### GeecsMotor

`set(pos)` sends a UDP move command then polls `Position` until within tolerance
(`move_timeout=30 s`).  Factory: `GeecsMotor.from_db_axis(device, variable, name=...)`.

### GeecsGenericDetector

Dynamically creates one signal per variable in `variable_list`.  Optional
`save_nonscalar_data=True` adds `localsavingpath` and `save` signals; the
`_scan_with_saving` plan wrapper sets these before the scan and clears them in
a finalise wrapper.

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

## Known Gaps (as of 0.3.0)

- `shot_control_information` is not yet threaded through `RunControl` when
  `use_bluesky=True` — BlueskyScanner gets no shot control in GUI mode.
  Deferred until the GUI/RunControl refactor in the parallel worktree lands.
- Scalar s-files / TDMS output not produced — Bluesky writes to Tiled only.
  `ScanAnalysis` still reads s-files; data pipeline transition is an open question.
- Optimization and Background scan modes not implemented.
- Pre/post-scan action sequences not implemented.
- `_claim_scan_number()` logs an error and returns `None` silently if
  `geecs_data_utils` is unavailable or the NetApp is not mounted.
