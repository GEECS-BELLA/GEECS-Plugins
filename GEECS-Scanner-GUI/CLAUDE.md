# GEECS-Scanner-GUI — Developer Context for Claude

PyQt5 data acquisition front-end for BELLA beamlines. Replaces Master Control's
rigid all-or-nothing scanning model with a flexible, opt-in, Python-driven
approach. Production-ready; runs on Windows in the lab.

## What It Does

- Define which devices to save per scan ("save elements")
- Execute scans: no-scan statistics, 1D parameter sweeps, Xopt optimization,
  background collection
- Configure triggered actions before/after scans (set variables, wait, check
  conditions)
- Organize multi-scan sequences via presets (MultiScanner)
- Organize data into per-day folders; optional Google Drive integration

## Package Layout

```
geecs_scanner/
  app/
    geecs_scanner.py          # Main window: GEECSScannerWindow(QMainWindow)
    run_control.py            # Orchestrates GUI ↔ BlueskyScanner handoff
    save_element_editor.py    # Dialog: define named groups of devices/variables
    scan_variable_editor.py   # Dialog: configure 1D scan variables + composite
    shot_control_editor.py    # Dialog: timing/trigger device configuration
    action_library.py         # Dialog: scripted action sequences
    multi_scanner.py          # Dialog: batch scan queue
    gui_dialogs.py            # Qt dialog helpers (device-error dialogs; main thread only)
    gui/                      # Qt Designer *_ui.py files (auto-generated)
    lib/
      action_control.py       # Standalone ActionManager wrapper (used by GUI outside scans)
      gui_utilities.py        # YAML loading, misc helpers
  engine/                     # What remains after G1 deleted the legacy scan
                              #   engine (scan_manager, data_logger, device_manager,
                              #   scan_executor, scan_data_manager, file_mover,
                              #   trigger_controller, lifecycle, default_scan_manager
                              #   — scan execution now lives in GeecsBluesky)
    action_manager.py         # Automated action execution (GUI-side, outside scans)
    device_command_executor.py # Command policy for the action path (retry/escalate)
    backend_selection.py      # Always-True stub kept for import compatibility
    scan_events.py            # Re-export shim of geecs_bluesky.events
    dialog_request.py         # DialogRequest re-export shim + DEVICE_COMMAND_ERRORS
    database_dict_lookup.py   # GEECS device database query interface
    models/
      scan_execution_config.py # Top-level validated scan config (GUI → engine handoff)
      scan_options.py          # Runtime knobs: rep rate, sync, save mode, etc.
      save_devices.py          # Device list + setup/closeout action config
      actions.py               # Action sequence Pydantic models
  optimization/
    base_optimizer.py         # Xopt wrapper
    base_evaluator.py         # Objective function base class + EvaluatorDataSource
    session_bridge.py         # The optimization_loader seam BlueskyScanner injects
    config_models.py          # Pydantic models for optimization configs
    evaluators/               # Image/beam analysis evaluators
    generators/               # Xopt algorithms: random, genetic, BAX
  utils/
    application_paths.py      # Path management (config + data folders)
    config_utils.py           # get_full_config_path, visa_config_generator
    exceptions.py             # Full typed exception hierarchy (ScanError subtypes)
    retry.py                  # Generic retry helper
    sound_player.py           # Audio feedback
  logging_setup.py            # Thread-safe centralized logging (QueueHandler → per-scan file)
main.py                       # Entry point
```

## Entry Point

```
main.py → GEECSScannerWindow.__init__() → show()
```

`main.py` initializes thread-safe logging, wraps `sys.excepthook` to route
uncaught exceptions to the logger, then creates and shows `GEECSScannerWindow`.

## Main Window: `GEECSScannerWindow`

- Reads experiment config from `~/.config/geecs_python_api/config.ini`
  (experiment name, repetition rate, timing device)
- Wires all UI signals to backend operations via Qt signals/slots
- Manages the list of available vs. selected save elements
- Four scan modes (radio buttons): **No-Scan**, **1D Scan**, **Optimization**,
  **Background**
- Event-driven status: the engine's `on_event` callback feeds the
  `_scan_event_received` pyqtSignal, whose main-thread handler
  (`_handle_scan_event`) updates the progress bar, status light, and buttons.
  There is **no polling timer** for scan state (the old 200 ms `QTimer` was
  removed in D5); the only remaining timer is a 10 s `scan_number_timer`
  that expires the cached scan-number display.
- Launches all auxiliary dialogs (SaveElementEditor, ShotControlEditor, etc.)

## Scan Execution Flow

```
User clicks Start
  → initialize_and_start_scan()        (builds ScanExecutionConfig from GUI state)
  → RunControl.submit_run(exec_config) (ScanExecutionConfig — validated Pydantic model)
  → BlueskyScanner.reinitialize(exec_config)   (fail-fast validation, no hardware)
  → BlueskyScanner.start_scan_thread()  ← new thread (the Bluesky RunEngine)
      device build + connect → pre-flight (operator dialogs) →
      scan-folder claim → plan execution (acquisition-mode dispatch,
      shot control, native device saving, Tiled persistence) → teardown
  GUI updates arrive as ScanEvents: on_event → _scan_event_received pyqtSignal
  → _handle_scan_event on the Qt main thread (no polling)
```

Everything inside the scan thread is GeecsBluesky's business — see
`GeecsBluesky/CLAUDE.md` for the engine architecture (acquisition modes,
plans, device layer, ScanRequest delegation).

### Scan backend

`BlueskyScanner` is the only backend (G1, scanner-gui 0.33.0, deleted the
legacy `ScanManager` engine). `engine/backend_selection.py` remains as an
always-True stub purely so old imports don't break; the `GEECS_USE_BLUESKY`
env var no longer does anything. The backend receives the GUI's `on_event`
callback and emits lifecycle, step/progress, and pre-flight dialog events
(no `DeviceCommandEvent` translation — deliberate; see
`Planning/gui_stewardship/00_overview.md`). `RunControl` submits either a
`ScanExecutionConfig` (legacy path) or a schema `ScanRequest` (delegated to
the engine's one runner).

## Threading Model

- **Main thread:** Qt event loop, all GUI updates, user input
- **Scan thread:** `BlueskyScanner` execution (the RunEngine and its
  persistent asyncio loop) — blocks on device I/O
- **Logging thread:** `QueueListener` background thread — safe log processing
  from any producer thread

Never update Qt widgets from the scan thread — use signals/slots or
`QMetaObject.invokeMethod` to bounce back to the main thread.

## Configuration File Hierarchy

All configs live in `~/.local/share/geecs_scanner/<Experiment>/` (or
`GEECS_DATA_PATH` env var):

```
<Experiment>/
  scan_devices/
    scan_devices.yaml          # 1D scan variable mappings
  composite_variables.yaml     # Multi-device expressions (numexpr)
  save_elements/               # One YAML per save element
  timing_configs/              # Shot control device definitions
  actions/                     # Action library YAMLs
  presets/                     # Full scan configs saved as presets
```

These are user-local and **not** checked into this repo. Changes persist between
sessions. Validate with Pydantic schemas in `engine/models/`.

## Key Dialogs

| Dialog | Purpose |
|---|---|
| `SaveElementEditor` | Create/edit named groups of device:variable pairs to log |
| `ScanVariableEditor` | Define what to sweep (device:variable, start/stop/step) or composite expressions |
| `ShotControlEditor` | Which device triggers shots; variable states for OFF/SCAN/STANDBY |
| `MultiScanner` | Queue multiple scan presets; run sequentially |
| `ActionLibrary` | Named action sequences: set variable, wait, check condition, call action |

## Optimization (Xopt Integration)

`optimization/base_optimizer.py` wraps [Xopt](https://github.com/ChristopherMayes/Xopt)
for Bayesian and genetic optimization loops.

- `BaseEvaluator` — concrete unified evaluator for both diagnostic-driven
  analyzers and direct scalar columns.  Subclasses usually implement
  `compute_objective()` and/or `compute_observables()`; per-shot variants exist
  for custom aggregation or filtering.  Loaded from a YAML config via
  `BaseOptimizer.from_config_file()`; the data source is the
  `EvaluatorDataSource` protocol (`base_evaluator.py`), fed by
  `optimization/session_bridge.py` from the Bluesky session's bin rows
  (the `scan_data_manager`/`data_logger` parameters are duck-typed
  stand-ins — the real classes were deleted with the legacy engine).
- Diagnostic data sources are configured with `analyzers: [...]` in evaluator
  kwargs.  Entries may be bare diagnostic stems or dict entries with
  `{diagnostic: X, ...overrides}`; they are loaded through
  `image_analysis.config.load_diagnostic()` and wrapped with
  `scan_analysis.config.create_scan_analyzer(..., use_injected_data=True)`.
- Direct scalar data sources are configured with `scalars: [...]` in evaluator
  kwargs and read from the current bin's rows via the `EvaluatorDataSource`
  seam.  Analyzer output keys are already namespaced by ScanAnalysis;
  scalar column names are used verbatim.
- Generators in `optimization/generators/` select next scan parameters
- Configured via Pydantic models in `optimization/config_models.py`
- `MultiDeviceScanEvaluator` and `ScalarLogEvaluator` were removed in 0.27.0;
  do not build new code or YAML around those names.

## Dependency Notes

- **PyQt5** — Windows-only in `pyproject.toml` (macOS/Linux can run with
  minor Qt binding changes but is not the target platform)
- **geecs-pythonapi** — Device TCP connections, database dict lookup. Use
  `ScanDevice` for subscribing to device variable streams.
- **ImageAnalysis** — Used in optimization evaluators for per-shot analysis
- **geecs-data-utils** — `ScanConfig`, `ScanMode` enums for scan parameters
- **Xopt** — Bayesian/genetic optimization algorithms
- **numexpr** — Fast evaluation of composite variable expressions

## Error Handling Patterns

In-scan device I/O is the engine's business (GeecsBluesky, via the gateway
`:SP` put primitive).  On the GUI side, device command failures in the
**action path** (the action library's Execute button, via
`app/lib/action_control.py`) flow through `DeviceCommandExecutor`
(`engine/device_command_executor.py`).  Never call `device.set()` /
`device.get()` directly from action logic — use `cmd_executor.set()` /
`cmd_executor.get()`.

Retry policy (per error type):
- `GeecsDeviceCommandRejected` → retry up to `max_retries` times (transient comms)
- `GeecsDeviceExeTimeout` → escalate immediately (device hung; retry makes it worse)
- `GeecsDeviceCommandFailed` → escalate immediately (hardware error)

Escalation wraps the exception in a `DialogRequest` and emits a
`ScanDialogEvent`; the GUI renders it on the Qt main thread
(`app/gui_dialogs.show_device_error_dialog`) while the worker thread blocks
on `request.response_event`.  `True` = Abort; `False` = Continue.  The
engine's pre-flight checks use the same `DialogRequest` mechanism.

Scanner exception hierarchy (`utils/exceptions.py`):
```
ScanError
├── ConfigError          user-visible config problems (wrong range, bad YAML)
│   ├── ActionError      wrong action name / failed GetStep
│   └── ConflictingScanElements
├── DeviceCommandError   hardware failure; wraps geecs_python_api errors
│   └── TriggerError     trigger failure; always scan-fatal
├── DeviceSynchronizationError
│   └── DeviceSynchronizationTimeout
├── ScanAbortedError     user requested stop
└── DataFileError        file / network I/O failure
    └── OrphanProcessingTimeout
```

`GeecsDeviceInstantiationError` (from the API layer) is caught at device-open
boundaries and re-raised or logged with context.

## Scan Event System

The engine emits a typed event stream via an `on_event` callback injected at
construction time.  The GUI consumes this stream: `RunControl` wires
`on_event` to the window's `_scan_event_received` pyqtSignal, and
`_handle_scan_event` dispatches to per-type handlers on the Qt main thread
(progress, status light, restore-failure warnings, device-error dialogs).
The same callback also serves headless consumers and tests.

### Event hierarchy (`geecs_bluesky.events`; `engine/scan_events.py` is a re-export shim)

```
ScanEvent
├── ScanLifecycleEvent   state: ScanState (INITIALIZING/RUNNING/STOPPING/DONE/ABORTED)
│                        total_shots: int  — non-zero only on INITIALIZING
├── ScanStepEvent        step_index, total_steps, shots_completed, phase ("started"/"completed")
├── DeviceCommandEvent   device, variable, outcome ("accepted"/"rejected"/"failed"/"timeout")
├── ScanErrorEvent       message, recoverable: bool, exc: BaseException | None
├── ScanRestoreFailedEvent  device, message
└── ScanDialogEvent      request: DialogRequest  — rendered by the GUI as a
                         modal device-error dialog (app/gui_dialogs.py)
```

`ScanState` is a `str, enum.Enum` — values serialise naturally to JSON / log messages.

### Usage

```python
from geecs_bluesky.scanner_bridge import BlueskyScanner

events = []
scanner = BlueskyScanner(
    experiment_dir="Undulator",
    shot_control_information=shot_info,
    on_event=events.append,
)
```

### Contract

- **`_emit()` is defensive** — callback exceptions are caught and logged at DEBUG;
  they never propagate into the scan engine.
- **`on_event` is optional** — passing `None` (the default) disables emission
  with no performance cost.
- **Who emits**: `BlueskyScanner` (lifecycle, step/progress, pre-flight
  dialogs); `DeviceCommandExecutor` (per-command outcomes on the GUI-side
  action path — the Bluesky scan path deliberately does not emit
  `DeviceCommandEvent`s).

### `total_shots` note

`ScanLifecycleEvent.total_shots` on the INITIALIZING event is an estimate:
`int(acquisition_time_seconds × rep_rate_hz)`.  The actual rep rate is
hardware/laser-driven; `rep_rate_hz` in `ScanOptions` is a software approximation
used to scale the count.  Treat `total_shots` as a progress denominator, not a
guaranteed shot count.

## Key Files for New Developers

1. `main.py` — logging init + Qt app launch
2. `geecs_scanner/app/geecs_scanner.py` — main window signal wiring
3. `geecs_scanner/app/run_control.py` — GUI ↔ BlueskyScanner bridge
4. `GeecsBluesky/geecs_bluesky/scanner_bridge/bluesky_scanner.py` — the scan engine's GUI face
5. `geecs_scanner/engine/models/scan_execution_config.py` — validated scan config model
6. `geecs_scanner/engine/device_command_executor.py` — action-path device command policy
7. `geecs_scanner/app/save_element_editor.py` — pattern for YAML-backed dialogs

## Current Direction: Stewardship

The old refactor roadmap ("Blocks", decompose steps D1–D5, tracked in the
now-deleted `Planning/STATUS.md`) **completed**: the typed event stream, the
`pyqtSignal` bridge, `FileMover` and `ScanLifecycleStateMachine` extraction,
phased `_start_scan()`, and removal of the 200 ms polling timer all landed.
The GUI is fully event-driven on the legacy path.

Current direction is stewardship, not decomposition — see
`Planning/gui_stewardship/00_overview.md` (repo root) for the audit,
strategic frame, and recommended investments. The legacy engine
(`ScanManager`/`DataLogger`/`DeviceManager`) was **deleted outright in G1**
(0.33.0, PR #487 — the CHANGELOG documents what was kept and why);
`BlueskyScanner` is the only backend. GUI investment goes into the durable
front-end jobs (config editing, progress display, operator dialogs,
optimization setup) — and this package as a whole is slated for replacement
by GEECS-Console (the greenfield PyQt front-end) at the M6 cutover.

## Known Tech Debt

- **Bluesky backend does not translate `DeviceCommandEvent`s** (deliberate —
  nothing consumes them).  Step/progress and pre-flight dialog events landed
  2026-07-07; see `Planning/gui_stewardship/00_overview.md` §4–5.
- **`enable_global_time_sync` / `global_time_tolerance_ms` are offered but
  not yet consumed by the Bluesky backend** — deliberately kept pending
  parity (#535); the other G1-orphaned knobs (`on_shot_tdms`,
  `save_direct_on_network`, `master_control_ip`/ECS dumps,
  `randomized_beeps`/`SoundPlayer`) were deleted in 0.35.0.  ECS dumps on
  the Bluesky path are still owed separately.
- **`GeecsDevice` `None` return on hardware rejection**: `device.set()` can
  return `None` when `GeecsDeviceCommandFailed` is raised in the UDP listener
  thread.  Root fix requires API changes (deferred with the python-api
  refactor).
