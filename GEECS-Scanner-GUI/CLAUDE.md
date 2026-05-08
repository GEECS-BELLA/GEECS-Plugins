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
    run_control.py            # Orchestrates GUI ↔ ScanManager handoff
    save_element_editor.py    # Dialog: define named groups of devices/variables
    scan_variable_editor.py   # Dialog: configure 1D scan variables + composite
    shot_control_editor.py    # Dialog: timing/trigger device configuration
    action_library.py         # Dialog: scripted action sequences
    multi_scanner.py          # Dialog: batch scan queue
    gui/                      # Qt Designer *_ui.py files (auto-generated)
    lib/
      action_control.py       # Standalone ActionManager wrapper (used by GUI outside scans)
      gui_utilities.py        # YAML loading, misc helpers
  engine/
    scan_manager.py           # Central scan orchestrator (runs in a thread)
    device_manager.py         # Device subscriptions + config loading
    data_logger.py            # Per-shot data polling + file management
    scan_executor.py          # Step-by-step scan execution logic
    action_manager.py         # Automated action execution
    scan_data_manager.py      # Output path setup, data conversion, file I/O
    device_command_executor.py # Single policy point for all device.set()/get() calls
    scan_events.py            # Typed ScanEvent hierarchy + ScanState enum (Block 3)
    database_dict_lookup.py   # GEECS device database query interface
    trigger_controller.py     # Timing/trigger management (OFF/STANDBY/SCAN/SINGLESHOT)
    dialog_request.py         # Thread-safe dialog request type + escalation helpers
    models/
      scan_execution_config.py # Top-level validated scan config (GUI → engine handoff)
      scan_options.py          # Runtime knobs: rep rate, sync, save mode, etc.
      save_devices.py          # Device list + setup/closeout action config
      actions.py               # Action sequence Pydantic models
  optimization/
    base_optimizer.py         # Xopt wrapper
    base_evaluator.py         # Objective function base class
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
- 200ms `QTimer` polling `update_gui_status()` → updates progress bar and
  status text without blocking the Qt event loop
- Launches all auxiliary dialogs (SaveElementEditor, ShotControlEditor, etc.)

## Scan Execution Flow

```
User clicks Start
  → initialize_and_start_scan()        (builds ScanExecutionConfig from GUI state)
  → RunControl.submit_run(exec_config) (ScanExecutionConfig — validated Pydantic model)
  → ScanManager.reinitialize(exec_config)
      → DeviceManager.reinitialize()   (subscribe to save devices)
  → ScanManager.start_scan_thread()    ← new thread
      Phase 1 (outside scan log):
        trigger_off()
        ScanDataManager.initialize_scan_data_and_output_files()  (creates scan folder)
      Phase 2 (inside per-scan scan.log):
        pre_logging_setup()
          → ScanDataManager.configure_device_save_paths()  (via DeviceCommandExecutor)
          → DeviceManager.handle_scan_variables()
          → ActionManager.execute_action("setup_action")   (via DeviceCommandExecutor)
        DataLogger.start_logging()
        ScanStepExecutor.execute_scan_loop()
          For each step:
            DeviceCommandExecutor.set()   (retry/escalate per error type)
            wait_for_acquisition()
            DataLogger logs per-shot heartbeat + updates shot_id context
        stop_scan()                       (data written before device teardown)
  GUI QTimer polls ScanManager.estimate_current_completion() every 200ms
```

## Threading Model

- **Main thread:** Qt event loop, all GUI updates, user input
- **Scan thread:** `ScanManager` execution — blocks on device I/O and file ops
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

- `BaseEvaluator` — template-method base; subclasses implement `_get_value()`.
  Owns the shared hook API: `compute_objective`, `compute_objective_from_shots`
  (default mean-aggregation), `compute_observables` (default empty), and
  `_compute_outputs` which merges objective + observables into the final dict.
  Loaded from a YAML config via `BaseOptimizer.from_config_file()`; `DataLogger`
  and `ScanDataManager` are injected at that point — evaluators are purely reactive
  (they read whatever `DataLogger.bin_num` says is current).
- `MultiDeviceScanEvaluator` — runs one or more `SingleDeviceScanAnalyzer`s and
  builds a per-shot scalar list for `_compute_outputs`; supports `per_bin` and
  `per_shot` analysis modes per analyzer, with mixed-mode merging.
- `ScalarLogEvaluator` — reads scalars directly from `DataLogger.log_entries`
  columns; no image analysis required. Same hook API as above.
- Generators in `optimization/generators/` select next scan parameters
- Configured via Pydantic models in `optimization/config_models.py`

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

All device command failures during a scan flow through `DeviceCommandExecutor`
(`engine/device_command_executor.py`).  Never call `device.set()` / `device.get()`
directly from scan logic — use `self.cmd_executor.set()` / `self.cmd_executor.get()`.

Retry policy (per error type):
- `GeecsDeviceCommandRejected` → retry up to `max_retries` times (transient comms)
- `GeecsDeviceExeTimeout` → escalate immediately (device hung; retry makes it worse)
- `GeecsDeviceCommandFailed` → escalate immediately (hardware error)

Escalation calls `on_escalate(exc, context) -> bool` on the main thread via
`ScanManager.dialog_queue` + `DialogRequest`.  `True` = Abort (sets `stop_event`);
`False` = Continue.

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

## Scan Event System (Block 3)

The engine emits a typed event stream via an `on_event` callback injected at
construction time.  The GUI does not yet consume this stream (Block 7 will replace
the 200 ms polling timer); the callback is intended for headless consumers, tests,
and future GUI wiring.

### Event hierarchy (`engine/scan_events.py`)

```
ScanEvent
├── ScanLifecycleEvent   state: ScanState (INITIALIZING/RUNNING/STOPPING/DONE/ABORTED)
│                        total_shots: int  — non-zero only on INITIALIZING
├── ScanStepEvent        step_index, total_steps, shots_completed, phase ("started"/"completed")
├── DeviceCommandEvent   device, variable, outcome ("accepted"/"rejected"/"failed"/"timeout")
├── ScanErrorEvent       message, recoverable: bool, exc: BaseException | None
├── ScanRestoreFailedEvent  device, message
└── ScanDialogEvent      request: DialogRequest  — for Block 7 GUI wiring
```

`ScanState` is a `str, enum.Enum` — values serialise naturally to JSON / log messages.

### Usage

```python
from geecs_scanner.engine import ScanManager, ScanLifecycleEvent, ScanStepEvent

events = []
mgr = ScanManager(
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
- **Three classes emit**: `ScanManager` (lifecycle), `ScanStepExecutor` (step
  boundaries), `DeviceCommandExecutor` (per-command outcomes).

### `total_shots` note

`ScanLifecycleEvent.total_shots` on the INITIALIZING event is an estimate:
`int(acquisition_time_seconds × rep_rate_hz)`.  The actual rep rate is
hardware/laser-driven; `rep_rate_hz` in `ScanOptions` is a software approximation
used to scale the count.  Treat `total_shots` as a progress denominator, not a
guaranteed shot count.

## Key Files for New Developers

1. `main.py` — logging init + Qt app launch
2. `geecs_scanner/app/geecs_scanner.py` — main window signal wiring
3. `geecs_scanner/engine/scan_manager.py` — scan orchestrator
4. `geecs_scanner/app/run_control.py` — GUI ↔ ScanManager bridge
5. `geecs_scanner/engine/device_command_executor.py` — all device command policy
6. `geecs_scanner/engine/scan_events.py` — typed event hierarchy + ScanState enum
7. `geecs_scanner/engine/data_logger.py` — real-time data acquisition
8. `geecs_scanner/engine/models/scan_execution_config.py` — validated scan config model
9. `geecs_scanner/app/save_element_editor.py` — pattern for YAML-backed dialogs

## Known Tech Debt

- `on_device_error` attribute still exists on `ScanStepExecutor` and
  `ScanDataManager` and is set by `ScanManager.__init__`, but is no longer
  used internally — both classes now go through `cmd_executor`.  Safe to
  remove in a follow-up cleanup.
- `ScanManager` is still the largest single file (~1100 lines) and mixes
  orchestration, state management, and lifecycle bookkeeping.  See ROADMAP.md
  Block 5 for the planned state machine decomposition.
- The `GeecsDevice` UDP listener raises `GeecsDeviceCommandFailed` in a
  separate thread, so `device.set()` can silently return `None` on hardware
  rejection.  Guarded in `scan_executor.py` with a `None` check, but the
  root fix requires GeecsDevice API changes (see ROADMAP.md — API cleanup).
