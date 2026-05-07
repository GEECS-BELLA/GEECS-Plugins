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
    lib/                      # Menu bar, UI utilities, action control helpers
  data_acquisition/
    scan_manager.py           # Central scan orchestrator (runs in a thread)
    device_manager.py         # Device subscriptions + config loading
    data_logger.py            # Per-shot data polling + file management
    scan_executor.py          # Step-by-step scan execution logic
    action_manager.py         # Automated action execution
    scan_data_manager.py      # Processed scan data handling
    database_dict_lookup.py   # GEECS device database query interface
    schemas/                  # Pydantic models for YAML-backed configs
    utils.py                  # Config path resolution, VISA helpers
  optimization/
    base_optimizer.py         # Xopt wrapper
    base_evaluator.py         # Objective function base class
    config_models.py          # Pydantic models for optimization configs
    evaluators/               # Image/beam analysis evaluators
    generators/               # Xopt algorithms: random, genetic, BAX
  utils/
    application_paths.py      # Path management (config + data folders)
    exceptions.py             # ActionError, ConflictingScanElements, etc.
    sound_player.py           # Audio feedback
  logging_setup.py            # Thread-safe centralized logging
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
  → initialize_and_start_scan()    (compiles config dict from GUI state)
  → RunControl.submit_run(config)
  → ScanManager.reinitialize(...)
  → ScanManager.start_scan_thread(scan_config)   ← new thread
      → DeviceManager.load_config()              (subscribe to devices)
      → ScanStepExecutor.execute_steps()
          For each step:
            Set device variable via TCP (geecs-pythonapi)
            Wait for stabilization
            DataLogger.log_data()               (poll, save to files)
            Move/rename files to daily folder
  GUI QTimer polls ScanManager.estimate_current_completion() every 200ms
  On completion: ScanManager signals → GUI re-enables controls
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
sessions. Validate with Pydantic schemas in `data_acquisition/schemas/`.

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

- `ActionError` — User-facing errors displayed in message boxes; scan aborts
- `ConflictingScanElements` — Two save elements share a device but incompatible
  configs; caught at scan start
- `GeecsDeviceInstantiationError` — TCP connection failed; wrapped as
  `ActionError` with friendly message

## Key Files for New Developers

1. `main.py` — logging init + Qt app launch
2. `geecs_scanner/app/geecs_scanner.py` — main window signal wiring
3. `geecs_scanner/data_acquisition/scan_manager.py` — scan orchestrator
4. `geecs_scanner/app/run_control.py` — GUI ↔ ScanManager bridge
5. `geecs_scanner/data_acquisition/data_logger.py` — real-time data acquisition
6. `geecs_scanner/app/save_element_editor.py` — pattern for YAML-backed dialogs
