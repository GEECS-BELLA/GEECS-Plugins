# Changelog — geecs-scanner

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.17.0] — 2026-05-11

### Changed
- **Python 3.11 now required.** `GeecsBluesky` (a hard dependency) requires
  `>=3.11` via `ophyd-async`; the `pyproject.toml` constraint is updated to
  reflect this. Users on 3.10 must upgrade before installing.

## [0.16.0] — 2026-05-08

### Changed — Bold Refactor: Delete and Extract

- **Phase 1 (ScanManager cleanup)**: Deleted `estimate_current_completion()` (never
  called anywhere) and the `trigger_off()` / `trigger_on()` proxy methods on
  `ScanManager` (pure one-line delegates to `TriggerController`).  The single
  internal call site in `_phase1_pre_scan()` was inlined.  Net: ~18 lines deleted.
- **Phase 2 (split `initialize_and_start_scan`)**: The 176-line method that mixed
  widget reads, YAML loading, config building, and scan submission was split into
  three named parts: `_collect_ui_scan_config()` (widget reads + YAML loading),
  `_build_exec_config()` (pure config construction), and a ~40-line orchestrator.
  Also fixed a latent `NameError` bug where `scan_config` was accessed outside its
  defining `if` block.
- **Phase 3 (AppController extraction)**: New `geecs_scanner/app/app_controller.py`.
  `AppController` owns the `RunControl` lifecycle (creation, all exception handling),
  database access (with `DatabaseDictLookup` fallback), scan submission, stop, and
  UI-coordination flags (`is_in_multiscan`, `is_in_action_library`, `is_starting`).
  `GEECSScannerWindow` now holds `self.controller: AppController` and exposes
  `@property RunControl` for backward-compatible read access.  Config-file write
  logic extracted to module-level `_write_config_if_changed()` helper.

## [0.15.0] — 2026-05-08

### Changed — Decompose Phase (D1–D5)

- **D1**: Added 18 behavioral tests for `DataLogger` in
  `tests/engine/test_data_logger.py` — covers `_log_device_data`, shot indexing,
  `FileMover` task queuing, and standby detection.
- **D2**: Extracted `FileMoveTask` and `FileMover` (~430 lines) from
  `data_logger.py` into `engine/file_mover.py`.  `ScanManager` now creates and
  owns the `FileMover`; `DataLogger.start_logging(file_mover=...)` accepts it as
  an injected dependency.  All `data_logger.file_mover.*` chains in `ScanManager`
  replaced with direct `self.file_mover.*` access.
- **D3**: Extracted `ScanLifecycleStateMachine` into `engine/lifecycle.py`.
  `ScanManager._state` / `_state_lock` / `_set_state()` / `current_state` all
  delegate to `self._lifecycle`.  Tests updated to test the state machine directly.
- **D4**: Split `ScanManager._start_scan()` (145 lines) into three named phase
  methods — `_phase1_pre_scan()`, `_resolve_scan_id()`, `_phase2_acquire()`.
  `_start_scan()` is now 12 lines and reads as a plain recipe.
- **D5**: Removed the 200 ms `QTimer` from `GEECSScannerWindow`.  Scan lifecycle
  state is now driven purely by events.  Mode transitions (multiscan, action library)
  and `RunControl` changes call `update_gui_status()` directly at the right moments.

## [0.14.0] — 2026-05-08

### Added
- Block 5: explicit state machine in `ScanManager`.  Added `_state: ScanState`,
  `_state_lock: threading.Lock`, `_set_state()`, and `current_state` property.
  All lifecycle transitions now go through `_set_state()`, which atomically
  updates `_state` and emits a `ScanLifecycleEvent`.
- `ScanState.PAUSED_ON_ERROR` — new state entered when `request_user_dialog()`
  blocks waiting for the operator's abort/continue decision.  The engine
  transitions to `STOPPING` (abort) or `RUNNING` (continue) after the dialog
  is dismissed.
- Block 7: event-driven GUI.  `GEECSScannerWindow` now subscribes to the scan
  event stream via a `pyqtSignal(object)` bridge; per-event handlers update
  status indicator colour, progress bar, button states, and restore-failure
  warnings instead of the 200 ms polling timer.
- `RunControl.__init__` accepts an `on_event` callback and passes it through
  to `ScanManager`.

### Removed
- `ScanManager.dialog_queue` — device-error dialogs are now delivered via
  `ScanDialogEvent` through the `on_event` callback / `pyqtSignal` bridge.
- `ScanManager.restore_failures` list — restore failures are now emitted as
  `ScanRestoreFailedEvent` s and accumulated by the GUI event handler.
- `RunControl.is_in_setup`, `is_busy()`, `is_in_stopping`, `is_stopping()`,
  `clear_stop_state()`, `get_progress()`, `is_active()` — all replaced by
  event-driven state tracking in `GEECSScannerWindow`.
- `GEECSScannerWindow._was_scanning` flag — no longer needed; the DONE/ABORTED
  lifecycle event is the authoritative transition signal.
- `import queue` from `scan_manager.py` and `geecs_scanner.py` — no longer
  used after `dialog_queue` removal.

## [0.13.0] — 2026-05-08

### Added
- Block 6: typed event stream for the scan engine.  `ScanEvent` dataclass
  hierarchy (`ScanLifecycleEvent`, `ScanStepEvent`, `DeviceCommandEvent`,
  `ScanErrorEvent`, `ScanRestoreFailedEvent`, `ScanDialogEvent`) defined in
  `engine/scan_events.py`.
- `ScanManager` accepts an optional `on_event: Callable[[ScanEvent], None]`
  callback (injected on construction).  Emits `ScanLifecycleEvent` at each
  state transition (INITIALIZING → RUNNING → DONE / ABORTED).
- `ScanStepExecutor.execute_scan_loop()` emits `ScanStepEvent` (phase
  "started" / "completed") before and after each step; carries `step_index`,
  `total_steps`, and `shots_completed`.
- `DeviceCommandExecutor.set()` / `.get()` emit `DeviceCommandEvent` on every
  outcome (accepted / rejected / timeout / failed).
- `_emit()` defensive wrapper — callback exceptions are caught and logged at
  DEBUG level; never propagate into the engine.
- Unit tests: `tests/engine/test_event_emission.py` — 17 network-free tests
  covering all `DeviceCommandExecutor` outcomes and the full
  `execute_scan_loop` event sequence.
- Hardware integration test skeleton:
  `tests/integration/hardware/test_scan_manager_hardware.py` — lifecycle and
  step-event assertions against live hardware (gated by
  `GEECS_HARDWARE_AVAILABLE` environment variable).
- Updated `tests/conftest.py` `hardware_available` fixture from always-skip
  stub to `GEECS_HARDWARE_AVAILABLE` env-var gate.
- Exported `ScanState`, `ScanLifecycleEvent`, `ScanStepEvent`,
  `DeviceCommandEvent`, `ScanEvent` from `geecs_scanner.engine`.

## [0.12.2] — 2026-05-08

### Changed
- `engine/data_logger.py`: replaced `geecs_python_api.tools.files.timestamping.extract_timestamp_from_file`
  with `geecs_data_utils.timestamp_from_filename`. Removes dependency on the
  deleted `GEECS-PythonAPI` timestamping module.

## [0.12.1] — 2026-05-08

### Changed
- Restored 14 log lines from DEBUG back to INFO across `action_manager`,
  `scan_data_manager`, `device_manager`, `scan_manager`, and `data_logger`.
  These were demoted in 0.9.1 to reduce terminal noise, but with the
  log-triage tooling in place the richer log signal is more valuable than
  the brevity. Restored lines: action sequence start/complete, pre-scan and
  closeout action attempts, configure-save-paths per device, scan-info file
  written, scan-info loaded, device unsubscribe lifecycle, orphan-file sweep
  start, FileMover worker started/stopped, per-device acq_timestamp during
  global time sync, and timestamp-tolerance-failed fallback message.

## [0.12.0] — 2026-05-08

### Added
- `DeviceCommandExecutor` — single policy object for all `device.set()` /
  `device.get()` calls during a scan.  Enforces a per-error-type retry
  policy: `GeecsDeviceCommandRejected` retries up to `max_retries` times;
  `GeecsDeviceExeTimeout` and `GeecsDeviceCommandFailed` escalate immediately
  (no retry, since retrying a hung or hardware-failed device makes it worse).
  The `escalate()` method routes failures to the operator dialog callback and
  sets the scan `stop_event` when the user chooses Abort.
- Exported from `geecs_scanner.engine` package.

### Changed
- `ScanStepExecutor`: replaced scattered `retry()` + `escalate_device_error()`
  calls in `move_devices_parallel_by_device` with `cmd_executor.set()` /
  `cmd_executor.escalate()`.  `max_retries` / `retry_delay` parameters removed
  from `move_devices_parallel_by_device` (now configured on the executor).
- `ScanDataManager.configure_device_save_paths`: fixed bug where the return
  value of the escalation callback was ignored, so an Abort choice never
  stopped the device-configuration loop.  Now correctly returns early on
  Abort and skips the failed device on Continue.
- `ActionManager._set_device`: routes device failures through
  `cmd_executor.set()` / `cmd_executor.escalate()`; re-raises on Abort so
  the enclosing action sequence fails cleanly.
- `ScanManager`: creates a `DeviceCommandExecutor` instance in `__init__` and
  injects it into `ScanStepExecutor`, `ScanDataManager`, and `ActionManager`.
  Also re-injects into the new `ScanDataManager` created during `reinitialize()`.

### Fixed
- `ActionControl`: `cmd_executor` was never wired into its standalone
  `ActionManager`, causing `AttributeError` on any action execution outside a
  scan.  Now creates a `DeviceCommandExecutor` at construction time.
- `ScanStepExecutor.move_devices_parallel_by_device`: `device.set()` can return
  `None` when the device raises `GeecsDeviceCommandFailed` in the UDP listener
  thread rather than the calling thread.  `None` is now treated as a command
  failure and raises `DeviceCommandError` so the normal escalation dialog fires
  instead of crashing with `TypeError` at the tolerance check.
- Scan log now includes scan config summary (device, range, step, wait, mode)
  at INFO level immediately after scan start — triage agent no longer needs to
  read the `.ini` file for context.
- Per-variable hardware set commands restored to INFO level in scan log
  (`[DeviceName] setting Var → value`); quieted previously by the async/file-mover
  noise reduction pass.
- Per-shot heartbeat (`shot N`) logged at INFO from `DataLogger` so acquisition
  progress is visible in the scan log during long wait-time steps.

## [0.11.0] — 2026-05-07

### Added
- `geecs_scanner.data_acquisition.trigger_controller.TriggerController` —
  encapsulates all shot-trigger interactions (`trigger_on`, `trigger_off`,
  `set_standby`, `singleshot`) previously spread across `ScanManager`.
  `ScanStepExecutor` now receives a `TriggerController` via constructor
  injection instead of dynamically-injected `trigger_on_fn`/`trigger_off_fn`
  callable attributes.

### Changed
- `ScanStepExecutor.__init__`: removed `shot_control` positional parameter;
  added `trigger_controller: Optional[TriggerController] = None`. The
  `hasattr` guard pattern for `trigger_on_fn`/`trigger_off_fn` is replaced
  with a typed `if self.trigger_controller is not None` check.
- `ScanManager`: `_set_trigger`, `trigger_on`, and `trigger_off` methods
  are now thin delegations to `self.trigger_controller`; `SINGLESHOT` and
  `STANDBY` states use `trigger_controller.singleshot()` and
  `trigger_controller.set_standby()`.
- `ActionManager`: removed `instantiated_devices` persistent device cache.
  Devices are now opened fresh at the start of each `execute_action` call
  (and `ping_devices_in_action_list` / `return_value`) and closed in a
  `finally` block regardless of success or failure.  This prevents stale
  TCP connections surviving across actions and removes the hidden cache that
  could hold devices open indefinitely.

## [0.10.0] — 2026-05-07

### Added
- `geecs_scanner.data_acquisition.scan_options.ScanOptions` — new Pydantic
  model for all engine-level execution options (`rep_rate_hz`,
  `enable_global_time_sync`, `global_time_tolerance_ms`, `master_control_ip`,
  `on_shot_tdms`, `save_direct_on_network`, `randomized_beeps`). Replaces the
  raw `options_dict: dict` that was previously threaded through
  `ScanManager` and `ScanStepExecutor`.

### Changed
- `ScanManager.__init__` and `reinitialize()` now accept and store a
  `ScanOptions` instance instead of a plain dict; all internal `.get("key")`
  accesses replaced with typed attribute access.
- `ScanStepExecutor.__init__` parameter `options_dict` renamed to `options:
  ScanOptions`; `on_shot_tdms` read via `options.on_shot_tdms`.
- `DataLogger.reinitialize_sound_player()` parameter changed from
  `Optional[dict]` to `Optional[ScanOptions]`.
- `SoundPlayer.__init__` no longer accepts an `options` dict; takes an explicit
  `randomized_beeps: bool` parameter instead.
- `RunControl.submit_run()` gains an optional `options: ScanOptions` parameter;
  when provided it is injected into the config dictionary before passing to
  `ScanManager.reinitialize()`.
- `GEECSScannerWindow._collect_options()` now returns a `ScanOptions` instance.
- Fixed bug where `On-Shot TDMS` and `Save Direct on Network` GUI toggles were
  defined in `BOOLEAN_OPTIONS` but absent from `OPTION_MAP`, so they were never
  collected and the engine always defaulted to `False`.
- Fixed bug where `scan_config.background` was set to a string `"True"/"False"`
  instead of a bool.

## [0.9.1] — 2026-05-07

### Changed
- Log level rationalization across all six `data_acquisition/` modules
  (`data_logger`, `device_manager`, `scan_manager`, `scan_executor`,
  `scan_data_manager`, `action_manager`).  Per-shot ticker events (standby
  checks, timestamp comparisons, async variable updates, file iteration,
  orphan retries, per-step device moves, optimizer chatter) demoted from INFO
  to DEBUG.  Two cases promoted to WARNING: file not found after all retries
  (orphaned to post-scan sweep), and missing `acq_timestamp` column during
  orphan sweep.  Lifecycle milestones (scan start/end/abort, device subscribed,
  config loaded, files written, devices synchronized) remain at INFO.
- Docstring purge in `data_acquisition/`: removed boilerplate NumPy-style
  docstrings from methods whose signatures and names are self-explanatory,
  leaving only non-obvious `Parameters`/`Returns`/`Raises` blocks and
  single-line summaries where warranted.

## [0.9.0] — 2026-05-07

### Added
- `geecs_scanner.utils.exceptions`: typed exception hierarchy rooted at `ScanError`.
  New types: `ConfigError`, `DeviceCommandError`, `TriggerError`,
  `DataFileError`. Existing names (`ActionError`, `ConflictingScanElements`,
  `ScanSetupError`, `OrphanProcessingTimeout`) are now subclasses of the
  hierarchy; all existing catch sites continue to work without change.
- `geecs_scanner.utils.retry`: `retry(fn, *, attempts, delay, backoff, catch,
  on_retry)` — centralizes retry-with-backoff logic for hardware call sites.
- `per_shot` analysis mode in `MultiDeviceScanEvaluator`: each image in a bin
  is analyzed individually instead of averaged, enabling richer per-shot
  statistical treatment (median, std dev, noise estimates for Xopt GP surrogate)
- `compute_objective_from_shots(scalar_results_list, bin_number)` hook on
  `MultiDeviceScanEvaluator`; default implementation mean-aggregates per-shot
  scalars and delegates to `compute_objective`, so existing subclasses require
  no changes when switching from `per_bin` to `per_shot`
- Mixed-mode support: when analyzers have different `analysis_mode` settings,
  `per_bin` scalars are merged into every shot dict before
  `compute_objective_from_shots` is called
- `ScalarLogEvaluator` — a new `BaseEvaluator` subclass that reads scalars
  directly from `log_entries` columns with no image analysis required; supports
  the same hook API (`compute_objective`, `compute_objective_from_shots`,
  `compute_observables`) and observables-only mode via `observables_only()`
- CI-friendly test suite (82 tests, no network or scan files):
  `test_base_evaluator`, `test_evaluator_get_scalar`, `test_evaluator_bax_mode`,
  `test_config_models`, `test_multi_device_scan_evaluator`,
  `test_scalar_log_evaluator`, `test_concrete_evaluators` (uses real
  `ImageAnalyzerResult` with synthetic scalars — no image files), plus shared
  fixtures (`FakeDataLogger`, `make_log_entries`) in `tests/optimization/conftest.py`

### Removed
- Legacy evaluators `ALine3_FWHM.py` and `HiResMagCam.py` (dead code, no known
  callers outside this repo; superseded by the `MultiDeviceScanEvaluator` pattern)
- `evaluation_mode` field removed from `BaseOptimizer` and `BaseOptimizerConfig`
  (was stored but never read; analysis mode is configured per-analyzer via
  `SingleDeviceScanAnalyzerConfig.analysis_mode`)
- Dead `move_devices()` method removed from `ScanStepExecutor`; only
  `move_devices_parallel_by_device()` remains (#291)

### Changed
- `DeviceSynchronizationError`, `DeviceSynchronizationTimeout`, and
  `ScanAbortedError` promoted from local definitions inside `scan_manager.py`
  to `geecs_scanner.utils.exceptions`. Import paths updated; no behaviour change.
- `move_devices_parallel_by_device()` uses `retry()` for hardware exceptions and
  raises `DeviceCommandError` (with chaining) on exhaustion; tolerance failures
  are logged as WARNING and no longer trigger a retry (#291)
- `_set_trigger()` in `ScanManager` uses `retry()` and raises `TriggerError` on
  exhaustion; `_start_scan()` catches `TriggerError` with `logger.critical` (#291)
- `FileMover._move_file()` retries `shutil.move()` on `OSError` (3 attempts,
  exponential backoff) and raises `DataFileError` on exhaustion; callers no longer
  silently discard move failures (#292)
- `FileMover._process_task()` and `_post_process_orphaned_files()` guard
  `home_dir.iterdir()` against `OSError` so a disconnected network share raises a
  typed exception rather than crashing a worker thread silently (#292)
- `ScanDataManager` filesystem failures (`initialize_tdms_writers`, `save_to_txt_and_h5`,
  `_make_sFile`) now chain into `DataFileError`; `process_results()` catches
  `DataFileError` explicitly before the broad handler (#292)
- `BaseEvaluator` stripped of dead-code methods (`_gather_shot_entries`,
  `validate_variable_keys_against_requirements`, `log_objective_result`,
  `get_device_shot_path`, `convert_log_entries_to_df`, `get_shotnumbers_for_bin`);
  `pandas` import moved inside `get_current_data` to avoid module-level import cost
- `BaseEvaluator` now owns the shared hook API: `compute_objective`,
  `compute_objective_from_shots` (default mean-aggregation), `compute_observables`
  (default empty dict), and `_compute_outputs` helper that handles objective
  computation, observable merging, and output-key shadowing checks; eliminates
  duplication that previously existed in both `MultiDeviceScanEvaluator` and
  `ScalarLogEvaluator`
- `MultiDeviceScanEvaluator`: unified `merged` slots approach replaces the
  `has_per_shot` branching; added `primary_device` property; `_get_value` now
  delegates to `_compute_outputs` after building the shot list
- `config_models.py`: `SaveDeviceConfig` import moved to `TYPE_CHECKING` + lazy
  inside `_load_and_check` to break the module-level chain to live DB connections

### Fixed
- `pint` pinned to `>=0.24` in `pyproject.toml`; lock file updated from 0.22 to
  0.24.4, resolving a `NumPy 2.0` incompatibility (`np.cumproduct` removal) that
  prevented `image_analysis.types` from being imported in tests
- `BaseOptimizerConfig.model_rebuild()` called at module load so Pydantic v2 can
  resolve the `SaveDeviceConfig` forward reference at validation time; previously
  raised `PydanticUserError` and prevented any optimization run from starting
- Closes #339

## [0.8.2] — 2026-04-15

### Fixed
- Orphan file-move tasks now drain in parallel instead of serially. The
  0.5 s retry delay was previously applied inside `move_files_by_timestamp`
  (the queueing call), which caused `_post_process_orphan_task` to sleep
  0.5 s per task before each enqueue — serialising the entire drain through
  a single thread regardless of the 16-worker pool. The sleep is now applied
  inside `_process_task` (the worker), so all orphan tasks are queued
  immediately and processed concurrently, improving end-of-scan drain
  throughput from ~2 files/s to ~20 files/s
- `_post_process_orphan_task` now resets `retry_count` to 0 before
  re-queuing each task, eliminating the 0.5 s per-worker sleep during the
  post-scan drain. No new files are written after the scan ends, so the
  delay serves no purpose and was costing ~2 s of wall time for a typical
  60-task backlog across 16 workers
- `PermissionError` on `file.is_file()` inside `_process_task` no longer
  crashes the entire task. When a device holds a write lock on a file at
  the moment the worker tries to stat it, the file is now skipped and the
  task continues; the retry or end-of-scan orphan sweep picks it up once
  the lock is released

## [0.8.1] — 2026-04-15

### Fixed
- **Qt thread-safety**: device-error dialogs are now shown on the Qt main thread
  instead of the scan worker thread, eliminating the production hang where
  `exec_()` blocked indefinitely (closes #312). Worker threads submit a
  `DialogRequest` to `ScanManager.dialog_queue` and block on a `threading.Event`;
  the 200 ms `update_gui_status` timer drains the queue and shows the dialog safely
- **Full command-error coverage**: all three error types (`GeecsDeviceExeTimeout`,
  `GeecsDeviceCommandRejected`, `GeecsDeviceCommandFailed`) are now caught and
  escalated with a Continue / Abort dialog at every device-set call site:
  `ScanStepExecutor`, `ScanManager._set_trigger`, `ScanDataManager.configure_device_save_paths`,
  `ActionManager._set_device`, and `ScanManager.restore_initial_state`
- **All queued variables shown in dialog**: when a device set fails, the dialog
  lists every variable queued for that device so the operator knows the full
  hardware state to check
- **Root-cause exception preserved across retries**: the first exception
  (e.g. `GeecsDeviceExeTimeout`) is tracked and shown even if subsequent retry
  attempts produce a different error type (e.g. `GeecsDeviceCommandRejected`)
- **`restore_initial_state` deadlock eliminated**: calling a blocking dialog from
  the scan worker thread while `stop_scanning_thread().join()` waited on the main
  thread caused a hard freeze. Restore failures are now collected non-blocking into
  `ScanManager.restore_failures`; a one-shot `QMessageBox` is shown by the main
  thread once the scan thread has exited
- **`GeecsDeviceInstantiationError` device name surfaced**: the reinitialize-failure
  dialog now shows the failing device name (e.g. "TCP connection test failed for
  UC_ModeImager") instead of the generic "Check log for problem device(s)"
- **Action library error consistency**: three previously inconsistent outcomes
  (GUI crash, silent log, popup) unified — `GeecsDeviceInstantiationError` no
  longer crashes via a missing `.message` attribute; `DEVICE_COMMAND_ERRORS`
  during action execution now surface the same error dialog via the
  `on_user_prompt` callback instead of silently logging

## [0.8.0] — 2026-04-13

### Fixed
- `stop_scan()` now writes scalar data files before any device interaction,
  ensuring `ScanData*.txt` and `s*.txt` are always produced even if a closeout
  action fails (closes #309)
- Closeout action failure no longer aborts the remaining shutdown sequence
- `_clear_existing_devices()` now disconnects devices in parallel via
  `ThreadPoolExecutor`, eliminating O(N) serial teardown (closes #308)
- `_stop_saving_devices()` dispatches `save=off` to all camera devices in
  parallel; per-command exceptions are caught individually so one failure does
  not skip remaining devices

### Removed
- Deprecated `save_data` boolean flag removed from `ScanManager`
