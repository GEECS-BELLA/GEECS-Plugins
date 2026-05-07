# Changelog — geecs-scanner

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
