# Changelog — geecs-scanner

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.8.1] — 2026-04-13

### Fixed
- Device-error dialogs (execution timeout, command rejected, command failed) are
  now shown on the Qt main thread instead of the scan worker thread, eliminating
  the production hang where `exec_()` blocked indefinitely (closes #312)
- `GeecsDeviceCommandRejected` and `GeecsDeviceCommandFailed` final-attempt
  escalations now surface a user dialog (Continue / Abort) instead of silently
  stopping the scan with no notification
- `ActionManager._prompt_user_quit_action` is now routed through the same
  main-thread dialog queue, fixing a parallel thread-safety issue
- Queue-drain mechanism lives in `GEECSScannerWindow.update_gui_status()`
  (200 ms QTimer); worker threads block on `threading.Event` until the user
  responds — zero busy-wait

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
