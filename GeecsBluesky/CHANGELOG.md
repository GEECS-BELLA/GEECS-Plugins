# Changelog

All notable changes to `geecs-bluesky` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.3] - 2026-06-09

### Changed

- **GUI lifecycle events** — `BlueskyScanner` now accepts the GUI `on_event`
  callback, exposes `current_state`, and emits lifecycle transitions for
  initializing, running, completed, and aborted scans.  This lets the Scanner GUI
  re-enable its controls when a Bluesky-backed scan finishes.

## [0.3.2] - 2026-06-09

### Fixed

- **Windows MySQL connector crash** — GEECS database lookups now force
  `mysql-connector-python` to use its pure-Python implementation
  (`use_pure=True`), matching the legacy API layer.  The connector's C extension
  has crashed silently on lab Windows machines with 9.x.

### Tests

- Added a DB lookup regression test that verifies `use_pure=True` is passed to
  `mysql.connector.connect()`.

## [0.3.1] - 2026-06-08

### Added

- **Non-scalar save-path event metadata** — `GeecsGenericDetector` now emits
  derived per-event fields for non-scalar devices: device `acq_timestamp` and
  the configured save directory.  `BlueskyScanner` configures these fields when
  it assigns `localsavingpath` for `save_nonscalar_data=True` detectors, and
  includes the per-device save paths in run-start metadata.  File names remain
  hardware-native; downstream readers should join files to events by device
  `acq_timestamp`, not by a synthetic shot counter.

### Tests

- Added offline `FakeGeecsServer` coverage for non-scalar save-path metadata.

## [0.3.0] - 2026-05-08

### Added

- **DG645 shot control — per-step arm/disarm** — `BlueskyScanner` accepts an
  optional `shot_control_information` dict (matching the GEECS Scanner GUI timing
  YAML format).  The DG645 is armed to `SCAN` state after each motor move and
  disarmed to `STANDBY` after shots are collected, keeping the trigger off during
  motion.  A `bpp.finalize_wrapper` guarantees disarm even on mid-step abort.
- **`_UdpSetter`** — minimal Bluesky `Movable` wrapping a single GEECS UDP
  variable as a string-typed settable.  Used internally for shot control; avoids
  ophyd device overhead and works for both numeric delays and string state words.
- **`geecs_step_scan` arm/disarm parameters** — `arm_trigger` and `disarm_trigger`
  optional callables added to `geecs_step_scan`.  Each is a plan generator called
  after the motor move (arm) and after shots are collected (disarm) per step.
- **`BlueskyScanner._build_shot_controller()`** — resolves the shot control device
  from the GEECS MySQL DB and creates one `_UdpSetter` per configured variable.
- **`_set_trigger_state(state)`** — Bluesky plan stub that drives all shot control
  variables to a named state (`SCAN`, `STANDBY`, `OFF`, `SINGLESHOT`).  Empty-string
  values in the YAML are skipped (matching legacy `TriggerController` behaviour).
  Uses `bps.abs_set` + `bps.wait` rather than `bps.mv` to avoid the `.parent`
  attribute requirement of `bps.mv`.
- **`tests/test_shot_control.py`** — 10 unit tests covering `_UdpSetter`,
  `_set_trigger_state`, and arm/disarm ordering in `geecs_step_scan`.  All run
  against `FakeGeecsServer` — no hardware required.
- **`test_bluesky_scanner.py`** — hardware integration test with three scenarios:
  NOSCAN (UC_TopView), STANDARD step scan (U_ESP_JetXYZ 4→5 mm), and NOSCAN with
  DG645 shot control.  Verifies event counts, motor readback, `acq_timestamp`
  presence, and post-scan DG645 `Trigger.Source` state.  All 6 checks pass on
  real lab hardware.
- **`mysql-connector-python`** added as a direct Poetry dependency (previously
  required manual installation).

### Changed

- **`BlueskyScanner.reinitialize(exec_config)`** — now accepts a duck-typed
  `ScanExecutionConfig` object (or any `SimpleNamespace` with `.scan_config`,
  `.options`, `.save_config` attributes).  `shots_per_step` is derived from
  `rep_rate_hz × wait_time` (rounded, minimum 1) since `ScanOptions` has no
  explicit shots field.  Replaces the previous `config_dictionary` dict handoff.
- **`BlueskyScanner.start_scan_thread()`** — takes no arguments; uses the config
  stored by `reinitialize()`.

## [0.2.0] - 2026-05-07

### Added

- **TiledWriter integration** — `BlueskyScanner.__init__` now accepts optional
  `tiled_uri` and `tiled_api_key` parameters.  When `tiled_uri` is provided, a
  `bluesky.callbacks.tiled_writer.TiledWriter` is subscribed to the RunEngine
  so every scan is persisted to the Tiled catalog automatically.  Gracefully
  skips (logs a warning) if `tiled[client]` is not installed or the server is
  unreachable, so the scanner remains functional without Tiled.
- `tiled[client]` added as an optional Poetry dependency
  (`poetry install -E tiled` to enable).

## [0.1.0] - 2026-04-21

### Added

- Initial release: BlueskyScanner bridge, GeecsMotor, GeecsSettable,
  GeecsGenericDetector, GeecsTriggerable, TCP-backed signal cache, scan
  numbering, per-device image saving, STANDARD and NOSCAN scan modes.
