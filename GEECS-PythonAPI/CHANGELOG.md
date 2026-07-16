# Changelog — geecs-python-api

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.6.1] — 2026-07-16

### Changed

- Mechanical pre-commit normalization (repo-wide `pre-commit run --all-files`
  pass): trailing-whitespace / end-of-file / CRLF fixes, ruff-format
  reformatting, safe ruff autofixes (unused-import removal in the Hexapod
  testing notebook), and cleared notebook outputs. No behavior changes; no
  package code restructured (refactor moratorium respected).

## [0.6.0] — 2026-06-23

### Added
- Camera frame helpers in `geecs_python_api.controls.devices.camera` (both also
  exported from `geecs_python_api.controls.devices`), giving consumers a decoded
  image from a camera device without hand-rolling the wire-format handling. Both
  return/skip gracefully for non-camera devices, and live in a side-effect-free
  module (not methods on the generic `GeecsDevice`, and importable without
  hardware/DB) pending a dedicated camera device class:
  - `latest_image(device)` — **pull**: lazily decode the device's most recent
    frame (`device.state["image"]`) to a 2-D NumPy array via
    `geecs_data_utils.io.decode_imaq_image_string`. Returns `None` for non-camera
    devices and before the first frame arrives.
  - `on_image(device, callback)` — **push**: register an update listener that
    decodes each incoming frame and invokes `callback(image)`. Decodes from the
    raw message (the update listener fires before `state` is parsed, so reading
    `state["image"]` there would be one frame stale). Returns the listener name
    for `unregister_update_listener`. The callback runs on the TCP listener
    thread, so it should hand frames to a queue rather than block.

## [0.5.3] — 2026-06-23

### Fixed
- `TcpSubscriber.async_listener` now decodes subscription payloads as `latin-1`
  instead of `ascii` with `errors="replace"`. The old decode destroyed every
  byte ≥ `0x80`, corrupting binary image payloads (IMAQ "Flatten Image to
  String" frames) beyond recovery. `latin-1` is a lossless 1:1 byte↔char map and
  a strict ASCII superset, so existing text parsing is unaffected; recover raw
  bytes downstream with `msg.encode("latin-1")`.
- `GeecsDevice._subscription_parser` now tokenises the message body on the
  literal `nval,` / `nvar` delimiters instead of splitting on commas and `>>`.
  Binary variable values — notably IMAQ flattened camera images, whose pixel/JPEG
  bytes routinely contain comma (`0x2C`) and `>>` (`0x3E3E`) bytes — were
  previously truncated and silently dropped from the state dict (observed as an
  empty `image` from Basler cameras, and content-dependent drops elsewhere). The
  output dict structure is unchanged for existing scalar-only messages.

## [0.5.2] — 2026-05-11

### Fixed
- `EventHandler.unregister` no longer warns when called on a known event with
  an empty subscriber dict. The check `if not subs` was treating an empty `{}`
  the same as a missing event; corrected to `if event_name not in self.events`
  so the silent no-op matches the documented behaviour.

## [0.5.1] — 2026-05-11

### Fixed
- `GeecsDatabase._get_db()`: added `use_pure=True` to `mysql.connector.connect()`.
  The `mysql-connector-python` 9.x C extension DLL crashes silently on Windows,
  killing the process before any Python exception can be raised. The pure Python
  connector is functionally identical for all queries this codebase makes and is
  safe on all platforms.

## [0.5.0] — 2026-05-11

### Changed
- **Python 3.11 now required** when used alongside `geecs-scanner-gui` /
  `GeecsBluesky`. Constraint updated from `>=3.10` to `>=3.11`.
- `pyproject.toml`: bump `mysql-connector-python` from `^8.2.0` to `^9.7.0` to
  align with `GeecsBluesky` and make the full monorepo co-installable.

### Changed
- `pyproject.toml`: bump `mysql-connector-python` from `^8.2.0` to `^9.7.0` to
  align with `GeecsBluesky` and make the full monorepo co-installable.

## [0.4.3] — 2026-05-11 *(superseded by 0.5.0)*

*(Rolled into 0.5.0 — do not install this version separately.)*

## [0.4.2] — 2026-05-08

### Removed
- `api_defs.py`: dropped deprecated `ScanTag` backward-compat stub and its
  unconditional module-level `warnings.warn()` (fired on every import). No
  external callers found — `ScanTag` now lives exclusively in `geecs_data_utils`.
- `api_defs.py`: removed dead `exec_async()` helper and unused `dateutil` import.
- `controls/__init__.py`: removed `__getattr__` shim that re-exported `ScanTag`;
  no callers found across the monorepo.

## [0.4.1] — 2026-05-08

### Fixed
- `GeecsDevice._execute()`: when `is_valid()` is False (device not connected),
  now raises `GeecsDeviceInstantiationError` instead of silently returning
  `None`. Callers (`set()`, `get()`) could previously receive `None` and pass
  it through, masking the disconnected-device error.
- `UdpServer.__init__()`: added `SO_REUSEADDR` before `bind()`. On macOS and
  Windows, rapid open/close cycles between scans caused `OSError: [Errno 48]
  Address already in use` (Windows: WinError 10048) because the previous
  socket's port hadn't been released yet.

### Added
- Hardware integration test suite in
  `tests/geecs_python_api/controls/test_geecs_device_hardware.py`.
  Targets `U_S1H` (power supply). Run with `poetry run pytest --hardware`.
  Includes regression tests that reproduced both bugs before the fixes.
- `tests/conftest.py`: `--hardware` CLI flag and `geecs_exp_info` session
  fixture for database-backed hardware tests.
- `[tool.pytest.ini_options]` in `pyproject.toml` with `hardware` marker and
  `testpaths = ["tests"]`.

### Changed
- `pyproject.toml`: removed stale `{include = "labview_interface"}` package
  entry (module was deleted in 0.4.0).

## [0.4.0] — 2026-05-08

### Removed
- `htu_scripts/` — HTU-specific utility scripts (not part of the library API)
- `labview_interface/` — LabVIEW TCP/UDP bridge (unused, no known callers)
- `geecs_python_api/analysis/` — scan analysis utilities (superseded by
  `ScanAnalysis` package)
- `geecs_python_api/tools/` — distributions, file utilities, and interface
  helpers (timestamping functions migrated to `geecs_data_utils.utils`)
- `geecs_python_api/controls/experiment/` — `Experiment` wrapper class
  (unused outside HTU-specific scripts)
- `geecs_python_api/controls/devices/HTU/` — HTU device subclasses
  (experiment-specific, not a library concern)

### Changed
- `PW_scripts/` relocated to `extras/PW_scripts/` (not a library concern;
  preserved in `extras/` for ad-hoc reference)

## [0.3.1] — 2026-04-13

### Fixed
- `dequeue_command()` now catches `GeecsDeviceCommandRejected` (logged as
  warning) and bare `Exception` (logged as error) so rejected commands never
  produce unhandled "Exception in thread" output in daemon threads
- `_process_command()` guards against `dev_udp is None` (device already closed)
  with an early return instead of a bare `assert`, eliminating `AssertionError`
  tracebacks when dequeue threads outlive `device.close()`
