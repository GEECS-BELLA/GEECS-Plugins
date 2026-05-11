# Changelog — geecs-python-api

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
