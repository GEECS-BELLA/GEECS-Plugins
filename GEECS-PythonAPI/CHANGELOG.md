# Changelog — geecs-python-api

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
