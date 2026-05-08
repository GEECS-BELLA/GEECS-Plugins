# Changelog — geecs-data-utils

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] — 2026-05-08

### Added
- `timestamp_from_string(string)` and `timestamp_from_filename(file)` migrated
  from `geecs_python_api.tools.files.timestamping`. Both are exported from the
  package root. Eliminates the scanner's dependency on the now-deleted
  `GEECS-PythonAPI` timestamping module.
- `tests/test_utils.py` — first test suite for `geecs_data_utils`.

## [0.4.1] — 2026-05-07

### Changed
- `ScanConfig` migrated from `@dataclass` to `pydantic.BaseModel`.
  Construction syntax is unchanged (all fields use keyword arguments); the
  migration adds runtime validation and makes `ScanConfig` composable with other
  Pydantic models throughout the scanner engine.

## [0.4.0] — 2026-05-07

### Added
- `scan_log_loader` module providing `LogEntry`, `Severity`, `parse_lines`,
  `parse_scan_log`, and `load_scan_log`. Reads the per-scan log format
  written by `geecs_scanner.logging_setup.attach_scan_log` (multi-line
  tracebacks aggregated into the preceding record). Returned models are
  shared with the new `geecs-log-triage` subpackage and intended for any
  consumer needing to read scan logs (notebooks, plotting helpers,
  diagnostics tooling).

## [0.3.0] — 2026-05-06

### Added
- `GeecsPathsConfig` now reads an optional `wavekit_config_path` key from the
  `[Paths]` section of `config.ini` and exposes it as an attribute (consistent
  with the existing `frog_dll_path` / `frog_python32_path` pattern). Returns
  `None` if the key is absent or the path does not exist.

## [0.2.1] — current
<!-- Add entries here when changes are made -->
