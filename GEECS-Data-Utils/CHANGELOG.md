# Changelog — geecs-data-utils

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.6.0] — 2026-05-12

### Added
- `ScanPaths.build_device_file_map`, `ScanPaths.build_asset_filename`, and
  `ScanPaths.build_asset_path` accept an optional `device_file_stem` kwarg
  (default `None` → falls back to `device`). Use this when a device's data
  folder name differs from the token used inside per-shot filenames — e.g.,
  folder `U_BCaveMagSpec-interpSpec` containing files named
  `Scan042_U_BCaveMagSpec_001.csv`.
- `ScanData._append_expected_asset_columns`, `ScanData.set_data_frame`,
  `ScanData.load_scalars`, and `ScanData.from_date` accept an optional
  `stem_override: dict[str, str]` kwarg, mirroring the existing
  `ext_override` pattern. Maps device folder names to their in-filename
  stems so the `<device>_expected_path` DataFrame columns resolve to real
  files for affected devices. Without the override, those columns
  previously contained nonexistent paths for any device where the folder
  name and filename stem differ.

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
