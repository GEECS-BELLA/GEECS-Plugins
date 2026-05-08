# Changelog

All notable changes to `geecs-log-triage` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-05-08

### Fixed
- `OSError` (covers `WinError 10048` stale-socket) added to
  `CLASSIFICATION_MAP` as `HARDWARE_ISSUE` — previously classified as
  `UNKNOWN`.
- `"listen called with no socket bound"` and `"WinError 10048"` added to
  `_MESSAGE_HINTS` so the cascading no-socket-bound errors in the same scan
  are also classified as `HARDWARE_ISSUE`.

## [0.2.0] - 2026-05-08

### Added
- `render.py` — `render_markdown(report)` renders a `TriageReport` as a
  human-readable Markdown document, grouped by classification with per-fingerprint
  subsections showing affected scans, message, and (collapsible) sample traceback.
- `harvester.day_folder_for(date, experiment)` — returns the day folder path
  (parent of `scans/`) for use by the CLI and external consumers.
- `harvester.harvest_scan(date, experiment, scan_number)` — harvests a single
  scan by number (matches `Scan042` and `Scan42` forms) without requiring the
  caller to know the full path.
- CLI `--scan N` argument: triage a single scan by number within `--date`.
- CLI `--format {json,md}` argument (default `md`): select output format.
- CLI auto-output: when `--date` + `--experiment` are given and `--output` is
  omitted, the report is written to `{day_folder}/triage.md` automatically and
  the path is echoed to stderr.

### Changed
- `cli._emit` now routes through `render_markdown` when `--format md` is
  requested; JSON format behaviour unchanged.
- `_iter_scan_folders_for_date` refactored to call `day_folder_for` internally
  (no behaviour change).
- `render_markdown`, `day_folder_for`, `harvest_scan` exported from
  `geecs_log_triage.__init__`.

## [0.1.1] - 2026-05-07

### Changed
- `classifier.CLASSIFICATION_MAP` expanded with all `geecs_python_api` hardware
  exception names (`GeecsDeviceExeTimeout`, `GeecsDeviceCommandRejected`,
  `GeecsDeviceCommandFailed`) and the new `geecs_scanner` typed exceptions
  (`DeviceCommandError`, `TriggerError`, `DataFileError`, etc.).
- `_MESSAGE_HINTS` expanded with high-volume real-log patterns observed on
  May 5 2026: device timeouts, tolerance failures, orphan files, empty
  DataFrame saves, and the `attempted to unregister from unknown event` noise.
  Previously everything classified as `unknown`; hardware issues now surface
  as `hardware_issue` and config issues as `config_issue`.

## [0.1.0] - 2026-05-07

### Added
- Initial package scaffolding under `GEECS-Plugins` monorepo.
- `schemas.py` — Pydantic v2 models: `LogEntry`, `ErrorFingerprint`,
  `ErrorOccurrence`, `TriageReport`, `Severity`, `Classification`.
- `parser.py` — regex-based parser for the `geecs_scanner.logging_setup`
  scan-log format with multi-line traceback aggregation.
- `fingerprint.py` — message normalization (strip numerics, paths, IDs) +
  sha1-based 12-char fingerprint hash.
- `classifier.py` — maps known GEECS exception types and Python builtins to
  `bug_candidate`, `config_issue`, `hardware_issue`, `operator_error`,
  or `unknown`.
- `harvester.py` — walks scans for a date / date range using
  `geecs_data_utils.ScanPaths`, aggregates by fingerprint into a
  `TriageReport`.
- `cli.py` — `geecs-log-triage` console script with `--date`,
  `--date-range`, `--experiment`, `--level`, `--scan-folder`, `--output`
  arguments.
- Unit tests with synthetic scan-log fixtures.
