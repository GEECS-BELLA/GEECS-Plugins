# Changelog

All notable changes to `geecs-log-triage` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
