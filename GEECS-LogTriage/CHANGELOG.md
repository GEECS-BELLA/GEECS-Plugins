# Changelog

All notable changes to `geecs-log-triage` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
