# Changelog

All notable changes to `geecs-scan-log` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to semantic versioning.

## [0.1.0] - 2026-09-11

### Added

- Initial package: a read-only day-document view over scan folders.
- `geecs_scan_log.models` — `ScanSummary` and `DaySummary`, the derived
  view of a scan folder. Nothing here is stored by the logbook.
- `geecs_scan_log.scan_reader.read_day` — lists a day's `ScanNNN` folders
  and parses each `ScanInfoScanNNN.ini` into a `ScanSummary`. Read-only by
  construction: it never constructs `ScanPaths(read_mode=False)` and never
  calls `mkdir`.
- Scan status derived from `ScanEndInfo`: `success`, `failed` (with the
  failure reason surfaced), `incomplete` (folder exists, no ScanInfo), or
  `unknown`.
- `Campaign` — consecutive scans sharing a parameter and purpose group into
  one run. Derived from what the scanner already wrote, so nobody declares a
  campaign and nobody can forget to. A day above 20 scans renders campaigns
  (one rail row each, collapsed) instead of a flat list; a failure inside a
  collapsed campaign still surfaces on its header.
- `geecs_scan_log.router.create_log_router` — an `APIRouter` the Data
  Portal mounts at `/log`, serving `/log/day/{date}` and a JSON peer at
  `/log/api/day/{date}`.
