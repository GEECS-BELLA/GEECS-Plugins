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
- Fewer round trips per scan: one `os.scandir` of a scan folder yields both
  the ScanInfo path and the device list, replacing a `glob` plus an
  `iterdir`, and it carries the ScanInfo file's own stat for the cache key. Four round trips became two — measured
  1.85x faster on cold days (403 -> 218 ms per scan, A/B across untouched
  August dates).
- Concurrent folder reads (16 workers) and a cache of the per-scan file
  reads, keyed on the **ScanInfo file's** own mtime and size. A 108-scan day
  over VPN went from 27.3 s to 5.6 s on a cold share and milliseconds once
  cached. The key is the file's, never the folder's: the scanner finalises a
  scan by rewriting ScanInfo in place, which changes no directory entry, so
  a folder-keyed cache served a running scan's empty `ScanEndInfo` forever —
  losing exactly the failure reason this view exists to surface.
- Parsing borrowed rather than reimplemented: `ScanInfo` through
  `geecs_data_utils.scan_paths.read_scan_info_file` (shared with
  `ScanPaths.load_scan_info`) and the scan's start time through
  `scan_log_loader.first_log_timestamp`. One surface to fix per format.
- `ScanEndInfo = ""` classifies as `incomplete`, not `unknown`. The scanner
  writes it empty at claim time and fills it at the stop document, so empty
  means *not finalised*; it is the most common state on the real share (37
  of 49 ScanInfo files across four sampled days) and reporting it as
  "unrecognised" painted most of a day amber.
- Scan start time comes from `scan.log`, not the folder's modification time.
  Any later pass that writes into a scan folder moves that timestamp — it
  was measured over an hour off the real start.
- `/log/static/{name}` is a plain route, not `router.mount(StaticFiles(...))`.
  A `Mount` is a `BaseRoute`, not a `Route`, and `APIRouter.include_router`
  drops it silently on FastAPI versions this package's floor allows, 404ing
  the stylesheet and 500ing the page while CI stays green on a newer pin.
- Unpadded scan folders (`Scan42`) are no longer invisible, matching
  `geecs_log_triage.harvester`.
- A malformed numeric field (`inf`, `nan`) reads as absent instead of
  escaping `int()` and 503-ing the whole day.
- Expand/collapse reaches scans inside campaigns, not just the campaigns.
- Day navigation: a date picker, previous/next-day steps, a "back to today"
  link, and quick links centred on the shown date so stepping forward is as
  easy as stepping back.
- `geecs_scan_log.router.create_log_router` — an `APIRouter` the Data
  Portal mounts at `/log`, serving `/log/day/{date}` and a JSON peer at
  `/log/api/day/{date}`.
