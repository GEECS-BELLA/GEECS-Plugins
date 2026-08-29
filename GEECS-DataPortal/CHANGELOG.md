# Changelog — geecs-data-portal

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-08-29

### Added

- Deployment (portal arc phase 5): `deploy/geecs-data-portal.service`
  systemd unit (generic-account template, site specifics live in the
  `/etc/systemd/system` copy — CA-gateway precedent) and
  `DEPLOYMENT.md` runbook (prerequisites, own-checkout install,
  foreground smoke test, unit install, upgrade, troubleshooting).
  Fleet map promoted from *planned additions* to a deployed service
  row (worker host, HTTP 8200, `GET /health`).

## [0.2.0] - 2026-08-29

### Added

- Resource viewer (portal arc phase 4): per-run image gallery over the
  scope doc's tiering — capture-daemon HDF5 stacks via
  `geecs_data_utils.io.scan_stack` (Tier A), native per-shot files via
  `ScanPaths.build_asset_path`/`infer_device_ext` + `read_imaq_image`
  (Tier B), vendor-SDK formats shown as a path card (Tier C, `.himg`).
  Lazy per-shot loading (prev/next + shot input), percentile-windowed
  16-bit → 8-bit display rendering, device names validated against the
  scan folder (traversal guard), strictly read-only (tree-untouched
  pinned for hits and misses).  Routes: gallery in `/run/{uid}`
  (`?device=&shot=`), `/run/{uid}/image.png`.  Bluesky-native
  timestamp-named files (`<device>_<labview_seconds>.<ext>` — what
  production scans write today; live-verified on Scan 012 2026-08-21)
  join by the event row's `acq_timestamp` through the package's
  canonical machinery (`native_files` millisecond keys for files;
  `read_stack_timestamps` + the same keys for stacks — ScanAnalysis
  parity, robust to pre-scan extra frames), with ordinal order only as
  the no-metadata fallback; a shot with no exact match — including a
  device that missed the shot (NaN row) — 404s rather than serving a
  neighbouring shot's image.  Scan-folder re-basing uses the run's OWN
  day from its start time, never the caller's `day` param (a bookmarked
  link must not resolve today's same-numbered scan).

## [0.1.0] - 2026-08-29

### Added

- Review fixes on the scaffold PR: plottable-column semantics now come
  from the shared `tiled_schema.plottable_columns`/`numeric_series`
  (console parity — machinery excluded, dtype-tolerant telemetry
  plottable); X-axis selector in the run view with the console's
  stepped-scan default (scan variable on X); day/experiment picker and
  run-list filter forms; `experiment`/`day` URL-encoded in every href;
  plots moved to the matplotlib object API (no pyplot global state on
  the threadpool).
- Package scaffold (portal arc phase 3, per
  `Planning/data_portal/01_data_portal_scope.md`): FastAPI app over the
  `ScanCatalog` seam with server-rendered day view (run list), run view
  (metadata rows via the shared `metadata_rows`, numeric-column picker),
  server-side matplotlib scalar plots (`/run/{uid}/plot.png`), and a
  `/health` catalog probe.  CLI `geecs-data-portal` (default port 8200)
  injects `TiledScanCatalog.from_config()`.  Hermetic TestClient suite
  over fake catalogs.
