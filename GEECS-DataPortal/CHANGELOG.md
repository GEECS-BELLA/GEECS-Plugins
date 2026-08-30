# Changelog — geecs-data-portal

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] - 2026-08-29

### Added

- **Within-scan prefetch caches** (`geecs_portal/cache.py` — owner
  feature request, amending the scope doc's blanket lazy rule to "eager
  within a scan, lazy across scans"): `CachingScanCatalog` keeps
  completed runs' details (LRU-8; still-running runs expire in seconds),
  ending the repeated full-event-table Tiled reads; `ShotDataCache`
  (bytes-bounded LRU, ~1.5 GB) keeps completed runs' pixel data — a
  stack device's whole frames array in ONE HDF5 read on first touch, a
  native device's decoded shots warmed by a background thread walking
  the event rows — so stepping through a gallery serves from memory
  with zero filesystem access (pinned by delete-the-file-then-serve
  tests).  Still-running runs and ordinal (listing-order) resolutions
  are never cached.  `__main__` wraps the real catalog.  Hardened per
  review: the budget genuinely bounds the cache (per-entry cap =
  budget/3 — a single warming entry can never exceed it; oversize
  stacks serve per shot from disk, uncached), stack admission requires
  the daemon's `finalized=True` stamp (the stop doc lands before
  finalization — caching an un-finalized stack would 404 tail shots
  from memory forever), warms are throttled (2 threads) and run once
  per key per process (no hole re-probing per page view).

## [0.4.2] - 2026-08-29

### Changed

- CLAUDE.md / app.py docstrings: the "shared with the console's B4"
  claims restored — true again with Console 0.25.0 (docs-only).

## [0.4.1] - 2026-08-29

### Changed

- `resources.py` consumes the consolidated Data-Utils 0.20.0 join/tier
  machinery instead of private copies: `device_kind` is now THE one tier
  ladder (`load_shot_image` dispatches on it — badge and endpoint can
  never disagree), extension sets come from the shared taxonomy
  (`_vendor_only` retired), the stack join is
  `read_shot_for_acq_timestamp` (one h5py open per image request, was
  three; keep-FIRST duplicate-key semantics — ScanAnalysis parity, was
  keep-last), the native probe is the shared `probe_native_file`, and
  the day view's time cells use the shared `fmt_time_of_day`.
  `run_view` passes its device listing into `device_kind` (no second
  directory scan per page). Behavior preserved, with one improvement:
  in a non-canonical (dev/scratch) scan folder the gallery badge now
  classifies vendor/unrenderable/native correctly (the tier probe no
  longer needs `ScanPaths`) instead of showing a missing card; native
  loads there still degrade to the layout card, never a 500.

## [0.4.0] - 2026-08-29

### Fixed

Review-fix wave (max-effort #712 review findings):

- A shot beyond the run's recorded event rows now 404s instead of
  falling back to ordinal indexing — the fallback could serve an orphan
  frame (pre-scan stack extras) labeled as a shot that never happened.
  The gallery's "next →" link is bounded by the event-row count and the
  shot input clamps.
- A Tiled outage now returns 503 "catalog unavailable" from the run
  routes instead of 404 "run not found" (`KeyError` alone means unknown
  uid).
- A run with no usable start time no longer resolves via today's daily
  folder (same-numbered scan hazard) — an explicit `day` param or
  nothing.
- Non-canonical recorded scan folders and malformed capture stacks
  (missing/mistyped `/acq_timestamp`) degrade to the missing card / 404
  instead of 500 (`ValueError`/`KeyError`/`TypeError` now caught).
- Sticky query state: every template link builds its query through one
  helper — the plot selection survives shot stepping and device picks,
  the day filter survives prev/next-day navigation.
- `.dat`/`.tdms` devices get an honest "unrenderable" card instead of a
  false "vendor-SDK format" label (new tier kind).
- Plot axis labels render with `parse_math=False` (a `$` in a GEECS
  column name would 500 at savefig).

### Added

- Caching headers on `plot.png`/`image.png`: completed runs are
  immutable per URL, still-running runs revalidate.
- systemd unit pins `TZ` (daily folders are named by the scanner
  host's local date; a UTC-defaulted server would resolve evening scans
  into the next day) + runbook troubleshooting row.
- Hermetic-test guard: an autouse fixture keeps the suite off the real
  `config.ini` data root (`TestRunView` previously statted the share on
  developer machines); the test fake now lists runs newest-first per
  the catalog contract, with an ordering pin.

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
