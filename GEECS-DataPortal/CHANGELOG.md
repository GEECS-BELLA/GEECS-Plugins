# Changelog — geecs-data-portal

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.10.0] - 2026-08-31

### Changed

Plot-tab figures are now **authored server-side in Python** (the
renderer ruling from the plotly.py-vs-Altair bake-off — same vendored
plotly.js renderer, spec authorship moves down):

- New `geecs_portal/figures.py`: `shots_figure` / `binned_figure` build
  the complete Plotly figure (palette, base layout, the stacked
  multi-axis ladder, asymmetric error bars, log/date guards, display
  cosmetics) with plotly.py; the package gains a `plotly` dependency
  (server-side only — the browser keeps the vendored bundle).
- `/api/run/{uid}/frame` and `/binned` accept the URL-carried
  `display` JSON (validated at the boundary: wrong types, unknown
  fields, and non-finite numbers are 400s per the `bincfg` precedent;
  cosmetic *values* keep the page's degrade semantics — a non-hex
  color or non-positive marker size falls back to the default, because
  display state rides shared links) and return a ready `figure` field.
  Responses without `cols` carry no figure. The version-keyed `/api`
  cache rolls browsers onto the new shape. The raw `series`/`shot`/
  `bins` keys stay alongside `figure` deliberately — the `/api` layer
  remains the data contract; the duplication is the accepted cost.
- `run.html`: `drawShots`/`drawBinned`/`multiYAxes`/
  `applyDisplayToLayout` and the layout constants collapse into one
  `drawFigure` — `Plotly.react` over the served figure. The
  `display.layout` passthrough deliberately **stays client-side** with
  its prototype-pollution guard (the URL-carried patch never executes
  on the server), and the trace palette is now injected from
  `figures.TRACE_COLORS` so rail chips cannot drift from the traces.
- Aesthetics rider (separate commit, cheap to revert): outside tick
  marks and a one-step-subtler gridline color — the Vega-Lite look the
  owner picked out in the renderer bake-off, ported into the Plotly
  base layout.
- "Show the code" now reproduces the **figure**, not just the numbers:
  both snippets end with the `shots_figure`/`binned_figure` call that
  yields the identical figure the page renders (from the notebook
  frame, axis titles show raw column names — the page adds pretty
  names).

## [0.9.1] - 2026-08-31

### Fixed

Fix wave from the #728 promotion review (cloud review findings):

- **XSS via URL-carried display/filters JSON** (`run.html`): the three
  attribute sinks that interpolated shared-link state unescaped are
  closed — `traceColor()` now admits only hex colors (falling back to
  the palette), and the filter modal's low/high inputs render only
  actual numbers. Shared analysis links can no longer inject markup.
- **Union shot axis**: `/api/run/{uid}/frame` coalesces NA
  `scan_event_index` cells with the s-file's own `Shotnumber` (plain or
  collision-suffixed), so s-file-only union rows keep a shot axis
  instead of a null that Plotly silently dropped from the default plot.
  A run-only frame with a genuinely unknown shot still serializes null.
- **`bin_width <= 0` is a 400** at the `parse_bincfg` boundary (zero
  divided to `inf` inside `compute_bin_key` and escaped as a 500).

## [0.9.0] - 2026-08-31

### Added

Reverse-proxy mountability (OSPREY panel-tab feedback): the portal now
works at root **and** under any URL prefix — `proxy /portal → :8200`.

- Every template href/form/img/script URL and the page JS's `/api`
  fetch base (`const ROOT`) build through one per-request `root`
  prefix; the `/`, `/go`, and `/run/jump` redirects carry it too.
- The prefix auto-derives from the proxy's `X-Forwarded-Prefix` header
  (validated against a strict path-segment pattern — malformed values
  are ignored, a bare `/` means root). The middleware also re-prefixes
  the request path to the ASGI-canonical shape, so mounts named like a
  route head (`/run`, `/api`, …) route correctly and trailing-slash
  redirects keep the prefix. A static fallback is available as
  `geecs-data-portal --root-path /portal` (without those two
  guarantees — see DEPLOYMENT.md); the header, when present, wins.
- `/health` (present since 0.1.0) is the panel health probe — wire
  OSPREY's `web.panels.dataview.health_endpoint` at it.

## [0.8.1] - 2026-08-31

### Fixed

Multi-Y axis rendering (owner feedback: overlapping tick numbers,
grey always-there labels):

- Axes 3–4 stack outward via Plotly's native `autoshift` (the previous
  hand-set `position` put them on top of axes 1–2's ticks).
- Real color-matched axis titles on the two anchored axes; axes 3–4
  rely on colored ticks + the legend (Plotly does not shift a
  free-anchored axis's title with the axis — measured).
- The grey "Click to enter …" placeholders are gone: in-place editing
  is now granular (legend names, annotations, shapes) instead of
  blanket `editable: true` — axis titles are auto-set, or settable via
  the advanced layout box.
- Single-trace plots hide the redundant legend (the colored axis title
  names the trace); `automargin` on all axes.

## [0.8.0] - 2026-08-30

### Added

The plot-controls suite (owner architecture feedback: stop hand-rolling
one knob per request):

- **Everything Plotly gives for free, switched on**: scroll-wheel zoom,
  spike lines, hover-compare modes, and the built-in drawing tools
  (line / freehand / rect / circle / eraser) in the modebar — direct
  on-plot annotation, zero portal code to maintain.
- **The layout passthrough**: the display popup gains an "advanced" box
  taking any Plotly layout JSON, deep-merged onto the figure last (and
  URL-carried in `display.layout`). The entire Plotly layout schema —
  tick formats, fonts, legend placement, secondary-axis styling — is
  now reachable without new portal code; the curated fields remain the
  common-case UI. Malformed JSON keeps the popup open with the error,
  applying nothing.

## [0.7.0] - 2026-08-30

### Added

Analysis-tabs W1e — Plot-tab polish (owner feedback on the live tab):

- **Display settings popup** ("display…" next to show-the-code): log
  X/Y, explicit numeric axis ranges, marker size, per-trace colors —
  URL-carried like all view state; plus Plotly `editable: true`, so
  axis titles and legend names are click-to-edit in place.
- **Picker cleanup**: pretty names lead (`telemetry_` stripped via the
  shared `display_name`, raw name on hover), and the `ts_`
  event-recording timestamp columns hide behind an off-by-default
  "timestamps" toggle (`/api/.../columns` now carries a `timestamp`
  flag from the new schema helper).
- **Timestamps plot as datetimes**: a plotted timestamp column arrives
  as host-local ISO datetimes on a Plotly date axis — `ts_*` converted
  from Unix event time, `acq_timestamp` spellings from the LabVIEW
  wire epoch (`frame` responses carry a `kinds` map).
- **In-tab day/scan navigation**: the rail gains a scan dropdown (the
  day's runs — it navigates from the live URL, so unsaved-state loss is
  impossible), and the day steppers now go through `/run/jump/{day}` —
  same scan number on the target day (else its newest run) with the
  whole analysis state carried; only an empty day falls back to the
  day page.
- **Version-keyed `/api` caching**: completed-run responses cache
  immutable, so every `/api` fetch carries `v=<portal version>` —
  browser caches roll over exactly at upgrades (the payload shape
  changes with releases).

### Notes

- The datetime rendering is presentation-side: the `code` snippet
  mirrors the conversion (same `LABVIEW_EPOCH_OFFSET` shift), and the
  **binned** view deliberately serves raw numbers — a timestamp bin
  column keeps epoch-second labels.

## [0.6.0] - 2026-08-30

### Added

Analysis-tabs wave W1d — the Plot tab (the arc's first interactive
analysis surface; mockup rulings 2026-08-30):

- **Vendored Plotly** (the approved doctrine amendment, now written
  into CLAUDE.md): `geecs_portal/static/plotly-cartesian-3.1.1.min.js`
  (MIT, 1.4 MB, the cartesian partial bundle — scatter + heatmap cover
  the whole arc), served at `/static/` — still no npm, no CDN.
- **The `/api` JSON layer** — one-liners over the W1a–c data-utils
  primitives, each response carrying a `code` field (the notebook
  snippet that reproduces it exactly): `columns` (union pick list with
  `run`/`sfile` provenance + the stepped-scan default X), `frame`
  (per-shot series, filters applied), `binned` (centers + asymmetric
  error bands via `bin_frame`), `filter-count` (live pass count).
  Param parsing/JSON chores live in `geecs_portal/analysis.py`
  (`BadParam` → 400; NaN → `null`; ≤ 4 y columns).
- **The scan page rework** (`run.html`): rail (scan/day steppers that
  keep the whole analysis state, provider chips, named filter chips
  with enable/remove + live pass count) + Overview / Plot / Images
  tabs.  The Plot tab: type-to-filter column picker over the union
  frame, up to 4 Y columns on per-series axes, per-shot ⇄ binned
  toggle, bin-settings ⚙ popup, the OR-of-AND filters popup (with
  would-pass preview), "show the code".  All view state is
  URL-carried — a link IS the analysis (and the multi-user story).
- CLAUDE.md gains the three-layer contract and the "adding an analysis
  tab" checklist (the W2-must-be-dramatically-cheaper checkpoint).

### Changed

- The run page's server-rendered quick plotter is gone per the mockup
  ruling (Overview = metadata only); `/run/{uid}/plot.png` itself
  remains for embedding.  The `y` query param is now repeatable.

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
