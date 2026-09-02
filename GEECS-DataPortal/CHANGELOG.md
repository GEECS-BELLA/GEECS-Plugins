# Changelog — geecs-data-portal

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.17.0] - 2026-09-01

### Added

- **Analysis tab** on the scan page (PR 3 of the 04 design): every
  loadable diagnostic with its data device, applicability, state badge
  (`queued`/`running`/`done`/`failed`/`no_data`), a run / re-run button
  (disabled while any run is active for the scan), the error text and a
  collapsible captured log on failure, and the produced artifacts inline
  (raster images) or as download links — served through
  `/run/{uid}/artifact`. Inapplicable analyzers are collapsed under
  "other analyzers". The page polls the list endpoint every 1.5 s while
  a run is active and stops when it settles. The tab (and the
  `tab=analysis` URL state) appears only when runs are possible: the
  feature is configured, the `analysis` extra is installed and the scan
  folder resolves; otherwise a bookmarked or stepper-carried
  `tab=analysis` falls back to the Plot tab (`setTab`: no pane → plot).
  The list endpoint gains `artifacts` — each entry `{path, servable,
  inline}` decided server-side (`analysis_runs.describe_artifact`,
  the one inline-raster policy) so the page never re-derives it from
  a path's shape; ids reach the run / log handlers through `data-`
  attributes, never interpolated into attribute JS (review #765).

## [0.16.0] - 2026-09-01

### Added

- **Analysis runs** — the portal can now run one ScanAnalysis analyzer
  on one scan from the browser (`Planning/data_portal/04_analysis_run_design.md`,
  owner ruling 2026-09-01; the charter amendment "read-only except
  explicit analysis runs"). `geecs_portal/analysis_runs.py`:
  `AnalysisRunner` (one worker thread, one active job per scan,
  in-memory records with `queued`/`running`/`done`/`failed`/`no_data`,
  artifacts, error text and the run's log lines captured from the
  worker thread only), the analyzer factory seam (default =
  `load_diagnostic` + `create_scan_analyzer`, the same two calls the
  group loader runs in a loop; tests inject a fake), and the artifact
  containment helper. Endpoints: `GET /api/run/{uid}/analysis`
  (loadable diagnostics with applicability by data device, job record,
  files on disk under the analyzer's output dir), `POST
  /api/run/{uid}/analysis?analyzer=` (202 / 404 ladder / 409 while a
  job is active), `GET /run/{uid}/artifact?path=` (serves a produced
  file, resolved path must stay inside the scan's analysis folder).
  Deliberately NOT a task-queue participant — `run_analysis` is called
  directly; no status records, claims, heartbeats or Google Doc
  uploads. `cleanup()` runs on every outcome. Review-hardened (#763):
  the scan tag is parsed from the resolved folder (`ScanPaths(folder=…)`),
  never rebuilt from the start doc's time; the artifact endpoint is
  gated with the feature, serves raster images inline and everything
  else as `attachment` + `nosniff`; the job's final state is assigned
  after its log/finished fields; a `BaseException` from an analyzer is
  recorded (never a record stuck at `running`); the lifespan refuses
  new runs at shutdown and logs an in-flight one.
- The `analysis` extra now carries ScanAnalysis alongside
  ImageAnalysis; `__main__` pins `matplotlib.use("Agg")` before any
  analysis import (ScanAnalysis renderers use pyplot; the single
  worker thread serialises them).

### Changed

- `DEPLOYMENT.md`: the share must be mounted read-write where analysis
  runs are enabled; the extra and the flag are named in the
  prerequisites. Scope doc ruling 2 and the root dependency graph
  amended to match.

## [0.15.3] - 2026-09-01

### Fixed

- Responses computed over the union frame — `/api/run/{uid}/columns`,
  `frame`, `binned`, `bin-images` and `bin-image.png` — are no longer
  served `immutable` for completed runs. The event table is frozen once
  the run stops, but the s-file half of the union is not: ScanAnalysis
  appends its columns to `analysis/sN.txt` after the scan (hours later
  when re-run by hand), so a browser that had fetched the column list
  before analysis stayed pinned to the pre-analysis shape for a year
  (scan 32 showing 7 columns while scan 33 showed 30, 2026-09-01).
  These responses (and `filter-count`, which had no header at all)
  are now `no-cache` — a plain re-fetch, no validator is emitted; the
  per-shot `image.png` and `plot.png`, which read the event table
  alone, keep their immutable headers.

## [0.15.2] - 2026-09-01

### Fixed

- `poetry.lock` refreshed for ImageAnalysis 1.13.1 (#739): the
  `analysis` extra no longer drags pytest, pluggy and iniconfig into
  the **main** group (the leak the #737 review surfaced). The
  Poetry-deployed portal host is unaffected in practice (it installs
  the dev group, which carries pytest); the change is to the
  main-group / `pip install` closure. Lock-file refresh only — no code
  change.

## [0.15.1] - 2026-09-01

### Fixed

- DEPLOYMENT.md now names the `analysis` extra and the
  `--processing-configs` `ExecStart` flag (quoted — share paths carry
  spaces) that the processing selector needs — the fleet map's
  "each runbook names its extras" claim was false for this runbook
  (docs-only; #745 review finding).

## [0.15.0] - 2026-09-01

### Added

- **Image colormaps + display windowing** (owner ask, "step 1" of the
  interactive-images plan): the image endpoints (`image.png`,
  `bin-image.png` — raw and processed alike) take the shared
  `display` state's new curated fields `cmap` (matplotlib colormap
  name) and `plo`/`phi` (percentile-window overrides, defaults
  1/99.7). Types 400 at the parse boundary, values degrade (unknown
  colormap → grayscale, insane window → defaults) — display state
  rides shared links and must never fail one. A small "display…"
  popup in the Images plotbar edits them; URLs carry them everywhere
  (per-shot, grid, steppers). RGB inputs keep their own colors.
  Step 2 (interactive per-shot Heatmap view with hover pixel values)
  is planned post-promotion.

## [0.14.0] - 2026-09-01

### Fixed

Sam's first live test of the selector (Amp4, 2026-09-01) — the
diagnostic YAML was a legacy flat camera config, and the failure was
invisible:

- **The selector now offers only LOADABLE diagnostics**: each
  discovered stem is validated with a real `load_diagnostic` (cached
  against the tree's YAML mtimes), and an unloadable legacy config is
  an INFO log line naming the file — never a pickable entry that can
  only produce a broken image. A hand-edited URL still gets the
  honest 400.
- **Image failures surface their reason**: the per-shot image and
  every bin card carry an `onerror` hook that fetches the endpoint's
  4xx `detail` and shows it in place of the browser's broken-image
  icon (cleared on the next processing change).

## [0.13.1] - 2026-09-01

### Fixed

- The processing-selector test class now `importorskip`s
  `image_analysis`, and CI installs the portal with
  `--extras analysis` so those tests always **run** there (they
  failed red in the 0.13.0 CI, whose env lacked the extra; a minimal
  local env now skips them gracefully instead). Test/CI-only — no
  runtime change.

## [0.13.0] - 2026-09-01

### Added

- **Images tab ephemeral-processing selector** (W2a): a `processing`
  URL state naming an ImageAnalysis diagnostic to run write-free on
  the served pixels via `run_diagnostic_ephemeral` (ImageAnalysis
  1.13.0's seam — the structural no-writes contract lives there).
  Per-shot view renders the diagnostic's `processed_image`; per-bin
  view processes each member shot THEN averages (the correct order
  for nonlinear pipeline steps). Explicit-opt-in by doctrine (design
  doc finding 7 — two competing config-resolution paths exist, so the
  portal names its tree): the `--processing-configs <tree>` CLI flag /
  `create_app(processing_config_dir=…)` enables it, and the portal
  never falls back to the global config resolution. ImageAnalysis is
  a new OPTIONAL dependency (the `analysis` extra); without it — or
  without the flag — the selector hides and raw serving is untouched.
  Error ladder: unknown diagnostic 404, denylisted/miswired 400,
  analyzer failure 400 honestly, never a 500.

## [0.12.0] - 2026-09-01

### Added

- **Images tab per-bin averaged grid** (W2a): a per-shot ⇄ per-bin
  toggle mirroring the Plot tab's ONE URL-carried `view` state (a
  binned link means binned everywhere — deliberate), rendering a lazy
  grid of per-bin `nanmean` averages. Two new routes:
  `/api/run/{uid}/bin-images?device=&filters=&bincfg=` (membership
  JSON — bins/counts/member shots, notebook-reproducible via its
  `code` snippet) and `/run/{uid}/bin-image.png?bin=<index>` (one
  bin's average via the shared `average_frames`, display-windowed
  once after averaging). Both run the same `compute_bin_key` +
  groupby membership call, so the `bin` index is stable between
  listing and render; per-shot refusals carry over (never average a
  neighbour in: events bound, missed-shot skip, vendor 404), and a bin
  containing any native listing-order (ordinal-fallback) resolution
  serves `no-cache` — the same per-shot rule. `min_count` applies to
  the grid exactly as `bin_frame` applies it to `/binned` (per-bin row
  counts), so the shared binset popup governs both tabs.
- `resources.load_shot_array` (+ `ShotArray`): the tier ladder now
  resolves to raw pixels, with `load_shot_image` reduced to the
  render-one-shot wrapper — single-shot serving and per-bin averaging
  share one resolution path (and one `ShotDataCache` ride).

## [0.11.0] - 2026-09-01

### Added

Owner live-feedback round on 0.10.0:

- **Binned view plots against the X pick** — bins still *group* by
  `bincfg.bin_col`, but each bin now *plots at* the per-bin **mean of
  the selected X column** (`/binned?x=…` → `x_centers` in the payload,
  the figure's x positions, and the axis title). Same primitive, same
  bins: a second `bin_frame` call with `replace(cfg, value_cols=(x,),
  agg="mean")`, mirrored verbatim in the "show the code" snippet. No
  X keeps the bin labels as the axis, exactly as before. (X error
  bars deliberately deferred — mean placement is the first move.)
  `x_centers` come **reindexed onto the y result's bins** — the x
  call's dropna runs over x alone, so its surviving bins can differ,
  and positional zipping would silently plot points at the wrong
  bin's x (review-caught); a bin missing an x center degrades to a
  skipped point. A timestamp X serves raw seconds (the binned raw
  rule, extended deliberately). Coercible-string columns
  (dtype-tolerant telemetry) now 400 in binned view instead of
  500ing inside `bin_frame` — as y too, a pre-existing hole.
- **Plot size control**: `width`/`height` join the display vocabulary
  (popup inputs; same type-400/value-degrade rules). A fixed size also
  fixes the exported image size.
- **Copy plot to clipboard**: a modebar button exports the figure at
  2× and puts the PNG on the clipboard — copy-paste is how plots
  travel around the lab. Caveat: the async Clipboard API requires a
  secure context (https or localhost); on plain-http lab hosts the
  button degrades to the 2× PNG download with a note. The built-in
  camera download is 2× now too.

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
