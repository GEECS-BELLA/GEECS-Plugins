# Changelog — geecs-data-portal

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.36.0] - 2026-09-24

### Added

- The config editor's **summary preview**: `POST /configs/api/preview/summary`
  draws the document's summaries over the scan's first shots
  (`params.shots`, default 4, at most 8 — a handful, never a scan; shots the
  device missed or beyond the run are skipped), one panel per shot at its
  shot number labelled `shot`, through ScanAnalysis' `core_preview.preview_summary`
  (`processing.render_summary_as_run`). A v2 kind the core does not run has
  no summary preview (400).

### Changed

- The editor's frame preview and the Images tab's processing view analyse
  through ScanAnalysis' preview seam (`core_preview.preview_frame` /
  `prepare_document` + `measure_frame`); the portal no longer re-derives
  the run's compile/measure/draw pairing, and its byte-for-byte test now
  pins against that seam. The shot loaders (`_camera_frame`, `_line_trace`)
  are shared by the frame and summary previews.

## [0.35.0] - 2026-09-24

### Changed

- The config editor's preview is **the run's own draw**:
  `processing.render_document_as_run` renders the shot through the analysis
  sink's per-frame call (`single` with the document's `figure` block; a v2
  renderer translated), and `resources.figure_png(..., tight=True)` crops it
  like the sink's product PNGs — the pane shows the file a run would write.
  The portal's palette/window overrides no longer reach the editor preview
  (they stay on the Images tab's processing selector). Pinned byte-for-byte
  against `single(analyze_v2(frame), figure_of(recipe))`.
- The recipe's frame inputs (a background image under `{scan_dir}`) load
  from the document's device folder under the previewed scan, exactly as
  the run loads them (before, the placeholder stayed literal and the
  fallback constant — or an error — stood in for the real frame).
- A legacy-route preview that draws no figure is a 400, not a 500.
- The drawer's "duplicate as" reports nothing when the open document is a
  read-only format 2 diagnostic (the copy was never made); its tooltip
  names what the copy drops.
- The drawer's "duplicate for this device" patches the recipe's `device`
  (and the editor drops `output_name` / `input.folder` from the copy).

## [0.34.0] - 2026-09-24

### Changed

- Processing, the Images-tab selector and the editor previews load either
  analysis format; a v3 recipe's preview palette comes from its
  `figure.imshow`, a v2 diagnostic's from `scan.renderer` as before.

## [0.33.0] - 2026-09-23

### Added

- The config editor's live preview renders LINE diagnostics (`image.type: line`)
  on the shot's trace. It used to load images only, so a trace document could
  not be previewed. The shot resolves through the run path's own source rules
  (`scan_analysis.core_source.prepare_source`: `scan.file_tail`, `data_format`,
  the stack-only rule for `pva_stack`, the timestamp join by the diagnostic's
  device), with the picked device standing in for `scan.device`, and is read
  with the document's `data_loading`. Auxiliary columns reach the legacy
  fallback renderer (`processing.render_document_ephemeral` grew
  `auxiliary_data`); the core route reads the primary trace alone, as its
  scan run does. The shot's own event row feeds the mapper, so a row the run
  skips (`<device>-valid` false) is refused, not previewed.

## [0.32.2] - 2026-09-23

### Changed

- The real-factory test follows ScanAnalysis 1.33.0: a supported beam recipe
  from the configs tree now builds a `CoreScanAnalyzer`, and the `scan.device`
  override is read from its document. No runtime change.

## [0.32.1] - 2026-09-23

### Changed

- Legacy-fallback tests now use an unported flip; camera rotation and crosshair
  masks are supported directly by the shared core.

## [0.32.0] - 2026-09-23

### Changed

- Camera file-background processing and unsaved previews now use the new core
  through shared source preparation, preserving load-failure fallback and
  explicit geometry errors. Backgrounds are snapshotted once per request.

## [0.31.0] - 2026-09-23

### Changed

- Supported v2 recipes in the Images processing selector and unsaved config
  editor preview now use geecs-analysis directly, including object-API figures
  and coordinate-aware beam overlays. Unported recipes retain the legacy
  write-free backend; execution errors never trigger a silent fallback.
- Preserve process-each-shot-before-averaging for bin images, display windows,
  preview renderer options, error responses and config-editor saves. Explicit
  scan execution remains on the existing factory pending runner migration.

## [0.30.1] - 2026-09-23

### Changed

- `resources`: the illustrative lineout shape in the module docstring is
  `(16384, 2)` (GEECS-Core 0.11.2, #988 review). Docs only.

## [0.30.0] - 2026-09-21

### Added

- **The Images tab draws a line for a device whose shots are traces.** The
  PVA gateway's file plugin now captures array variables — scope traces and
  spectra — through the same stack layout as a camera, and the tab
  classified any folder holding a stack as an image stack: a `(2048, 2)`
  lineout rendered as a two-pixel-wide strip and a 1-D waveform would not
  render at all. `resources.stack_content` asks the stack which it is (the
  plugin declares it — rank alone cannot tell `(N, H, W)` pixels from
  `(N, n, 2)` rows) and the tab serves a line instead.
  - `GET /api/run/{uid}/trace` — one shot as a server-authored figure
    (`figures.trace_figure`, the same figures.py + theme-sentinel path as
    every other plot here), with the image endpoint's refusals: a shot
    beyond the recorded events and a device that missed the shot both 404.
    A camera shot stays a rendered PNG — a 2048² frame as JSON is absurd —
    while a few thousand trace samples travel as a figure and keep their
    hover readout.
  - `resources.load_shot_trace` reads it, through the same
    canonical-millisecond shot→frame join the image path uses and the
    reader that trims the gateway's padding
    (`Data1DType.PVA_STACK`, GEECS-Data-Utils 0.36.0) — the portal never
    sees a pad ceiling or a `wave_dx`.
  - The trace is drawn only while the Images pane is visible — by the boot
    path and by the theme handler alike — and re-themed without wiping a
    live graph's DOM. Plotly sizes a figure against its container and the
    vendored build carries no `ResizeObserver`, so a figure laid out into a
    `display:none` pane stays zero-sized (a shared link whose tab is `plot`
    opens exactly that way, and re-theming from another tab is the same
    door). Returning to the tab redraws the cached figure — never refetches
    — so a palette change made elsewhere still lands. Pinned by node tests
    over the template's own source.
  - A non-finite sample becomes a gap in the line rather than a 500.
    `page_figure` — the one gate every served figure passes through — now
    maps a non-finite float to `null`. Only an **all**-NaN lineout row is
    padding, so a row whose value column alone is NaN reaches the figure
    intact, and `json.dumps` emits invalid JSON for it (Starlette's
    `JSONResponse` refuses it outright).
  - The send-to-logbook caption names the trace. `plotCaption` fell
    through to `S.y` / `S.x` — the Plot tab's scalar state — for any host
    it did not recognise, so a trace posted into the logbook carried an
    unrelated caption.
  - A trace device's tab has no per-bin view, no ephemeral image diagnostic
    and no image cosmetics: those are pixel controls, and averaging traces
    across shots is the consumer's job because each shot carries its own
    axis. The shared per-shot/binned state is untouched — the Plot tab
    stays binned.

## [0.29.3] - 2026-09-17

### Changed

- Docs only: the Images tab's Tier A stack is credited to the PVA
  gateway's file plugin rather than the deleted capture daemon
  (`resources.py` module docstring), and the deployment section no longer
  says the portal runs "next to the capture daemon".

## [0.29.2] - 2026-09-16

### Changed

- Strip the `Planning/` provenance citations from docstrings and comments: a
  docstring now states the rule itself, and the derivation stays in git
  history. Part of the `Planning/` prune (#931); no behaviour change.

## [0.29.1] - 2026-09-16

### Fixed

- Opening Grid no longer silently narrows the Images gallery. Only explicit
  cell-to-Images navigation applies a bin selection, which survives reloads
  separately from the selected map cell and resets when bin definitions change.
- Empty or invalid numeric Grid edits retain the last valid maps and URL state.

- Plot, Grid and Images share shot identity, including sparse legacy s-files
  whose recorded shot numbers differ from row positions and suffixed column
  names after a native/s-file collision. Notebook figures use the same rule.

- Grid toolbar's send-to-scan-log button now exports the clicked average or
  error map, rather than the Plot tab's figure. Reuses the existing dialog,
  remembered log entry and view link, with map-specific captions and square
  plotting areas in the exported PNG.

## [0.29.0] - 2026-09-16

### Added

- Grid tab with paired scalar/error heatmaps and independent average/error
  selectors. Shared shot filters retain the complete geometry and distinguish
  filtered, unacquired, missing-value and insufficient-sample cells.
- Linear, logarithmic and equal-cell axis spacing in square plotting areas;
  measured-point rendering for spiral and other nonrectangular trajectories.
- Cell inspection, repeated-visit selection, filtered bin-image navigation,
  URL-carried state and a reproducible notebook snippet through `/api/run/{uid}/grid`.

## [0.28.1] - 2026-09-15

### Changed

- Merge of `master` (c4a04bb5) into `feature/native-bluesky-plans`: the
  portal code is master's 0.28.0 (the Plot tab's send-to-logbook button
  and its write verb, #917) plus the branch's 0.27.2–0.27.3 content (the
  shared-helper identity pin, the post-GEECS-Console doc wording). No
  code conflict between the two — only this file, the version, and the
  fleet map's portal row, which master rewrote while the branch added
  the capture-daemon row above it (both kept).

## [0.28.0] - 2026-09-15

### Added

- **Send a plot to the scan's log entry** — a Plot-tab modebar button
  that puts the rendered figure into this scan's entry in the logbook,
  and the portal's third write verb (`POST /api/run/{uid}/logbook`,
  `geecs_portal/logbook_send.py`; owner ruling 2026-09-15).

  It exists because the clipboard cannot work on the deployed portal:
  browsers expose clipboard *image* writes on secure contexts only, and
  the service is plain HTTP. This path does not care about the page's
  origin — the browser hands the PNG to the portal and the portal talks
  to the logbook **server-to-server**, which also means the logbook needs
  no CORS headers and the portal still never imports it.

  One send is four calls: create the scan's entry (the logbook stores an
  attachment only against an entry), upload the PNG, re-read the version
  the upload moved, and patch the body with the image paragraph. Two
  shapes are load-bearing rather than incidental:

  - Images are **appended to one entry**, not one entry each, because
    consecutive image paragraphs are what the logbook's renderer turns
    into a figure grid — the layout LogMaker's `gdoc_slot` numbering
    used to fake. Which entry is remembered **per browser**, not asked
    of the logbook, so two people plotting the same scan from two
    machines get two entries — accepted, because finding "this scan's
    entry" would append a plot into whatever note an operator happened
    to be writing. The page offers "a new entry" when appending is not
    what you want, and an entry deleted in the meantime starts a fresh
    one rather than failing.
  - The **portal link sits above the images**, not between them: a
    paragraph in the middle of the run would split the grid. Because the
    Plot tab's whole state is in its URL, that link restores the exact
    analysis, not just the scan.

  A conflict (someone saving the entry mid-send) is re-read and retried
  once — safe precisely because this is an append, never a replace.
  Failures land as themselves: a malformed image is 400, an unreachable
  logbook 503, the logbook's own verdicts on our payload (409/413/415)
  pass through, and anything else is 502 — including a success that is
  not JSON (a wrong port, a proxy maintenance page) and a reply missing
  a key it used to carry (version skew between two services on separate
  release cadences), neither of which may surface as "bad image" or as a
  portal traceback. A blank author, and an entry id that is not the
  logbook's `[0-9a-f]` id shape, are refused here as our own malformed
  request rather than forwarded to be refused there.

  Sending resolves the run's **day only** — never a scan folder — so a
  logbook write does not stat the SMB share. The page passes `?day=`
  like every other `/api/run` call: a run whose start document has no
  usable time resolves its day from that param alone, and dropping it
  would show the button and 404 every send.

  The append reads the entry's body before rewriting it, and that read
  demands `body_md` rather than defaulting it. A default would turn "the
  body did not come back" into "the body is empty", and the PATCH that
  follows would make it so — erasing whatever had been written. Missing
  or non-text is 502 and no write at all.

  Sending requires an **absolute** `--logbook-url`; a path-shaped base
  describes the browser's front door and names no host this process can
  dial, so it links but does not send and the button is hidden
  (`logbook_send` in the run page and `GET /api/run/{uid}`). That one
  URL serves both the portal process and operators' browsers, so
  `http://localhost:8400` passes every gate and then hands operators a
  dead link — name the host.

  Sent plots render at a **fixed 720×460** unless an explicit display
  width/height says otherwise. The note scales the bitmap into its
  column, so apparent text size is export width ÷ column width: a wide
  export in a three-up grid has unreadable axis labels, and a narrow one
  reads well. `httpx` moves from a dev dependency to a runtime one — the
  one call this process makes out of itself.

### Fixed

- **Copy plot to clipboard** stops looking broken. On the plain-HTTP
  deployment `navigator.clipboard` does not exist, so the button's
  documented download fallback fired every time with a note that said
  only "copy failed". The fallback stays — nothing page-side can put a
  bitmap on an http origin's clipboard, since `execCommand("copy")`
  carries text alone — but it is now legible: the modebar tooltip reads
  "Download plot PNG — clipboard copy needs an https or localhost page"
  when the API is absent, and a real failure names the error
  (`NotAllowedError` is almost always an unfocused window). The send
  button above is the answer for the deployed path.
- The clipboard write no longer loses the click's **user activation**:
  the PNG is rendered *inside* the `ClipboardItem` as a promise value
  rather than awaited before the write. Safari expires the gesture
  across an await and rejects the write that follows, so copy failed
  there even on a secure page. Chrome accepts either shape.

## [0.27.3] - 2026-09-14

### Added

- `tests/test_shared_helpers.py` pins by identity that the portal's
  `resolve_scan_folder` and `metadata_rows` are GEECS-Data-Utils' shared
  implementations. The pin lived in the Qt console's suite until that
  package was deleted (web scanner arc PR 6).

### Changed

- `CLAUDE.md` / `DEPLOYMENT.md` no longer describe the portal relative to
  the deleted GEECS-Console.

## [0.27.2] - 2026-09-13

### Changed

- Merge of `master` (b9677ca7) into `feature/native-bluesky-plans`: the
  portal code is master's 0.27.1 (the logbook split, #877; the
  `--root-path` doc correction, #878) plus the branch's file-plugin
  attribute rename (#829); `poetry.lock` relocked against the merged path
  dependencies (geecs-data-utils 0.32.0, geecs-schemas 0.24.0).

## [0.27.1] - 2026-09-13

### Changed

- `DEPLOYMENT.md` § Behind a reverse proxy no longer calls `--root-path`
  the mode "for a proxy that cannot send" `X-Forwarded-Prefix`: it is the
  prefix-preserving mode, and pairing it with a stripping proxy loses
  every stylesheet and script (the mounts answer prefixed paths only).
  Docs only; the behaviour is unchanged and the rule now lives in
  `geecs_web_theme.web`.

## [0.27.0] - 2026-09-13

### Removed

- **The logbook mount.** `--scan-log`, `--notes-db`, the `log` extra and
  the `/log` router are gone: the logbook is its own service (GeecsLogbook
  0.10.0, unit `geecs-logbook`, port 8400, its own `StateDirectory`). The
  portal's unit drops `StateDirectory=geecs-data-portal`; the entries a
  host kept there move once, per `GeecsLogbook/deploy/DEPLOYMENT.md`. A
  `site.env` still passing `--scan-log` fails at start — deliberately.

### Changed

- The run page's link to a scan's logbook card is built from
  **`--logbook-url`**, the logbook's base: an absolute URL verbatim, a
  path (`/log`, the front door's route) same-origin with the portal's own
  prefix; empty shows no link. `GET /api/run/{uid}`'s `logbook` field is
  the same URL.

## [0.26.1] - 2026-09-13

### Changed

- Merge of `master` (d6f74211) into `feature/native-bluesky-plans`: the
  two lines below were released in parallel and are listed in version
  order; a block marked *(master line, parallel release)* reuses a version
  number the branch also used for a different release.
- Portal code is master's 0.26.0; the branch's contribution is the
  file-plugin attribute rename (#829) and a relock of `poetry.lock`
  against the merged path dependencies.

## [0.26.0] - 2026-09-12

### Added

- The run page links to its scan's card in the scan logbook
  (`/log/day/YYYY-MM-DD#ScanNNN`, in the rail beside the day steppers)
  when the logbook is mounted, and `GET /api/run/{uid}` carries the same
  URL as `logbook` (null when there is nothing to link). Built from the
  mount prefix and the resolved day folder; the portal still never
  imports the logbook. A run from another experiment than the logbook's
  gets no link — scan numbers restart per experiment (review of #844).

## [0.25.0] - 2026-09-11

### Added

- The logbook's seed templates (its type buttons) are read from
  `logbook_templates/` beside the `--processing-configs` tree — the
  configs checkout the portal already has; no new flag. Without that
  tree the composers are plain.
- Unit template: `MemoryHigh=` / `MemoryMax=` on the portal service,
  rendered from two new **required** site.env keys
  `GEECS_PORTAL_MEMORY_HIGH` / `GEECS_PORTAL_MEMORY_MAX` (3G / 4G in the
  example, sized per host), so a runaway portal is throttled and then
  restarted by systemd instead of being the kernel's first OOM pick on
  the shared box (#834). Existing hosts add the two keys and re-render.

### Changed

- `/log/month/{YYYY-MM}` (the ops book) is now served under `--scan-log`
  alongside the day pages (GeecsLogbook 0.5.0).

## [0.24.1] - 2026-09-11

### Changed

- Docs and charter follow the logbook's foundation: uploads live in
  `attachments/` beside the notes database (the state directory is the
  one thing to back up), and the mirror writes `{experiment}/logbook/` on
  the share rather than a folder inside each day. No code change.

## [0.24.0] - 2026-09-11

### Added

- `--notes-db PATH` (and `create_app(notes_db=)`): the logbook's SQLite
  file, which makes `/log` writable — entries, drafts, attachments — with
  a markdown mirror into each day's `logbook/` folder on the share. A
  charter amendment: a write verb for **commentary only**, never the
  scans tree. Defaults to `logbook.db` under systemd's `$STATE_DIRECTORY`
  when set; the unit template now declares `StateDirectory=geecs-data-portal`.
  Without either, the logbook is the read-only day view.

### Changed

- The `log` extra now installs `geecs-logbook` (the package was renamed
  from `geecs-scan-log`).

## [0.23.0] - 2026-09-11

### Changed

- The portal draws its colours from `geecs_web_theme` rather than its own
  seven inline tokens, and serves that package's stylesheet and picker at
  `/theme` for every surface mounted in this app. **The portal's appearance
  changes**: it was dark-only and now offers three palettes, each with a
  light and a dark variant, defaulting to `laser` following the system.
- `--config-editor` passes the theme URL through, so the config editor
  adopts the same palette and the choice follows a viewer between pages.
- The run page's Plotly figure paints with the live theme tokens (ground,
  font, grid, and `--trace-1..4` for the marks) and re-draws on a theme
  change; `figures.py`'s colours are the JS-off fallback. The injected
  `TRACE_COLORS` contract is unchanged.
- Theme assets are linked through `{{ root }}` with a version query, so a
  reverse-proxy prefix resolves and a theme change is not served stale.
- The Google Fonts links are gone: no CDN assets, per this package's own
  doctrine — the token font stacks carry local fallbacks.

## [0.22.0] - 2026-09-11

### Added

- `--scan-log` mounts the scan logbook (`geecs_logbook`) at `/log`, behind
  the new `log` extra. Off by default, and skipped with a warning when no
  `--experiment` is given: the logbook reads one experiment's share and
  carries no facility default.

## [0.21.5] - 2026-09-12

### Changed

- `poetry.lock` relocked against the feature head: it recorded
  `geecs-data-utils` 0.26.1, `geecs-schemas` 0.19.0, `imageanalysis` 2.0.0
  and `scananalysis` 1.19.0 while the checkout carries 0.30.0 / 0.21.0 /
  2.2.0 / 1.21.0 (the path dependencies install from the checkout either
  way; the lock now says what a host install gets).  No dependency added
  or removed.

## [0.21.4] - 2026-09-11

### Changed

- The stack cache resolves the stamp dataset through
  `scan_stack.timestamps_dataset` (either the device-prefixed or the bare
  name, GEECS-Plugins#829).

## [0.21.3] - 2026-09-11

### Changed

- The in-memory stack cache reads the areaDetector NDFileHDF5 layout
  through `geecs_data_utils.io.scan_stack`'s dataset constants and its
  lock-free `open_stack` (the stacks are written on Windows and read over
  SMB; Codex review of #823).


## [0.21.2] - 2026-09-08

### Changed

- The unit template no longer bakes `--config-editor` into `ExecStart`; it
  appends `$GEECS_PORTAL_EXTRA_ARGS` from `site.env` (the reference
  `site.env.example` sets `--config-editor`), so a read-only viewer is a
  site.env choice, not a hand-edit of the rendered unit (review of #803,
  finding 6). The drawer's "duplicate as" says what is not copied.

## [0.21.1] - 2026-09-06

### Fixed

- The config editor preview draws the analyzer's **own figure** — its
  default palette (plasma for the 2D family), or the document's
  `scan.renderer` `cmap` / `vmin` / `vmax` when set — instead of the
  Images tab's gray pixel view, so the preview matches what a run of that
  document renders.

### Changed

- The drawer's preview is on demand (`preview` button, `auto` toggle) —
  ScanAnalysis 1.20.1; the label drops "live".

## [0.21.0] - 2026-09-06

### Added

- **The analysis config editor, in the scan page** (`--config-editor`, the
  04 design's deferred item; a second **write verb** after analysis runs,
  explicit opt-in, needs `--processing-configs` + the `analysis` extra).
  ScanAnalysis' editor router is mounted at `/configs`; the Analysis tab
  gets an **edit** button per analyzer (and an "edit configs" link to the
  full editor page) that opens the editor in a drawer over the scan page
  with a **preview**: the document under edit is rendered on the
  drawer's device + shot through `image_analysis.ephemeral.
  render_document_ephemeral` (0.21.1: on demand or per edit), so an ROI or
  threshold is dialled in without saving per iteration or switching
  windows. Save writes the configs tree (the selector and the Analysis
  list pick it up by mtime as before) and refreshes the shot image if it
  shows that diagnostic. `/api/run/{uid}` and the page carry
  `config_editor`. Nothing on the scans path is touched (pinned).

## [0.20.3] - 2026-09-05

### Changed

- The processing selector reads the diagnostic's typed `scan` section
  (`diag.scan.device`) — `AnalysisDiagnostic` v2 from GEECS-Schemas 0.19.0 /
  ImageAnalysis 2.0.0 types `scan:` in-document; the raw-mapping read is
  gone. `poetry.lock` refreshed for the new `geecs-schemas` path dependency
  of ImageAnalysis / ScanAnalysis. Behaviour unchanged; legacy flat camera
  configs in the tree still degrade to a log line.

## [0.20.2] - 2026-09-03

### Fixed

- `poetry.lock` refreshed for ImageAnalysis 1.14.1 (#752): the boto3
  chain (boto3 / botocore / jmespath / s3transfer) leaves the lock.
  matplotlib's resolution is unchanged here — the portal declares it as
  a main dependency itself. Lock-file refresh only — no code change.

## [0.20.1] - 2026-09-03

### Changed

- `deploy/geecs-data-portal.service` is now a **template** rendered from the
  host's `site.env` by `deploy/render_units.sh` (account/checkout root/poetry
  path as `@PLACEHOLDER@` holes; experiment, `TZ`, and the
  `--processing-configs` tree — `"${GEECS_CONFIGS_ROOT}/scan_analysis_configs"`,
  previously typed into the installed unit by hand — via `EnvironmentFile=`).
  `DEPLOYMENT.md` updated. No runtime change.

## [0.20.0] - 2026-09-02

### Added

- **The browsing API** — everything the pages show, as JSON, so a
  script or an AI agent (the OSPREY assistant's `geecs-data-portal`
  skill) can read what a browser shows and press what a browser
  presses without scraping HTML: `GET /api/day/{iso}?experiment=&filter=`
  (the day page's table, newest first, same filter haystack),
  `GET /api/run/{uid}` (the rail + the Overview table verbatim +
  start/stop documents + the device list + the scan steppers'
  neighbours and dropdown + the processing selector's options +
  `analysis_enabled`, the Analysis tab's gate), `GET
  /api/run/{uid}/device?device=` (one device's gallery tier and path —
  the click on a device name), and `GET /api/run/jump/{iso}?prefer=`
  (the day steppers' target, as data instead of a redirect). The
  listings and neighbour logic moved into shared helpers the templates
  now use too (`_list_day`, `_neighbours`, `_resolved_folder`,
  `_jump_target`, `_scan_label`, `_parse_iso_day`), so the HTML and
  JSON surfaces cannot drift. Served `no-cache` (a day gains
  scans, a running run gains its stop doc). Documents pass through
  `analysis.jsonable_document` (NaN / numpy / set → JSON-safe — a
  document key must never 500 a run).
- `GET /health` now carries `version` (the same key every `/api`
  fetch is busted by), and a test pins that `/openapi.json` lists the
  whole agent surface — the deployed route list is discoverable.

## [0.19.0] - 2026-09-02

### Changed

- Analysis tab artifacts (owner ruling after the first live look,
  2026-09-02): the scan-level figures — averaged image, image grid,
  1D summary, animation — show automatically; the per-bin visuals
  (`<name>_<bin>_processed_visual.*`) show **one at a time** behind a
  prev / next stepper ("bin N (i of M)"), like the Images tab's shot
  stepper, instead of every bin at once. The classification
  (`kind` ∈ summary / bin / other, `bin` number) is ScanAnalysis's
  own output-name contract, parsed by its
  `renderers.config.parse_output_filename` (ScanAnalysis 1.18.0, next to
  the code that builds the names; imported lazily) — a bin's data file
  (`_<bin>_processed.h5`) travels with its visual behind the stepper,
  never as an auto-shown link (visual first, then the data file). The
  listing is ordered summaries → others → bins by number, capped on
  DISTINCT bins only (never the summaries, never splitting a bin), and
  always describes what is on disk under the
  analyzer's output dir plus the finished job's non-file labels
  (before: the job's own list when done, which was the summaries only).
  A stepper click re-renders that analyzer's row only.

## [0.18.0] - 2026-09-01

### Added

- **Rendered view** on the Images tab (PR 4 of the 04 design, the
  interactivity stop-gap): `display.mode = "rendered"` (the "rendered
  figure" checkbox in the display popup) serves the analyzer's own
  matplotlib figure — overlays, axes, colorbar — instead of the windowed
  processed pixels, through ImageAnalysis 1.14.0's
  `render_diagnostic_ephemeral` (object-API figures, no pyplot on the
  threadpool; same write-free contract and denylist). Per-bin view
  renders the averaged processed image through the base renderer with
  per-shot overlays dropped. `cmap` and the percentile window apply
  (absent/unknown colormap → `gray`, the pixel view's palette; `mode`
  outside `("", "rendered")` is a 400 — the enum precedent). Without a
  `processing` selection the mode is ignored (raw pixels). Status
  ladder matches the pixel path: a result that ran but cannot be drawn
  (`RenderError`) is a 404 like "render failed" / "produces no
  processed image". Review #766 also: the three "configured? installed?"
  preambles collapsed into `_ephemeral_module()`; `resources.figure_png`
  now serves `/plot.png` too; `to_display_png` uses `safe_cmap`; a legacy
  dict-returning analyzer in the pixel path is a 404, not a 500.

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
