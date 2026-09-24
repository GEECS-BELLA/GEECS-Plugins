# GEECS-DataPortal — Developer Context for Claude

The scan-browsing web service — read-only except explicit analysis runs
(0.16.0): a zero-install browser view over the Tiled catalog (and, from
phase 4, the data share's image files) for anyone on the lab network.
**The architecture rules below are the spec** — read them before
extending this package.

## Architecture rules

- **Read-only, except explicit analysis runs, the config editor, and
  sending a plot to the logbook** (charter amendments: owner rulings
  2026-09-01, 2026-09-06, and 2026-09-15).
  The portal itself has no write verbs: no annotations.  Three
  exceptions, all explicit opt-ins.  (A MOUNTED scan logbook was a
  fourth, 0.22–0.26: a router at `/log` behind a `log` extra.  Since
  GeecsLogbook 0.10.0 it is **its own service** on port 8400 and this
  package only reaches it over HTTP — `--logbook-url` is the base the
  run page's "log" link is built from and the plot send posts to
  (`tests/test_logbook_link.py`, `tests/test_logbook_send.py`); `/log`
  on the portal is nobody's route.)  **The config
  editor** (`--config-editor`,
  0.21.0) mounts ScanAnalysis' `config_editor` router at `/configs` over
  the `--processing-configs` tree: it writes analysis-config YAML into
  **that tree only** (the share copy of the configs repo, uncommitted —
  same standing as the scanner's preset writes), never the scans tree
  (pinned in `tests/test_config_editor_mount.py`).  Its preview
  renders the *unsaved* document on the scan page's current shot through
  `image_analysis.ephemeral.render_document_ephemeral` — the same
  write-free seam as the Images tab, but drawn the analyzer's own way
  (its default palette or the document's `scan.renderer` cmap/vmin/vmax,
  never the pixel view's gray; pinned there too).  The other is `POST /api/run/{uid}/analysis`, which
  runs ONE ScanAnalysis analyzer on ONE scan on the user's click —
  `geecs_portal/analysis_runs.py`, calling `ScanAnalyzer.run_analysis`
  directly on a single worker thread with an in-memory job record.
  Deliberately **not** a participant in the file-based task queue
  (`scan_analysis.task_queue`: no status records, claim locks,
  heartbeats or Google Doc uploads) — "just run it"; GEECS-MCP's
  `run_scan_analysis` is a separate, experimental runner and the two
  do not coordinate (accepted with the ruling).  A run writes wherever
  ScanAnalysis writes: figures/HDF5 under `analysis/ScanNNN/`, s-file
  columns (warn-and-overwrite), and for some analyzers derived
  subfolders inside the scan folder.  Nothing on the scans path is
  ever *created* by this package (repo scan-folder invariant — see
  root `CLAUDE.md`; the analyzer's own `ScanPaths(read_mode=True)`
  holds it on the run side), and every lookup the portal makes leaves
  the tree untouched.  The share must be mounted read-write where runs
  are enabled (`DEPLOYMENT.md`).  The run verb is unauthenticated on
  the lab network, same standing as the MCP verb — accepted: outputs
  are regenerable and the share is internal.
  The third is `POST /api/run/{uid}/logbook` (0.28.0, owner ruling
  2026-09-15): the Plot tab's "send plot to this scan's log entry"
  button, `geecs_portal/logbook_send.py`.  It writes to **the logbook's
  own service**, over that service's public API — create the scan's
  entry, upload the PNG, append the image paragraph — and to nothing
  else: not the scans tree, not the analysis tree, not portal state.
  The portal still never imports GeecsLogbook; the edge is three URLs.
  It exists because clipboard *image* writes are a secure-context
  privilege and the deployment is plain HTTP, so "copy the plot" cannot
  work there and this can.  Two shapes follow from the logbook's own
  design and must not be "simplified": images are APPENDED to one entry
  rather than each making their own (consecutive image paragraphs are
  what its renderer turns into a figure grid — the layout LogMaker's
  `gdoc_slot` numbering used to fake), and the portal link goes on the
  line ABOVE the images, because a paragraph between two images splits
  that run.  Which entry is remembered **per browser**
  (`localStorage`), not asked of the logbook, so two people plotting the
  same scan from two machines get two entries and no grid — accepted,
  because the alternative (find this scan's entry through
  `GET /api/day/{day}/entries`) would append a plot into whatever note
  an operator happened to be writing.  Do not upgrade the doc to claim
  "one entry per scan" without changing that.
  Sending needs an **absolute** `--logbook-url`: the call is
  server-to-server, and a path-shaped base names the browser's front
  door, not a host this process can dial (it links but cannot send —
  `logbook_send` in the page context and the run JSON).
- **The ScanCatalog seam.**  `create_app(catalog)` takes any
  `geecs_data_utils.tiled_catalog.ScanCatalog`; this package never
  imports `tiled` directly and never talks to the catalog server except
  through that protocol.  The shared front-end helpers
  (`resolve_scan_folder`, `metadata_rows`, `daily_scan_folder`) come
  from GEECS-Data-Utils — never re-implement them here (import identity
  is pinned in `tests/test_shared_helpers.py` for `resolve_scan_folder`
  and `metadata_rows`; the pin moved here when the Qt console's scan
  browser was deleted, 2026-09-14).
- **Column semantics live in `geecs_data_utils.tiled_schema`.**  The
  plot pick list is `plottable_columns`, coercion is `numeric_series` —
  one module so no front end reinterprets a column.
  Never interpret event-schema column names or dtypes in this package.
- **No build chain.**  Server-rendered Jinja2 templates + minimal inline
  CSS; page behaviour is plain inline JS.  No npm, no CDN assets
  (control-room machines may lack internet).  The one amendment (owner
  approval 2026-08-30): **vendored, version-pinned JS assets committed
  under `geecs_portal/static/`** — today exactly one, the Plotly
  cartesian bundle (`plotly-cartesian-<version>.min.js`, MIT), served
  from `/static/`.  Upgrading it = replacing the file + the template
  reference in one commit; never fetch it at build or run time.
  Server-side matplotlib PNGs (via the **object API** —
  `matplotlib.figure.Figure`, never pyplot: no global figure registry
  on FastAPI's threadpool) remain for `plot.png`-style endpoints;
  interactive analysis tabs render client-side from the `/api` JSON.
- **The three-layer analysis contract**: pure
  primitives live in GEECS-Data-Utils (`scan_frame`, `row_filters`,
  `binning`) → the portal's `/api` endpoints are **one-liners over
  those primitives** (parsing/JSON chores live in
  `geecs_portal/analysis.py`) → tabs render the JSON with the vendored
  Plotly.  Since 0.10.0 the figure itself is **authored server-side in
  Python** (`geecs_portal/figures.py`, plotly.py — the bake-off
  ruling): endpoints return a ready `figure` and the tab JS is
  `Plotly.react` over it — plot logic belongs in `figures.py` (pure,
  unit-tested), never back in template JS.  The raw `series`/`shot`/
  `bins`/`counts` keys stay alongside `figure` **on purpose**: the
  `/api` layer is the data contract (scripts, debugging, and future
  tabs read numbers, not figures) — the payload duplication is the
  accepted cost, not an oversight.  Every endpoint response
  carries a `code` field: the notebook snippet that reproduces it
  exactly, **figure included** — `shots_figure`/`binned_figure` accept
  the reproduced frame directly (the reproducibility doctrine — never
  put numerics in an endpoint that a notebook can't import).  All view state is URL-carried (`tab`/`y`/`x`/`view`/
  `filters`/`bincfg`/`display` — filters/bincfg are the
  Pydantic/dataclass models' JSON, `display` the plot-cosmetics JSON,
  whose `layout` key is a raw Plotly-layout passthrough deep-merged
  last **in the browser** — the untrusted URL-carried patch never
  executes server-side, and its prototype-pollution guard lives with
  it: cosmetic asks should land there, not as new per-knob code):
  a link IS the analysis, which is also the multi-user story
  (statelessness; shared caches are a feature).  Cache policy is
  by INPUT, not by URL: the per-shot `image.png` and `plot.png` read
  the event table alone and cache immutable once the run has a stop
  doc; everything over the union frame (`columns`, `frame`, `binned`,
  `filter-count`, `bin-images`, `bin-image.png`) is `no-cache`
  (`_UNION_HEADERS`), because the s-file half of the union GROWS after
  the run — ScanAnalysis appends its columns hours later (the
  7-vs-30-columns incident, 2026-09-01).  Every `/api` fetch and
  bin-image card still carries `v=<portal version>` so an upgrade
  that changes a payload shape rolls over whatever a cache kept.
- **Prefix-agnostic URLs.**  The portal works at root and under any
  reverse-proxy mount (OSPREY panel tabs).  Templates never write a
  root-absolute portal URL directly: every href/action/src carries the
  `{{ root }}` context value (the per-request root path —
  `X-Forwarded-Prefix` via `_ForwardedPrefixMiddleware`, else
  `--root-path`), page JS builds fetches from `const ROOT`, and
  redirect endpoints prefix with `_root(request)`.  Pinned by
  `tests/test_app.py::TestReverseProxy`.
- **Hermetic tests.**  `tests/` drives the app through
  `fastapi.testclient.TestClient` over fake catalogs — no network, no
  data root, no config.ini.  Catalog failures must surface in the page
  (or as 404s), never as 500s or hangs.
- **Blocking catalog calls are fine here** (unlike a Qt GUI thread):
  FastAPI runs sync endpoints on a threadpool.  Keep endpoints sync
  unless something genuinely needs async.
- **Eager WITHIN a scan, lazy ACROSS scans** (owner amendment,
  2026-08-29, superseding the arc's original blanket "lazy loading is a
  hard rule"): one diagnostic's per-scan data is ~100s of MB — trivial
  against server RAM, while every NAS/Tiled round trip is the real
  cost.  `geecs_portal.cache` holds completed runs' details
  (`CachingScanCatalog`) and pixel data (`ShotDataCache`: whole stack
  frames arrays; native shots warmed in the background) so shot
  navigation serves from memory.  The lazy rule still binds across
  scans: never eager-load a day, and never cache a still-running run's
  pixels or an ordinal (listing-order) resolution.

## Package layout

```
geecs_portal/
  app.py         # create_app(catalog, default_experiment=…) — all routes
  analysis.py    # /api boundary chores: filters/bincfg/display parsing,
                 #   JSON shaping (NaN→null), "show the code" snippets
  analysis_runs.py  # analysis runs: AnalysisRunner (one worker thread,
                 #   one job per scan, thread-scoped log capture), the
                 #   ScanAnalysis factory seam, artifact containment
  logbook_send.py   # the four-call conversation that puts a rendered
                 #   plot into a scan's logbook entry (HTTP only)
  figures.py     # server-side Plot-tab figure authoring (plotly.py):
                 #   palette, base layout, multi-axis ladder, display
  resources.py   # (folder, device, shot) → PNG bytes / tiered refusal
  static/        # the vendored Plotly bundle (the ONE committed JS asset)
  __main__.py    # CLI (geecs-data-portal): real TiledScanCatalog + uvicorn;
                 #   --config-editor mounts scan_analysis.config_editor at /configs
                 #   --logbook-url  the logbook service's base URL (its own process,
                 #                  :8400); the run page's "log" link; empty = no link
  templates/     # base.html / day.html / run.html (Jinja2; every colour a GeecsWebTheme token)
tests/
  test_app.py        # TestClient over FakeCatalog/StubCatalog (+ /api)
  test_resources.py  # tmp scan trees: gallery routes + tier ladder + union
  test_analysis_runs.py  # the run ladder over an injected fake analyzer
  test_logbook_link.py   # the run page's link to the logbook service; /log is not served here
  test_logbook_send.py   # the send: the conversation (fake logbook over an
                         #   httpx transport) and the route's gate ladder
```

**Canonical day view (owner ruling 2026-09-13):** this package's `/day/`
is the Tiled catalog's list of runs; the logbook's `/day/` is the scan
folders on the share. When they disagree, **the folders are canonical for
now** (parity with LabVIEW Master Control) — Tiled is expected to take
over later, and the ruling and its consequences live in
`GeecsLogbook/CLAUDE.md`'s deferred-items table. Do not rule the other
way here.

Routes: `/` (redirect to today) · `/day/{iso}` (run list; `?experiment=`)
· `/run/{uid}` (the scan page: rail + Overview/Plot/Images/Analysis tabs
— Analysis only when runs are possible, see below;
`?tab=&y=&x=&view=&filters=&bincfg=&display=` is the Plot-tab state,
`?device=&shot=` the Images selection) · `/run/jump/{iso}` (day-step
redirect: `?prefer=<scan number>`, all other params carried to the
target day's matching/newest run; empty day → the day page) ·
`/api/run/{uid}/columns` (union pick list with provenance, a
`timestamp` flag on ts_ columns, + default X) ·
`/api/run/{uid}/frame?cols=&x=&filters=` (per-shot series, filtered;
timestamp columns arrive as host-local ISO datetimes with a `kinds`
map — presentation-side conversion the `code` snippet mirrors) ·
`/api/run/{uid}/binned?cols=&x=&filters=&bincfg=` (centers + asymmetric
error bands — served RAW, no datetime conversion: a timestamp bin
column keeps its epoch-second labels, and the `x` pick's per-bin mean
positions are raw seconds too when X is a timestamp column — datetime
rendering is per-shot-only, deliberately; `x_centers` come reindexed
onto the y result's bins so diverging per-column dropna can never
shift points) ·
`/api/run/{uid}/filter-count?filters=` (live pass count) ·
`/api/run/{uid}/bin-images?device=&filters=&bincfg=` (the Images tab's
per-bin membership: bins/counts/member shots over `compute_bin_key` +
the same groupby semantics as `/binned`; the `view` URL state is ONE
per-shot ⇄ binned toggle shared by the Plot and Images tabs —
deliberate: a binned link means binned everywhere)
· `/run/{uid}/plot.png?y=&x=` (server-rendered scalar PNG, kept for
embedding) · `/run/{uid}/image.png?device=&shot=` (one rendered device
shot) · `/run/{uid}/bin-image.png?device=&bin=<index>&filters=&bincfg=`
(one bin's `nanmean`-averaged device image — `bin` is the INDEX in
`bin-images` order, both endpoints run the same membership call;
member shots that resolve to pixels average via the shared
`average_frames`, windowed once after averaging; per-shot refusals
carry over; always `no-cache` — bin membership comes off the union
frame) · `/health` (catalog probe + version — the fleet-map health check)
· **the browsing API** (0.20.0 — the pages' own data as JSON, the
agent/script surface; served `no-cache`): `/api/day/{iso}?experiment=&filter=`
(the run table, newest first, the page's filter haystack) ·
`/api/run/{uid}` (summary + the Overview table verbatim + start/stop
docs + `devices` + stepper `neighbours`/`day_runs` +
`processing_options` + `analysis_enabled` + `logbook`, the run's card in
the logbook service or null — only with `--logbook-url` and for a run of
the portal's experiment) · `/api/run/{uid}/device?device=`
(one device's tier/path via the same `device_kind` probe as the page) ·
`/api/run/jump/{iso}?prefer=` (the day steppers' target as data).
The page and JSON routes share `_list_day` / `_neighbours` /
`_resolved_folder` — add a page-visible fact to BOTH surfaces through
those helpers, never to one template.  `/openapi.json` stays on (the
docs UI is off) so a client can discover the deployed routes; the OSPREY
deployment's `skills/geecs-data-portal` reference (htu-assistant repo)
mirrors every payload shape and must move with any change here.

Both image endpoints also take `?display=` (the shared display state's
image slice: `cmap` matplotlib-colormap name + `plo`/`phi` percentile
window — types 400 at parse, values degrade to grayscale/defaults;
edited via the Images plotbar's "display…" popup — plus `mode`
(0.18.0): `"rendered"` serves the analyzer's own figure via
`image_analysis.ephemeral.render_diagnostic_ephemeral` (object-API
`Figure`, never pyplot — the one sanctioned matplotlib path on a
request thread; per-bin = the base renderer over the averaged image,
overlays dropped; the colormap defaults to `gray` like the pixel
view); only meaningful with `processing`, ignored without it. `mode`
is an enum: an unknown value is a **400**, the `bincfg`-enum
precedent and a deliberate exception to "values degrade" — a wrong
mode is a broken link, not a cosmetic) and
`?processing=<diagnostic id>` (the `processing` URL state): the named ImageAnalysis diagnostic runs
**ephemerally** on the served pixels via
`image_analysis.ephemeral.run_diagnostic_ephemeral` — the write-free
seam (its structural no-writes contract lives in ImageAnalysis
CLAUDE.md "Ephemeral runs"; the read-only doctrine is preserved by
construction).  Per-shot renders the `processed_image`; per-bin
processes each member THEN averages (nonlinear-correct).  The feature
is **explicit-opt-in**: `--processing-configs <tree>` /
`create_app(processing_config_dir=…)` names the configs tree — the
portal deliberately never falls back to the global config resolution
(two competing resolution paths exist, so the portal names its tree
explicitly), and ImageAnalysis rides the optional
`analysis` extra; missing either hides the selector and 404s the
param.  Errors map honestly: unknown diagnostic 404,
denylisted/miswired 400, analyzer failure 400 — never a 500.
**Analysis runs** (0.16.0): `GET /api/run/{uid}/analysis`
lists every loadable diagnostic in the same `--processing-configs`
tree with `applicable` (its data device — `scan.device`, else the
name — has a folder in this scan), its in-memory `job` record
(`queued`/`running`/`done`/`failed`/`no_data`, artifacts, error, the
run's captured log lines) and `files` (what is on disk under
`analysis/ScanNNN/<output_name>/`, so a page loaded after a portal
restart still shows earlier outputs) and `artifacts` — what the tab
shows (the files on disk + the done job's non-file labels), each
`{path, servable, inline, kind, bin}` decided by
`analysis_runs.describe_artifacts` (the ONE inline-raster policy; the
summary/bin classification is ScanAnalysis's own contract,
`renderers.config.parse_output_filename`, imported lazily — the page
never guesses from a path; summaries render automatically, bins one at
a time behind a stepper, the bin cap applied after classification);
`POST
/api/run/{uid}/analysis?analyzer=<id>` starts a run (202 + record;
feature off / extra missing / unknown diagnostic / folder unresolvable
→ 404; a job already active for the scan → 409 with its record);
the **Analysis tab** (0.17.0) renders that listing: badge per state, run /
re-run (disabled while any run is active for the scan), error + captured
log on failure, artifacts inline (raster) or as download links, the
inapplicable analyzers collapsed; it polls every 1.5 s while a run is
active and the tab button exists only when `analysis_enabled` (feature
configured + extra installed + folder resolvable) — a bookmarked
`tab=analysis` otherwise falls back to Plot.
**The config editor drawer** (0.21.0, `config_editor` in the page and
`/api/run/{uid}` payload): the Analysis tab's per-analyzer **edit** button
opens `scan_analysis.config_editor`'s form in a drawer over the page
(`openConfigEditor` in `run.html` loads `/configs/static/editor.js` on
first use); the drawer's own device + shot pick what the preview
renders (on demand via its `preview` button, or per edit with `auto`); a save refreshes the analyzer list and, when the Images tab
shows that diagnostic, the shot image.  The editor's own API is documented
in `ScanAnalysis/CLAUDE.md`.
`GET /run/{uid}/artifact?path=<relative>` serves one produced file —
the resolved path must stay inside the scan's own analysis folder
(`analysis_runs.contained_artifact`; symlinks resolved), else 404.
Build + run + `cleanup()` all happen on the worker thread, so config
and analyzer failures become `failed` records, never 500s;
`run_analysis` returning `None` and `DataUnavailableWarning` map to
`no_data` (the worklist runner's own mapping).  The factory seam
(`create_app(analysis_factory=…)`) is how the tests drive the whole
ladder without ScanAnalysis; the default is `load_diagnostic` +
`create_scan_analyzer`, needing the `analysis` extra (which since
0.16.0 also carries ScanAnalysis).  The factory (and `__main__`,
earlier) pins `matplotlib.use("Agg")` before importing ScanAnalysis:
its renderers use pyplot, and the one worker thread is what
serialises pyplot *in this process* — never add a second pyplot user
on a request thread.  Known and accepted: the 2D wrapper's per-shot
stage forks a `ProcessPoolExecutor` from this thread-rich process
(asyncio loop, request threadpool, cache warmers) — the classic
fork-with-threads hazard, pre-existing in every embedding of
ScanAnalysis (LiveWatch, MCP) and not fixable portal-side; the live
check runs an analyzer *while* images are being browsed.  The scan tag
handed to the analyzer is parsed from the resolved folder
(`ScanPaths(folder=…)`), never rebuilt from the start doc (the
folder's day is the claim-time day; the start doc's time is stamped
later).  Artifacts: raster images inline, every other type an
`attachment` with `nosniff` (the share is writable by many hands).
Shutdown: the lifespan refuses new runs and logs an in-flight one; it
cannot interrupt it (DEPLOYMENT.md, stop timeout).
Template links build their queries through the one sticky-query helper
(and the page JS mirrors the analysis state into the stepper links) so
navigating one control never resets another; the day page's "clear"
link is the one deliberate exception.

## Adding an analysis tab (the checklist)

The W2-must-be-dramatically-cheaper checkpoint is the point of this
list — a new tab should be exactly these steps:

1. **Primitive** — the numerics live in GEECS-Data-Utils (or
   ImageAnalysis), pure and hermetically tested there.  If the tab
   needs new math, that PR comes first.
2. **Endpoint** — one `/api/run/{uid}/…` route in `app.py`: parse
   params via `analysis.py` (`BadParam` → 400 — and that means full
   type/arity validation of every field, not just enum membership:
   plain-dataclass configs validate nothing themselves, so a
   wrong-typed field that only explodes inside the primitive is a 500
   waiting for a hand-edited URL), call the primitive, shape with
   `jsonable_values`, attach a `code` snippet, serve with
   `_UNION_HEADERS` (never `_png_headers` — that is for responses
   over the event table alone; the s-file half of the union changes
   after the run).
3. **Figure** — if the tab plots, its figure builder goes in
   `figures.py` (pure functions, plotly.py, unit-tested in
   `tests/test_figures.py`) and the endpoint serves it as `figure`.
4. **Tab** — a `<section class="pane">` + a `.tab` button in
   `run.html`; `Plotly.react` over the served figure (vendored
   bundle); keep all view state in the URL (extend
   `readState`/`writeState`); expansion UI goes behind a popup/⚙,
   never into the rail.
5. **Tests** — endpoint tests against the fakes (hand-computed numbers,
   the 400/404 ladder, NaN→null, `no-cache` headers even on completed
   runs).

## Deployment

Runs on the queueserver worker host (default
port **8200**; Tiled is :8000, GEECS-MCP :8100).  The systemd unit is
`deploy/geecs-data-portal.service`; the runbook is `DEPLOYMENT.md`; the
fleet map (`docs/platform/fleet_map.md`) carries the service row and
must move in the same PR as any deployment change.
`geecs-data-portal --experiment <name>` serves directly for development.

## The resource viewer (`resources.py`, arc phase 4)

(run, device, shot) → displayable image, per the Tier A/B/C ladder in
`resources.py`.
Since 0.12.0 the ladder resolves to raw pixels first
(`load_shot_array` → `ShotArray`); `load_shot_image` is the
render-one-shot wrapper over it, and per-bin averaging consumes the
arrays directly — one resolution path, one cache ride:
capture HDF5 stacks first (`geecs_data_utils.io.scan_stack` — joined by
`read_stack_timestamps` + `native_files` millisecond keys against the
event row's `acq_timestamp`, ScanAnalysis parity; NEVER by frame
ordinal when a timestamp is available — pre-scan extra frames shift
ordinals, FORMAT.md caveat a), else native files via `native_files`
exact stat-probes (legacy `ScanNNN_device_shot` names tried first;
`read_imaq_image` loads pixels; never re-derive the filename
convention), vendor-SDK formats (`.himg`/`.has`) as a path card, never
rendered.  The event column matches through
`tiled_schema.device_acq_timestamp_column`; a device that missed the
shot 404s — a missing shot must never render a neighbour's image.  Display rendering is
percentile-windowed (1–99.7) 16-bit → 8-bit PNG.  **Lazy per-shot
loading is a hard rule** (the NAS pathology is file count) — never
eager-thumbnail a scan from native files.  Device names are validated
against the scan folder's actual subfolders (path-traversal guard), and
every lookup — hit or miss — must leave the tree untouched (pinned in
`tests/test_resources.py`).

### Not every stack is pixels (0.30.0)

The same file plugin captures a device's **array** variables — scope
traces and spectra — so a scan folder holds three kinds of stack through
one layout. `resources.stack_content(probe)` names which a device has,
from the stack's own declaration (`stack_content_kind` in
`geecs_data_utils.io.scan_stack`: the plugin writes `wave_*` attributes
for an array variable and never for an image one, and **rank alone
cannot tell `(N, H, W)` pixels from `(N, n, 2)` rows**). Anything not a
readable array stack reads as `"image"`, so a damaged or unfamiliar
stack degrades to the path the portal has always taken.

For an array device the tab serves a **figure, not a PNG**:
`load_shot_trace` → `ShotTrace` → `figures.trace_figure` →
`/api/run/{uid}/trace`. The two transports differ on purpose — a 2048²
frame as JSON is absurd, a few thousand trace samples want the hover
readout — and the split is keyed on the stack, never guessed in the
template. Everything else is shared: the same shot→frame join, the same
404s, the same theme-sentinel walk.

Two rules for anyone extending this:

- **The portal never un-pads and never rebuilds an axis.** The gateway
  once padded lineouts (stacks recorded 2026-09-19..24) and stores a
  waveform's axis as
  `wave_x0`/`wave_dx`; `read_1d_data` owns both facts and hands back the
  trace at its true length. A second copy of that arithmetic here is how
  the two drift.
- **Per-bin averaging is not offered for traces.** Each shot carries its
  own axis, so averaging across shots is the consumer's job (the arc's
  ruling: resampling belongs to whoever averages). The per-shot/binned
  state stays shared with the Plot tab; only this tab's view is pinned.

## Grid tab (0.29.0)

`GET /api/run/{uid}/grid` consumes strict JSON `gridcfg` (Data Utils `GridConfig`)
and the existing `filters`/`day` parameters. It returns complete cell geometry,
counts, member shots, independent center/error statistics, paired server-authored
figures and a notebook snippet. `gridcfg` and selected `gridbin` travel in run URLs. `imagebin` independently
records explicit cell-to-Images navigation; merely opening Grid never filters
the image gallery, and changing the bin definition clears this image selection.
`templates/grid_script.html` is included by `run.html`; it only handles controls,
selection, sizing and Plotly rendering. Numerical work stays in `scan_grid`.

Grid geometry is resolved before filters. Both maps share coordinates and selection
but use independent color scales. Figure edges preserve nonuniform/log spacing;
equal-cell mode labels indices with physical values. Square refers to pixel area,
not equal physical units. No interpolation is applied to arbitrary trajectories.
The Images link uses the same acquisition-bin column and filters as Grid, keeping
original bin indices for the image endpoint. Higher-dimensional slicing is deferred.

The shared send-to-log modebar action exports its clicked Plotly host (Plot,
Grid average or Grid error), through the same dialog and remembered browser
entry. Grid captions identify the scalar, statistic, axes and visit; exports
retain square plotting areas independently of the Plot tab's display settings.
