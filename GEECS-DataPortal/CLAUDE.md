# GEECS-DataPortal — Developer Context for Claude

The scan-browsing web service — read-only except explicit analysis runs
(0.16.0): a zero-install browser view over the Tiled catalog (and, from
phase 4, the data share's image files) for anyone on the lab network.
**The architecture rules below are the spec** — read them before
extending this package.

## Architecture rules

- **Read-only, except explicit analysis runs, the config editor,
  sending a plot to the logbook, and the Plot tab's settings sets**
  (charter amendments: owner rulings 2026-09-01, 2026-09-06 and
  2026-09-15; the settings sets are the author's, 2026-10-28, for the
  maintainer to confirm — below, "Open, for the maintainer").
  The portal itself has no write verbs: no annotations.  Four
  exceptions, all explicit opt-ins; the fourth, the settings sets, writes
  JSON files into the settings folder only (below, "The Plot tab"), never
  the data share's scan tree.  (A MOUNTED scan logbook was a
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
  `geecs_portal.processing.render_document_as_run` — ScanAnalysis'
  `core_preview.preview_frame`, the analysis sink's own per-frame call
  (`single` with the document's `figure` block, frame inputs resolved
  under the scan), cropped tight like the sink's PNGs, so the pane shows
  the product file a run would write; no portal palette or window reaches
  it (pinned byte-for-byte against that seam). Its **summaries** block
  (`POST /configs/api/preview/summary`, `render_summary_as_run` →
  `core_preview.preview_summary`) draws each summary over the scan's first
  shots (`params.shots`, at most 8 — never a whole scan on the shared
  host), one panel per shot. A document the core declines at compilation
  is a 400 naming the reason, in every view (the legacy ImageAnalysis
  fallback was retired in #1039).
  The Images page's processing preview (its big view, on a shot:
  `image.png?processing=`, the choices from the device route's
  `processing_options`) keeps
  `render_document_ephemeral` and the portal's own window, analysing
  through the same seam.
  The other is `POST /api/run/{uid}/analysis`, which
  runs ONE ScanAnalysis analyzer on ONE scan on the user's click —
  `geecs_portal/analysis_runs.py`, calling `ScanAnalyzer.run_analysis`
  directly on a single worker thread with an in-memory job record.
  Deliberately **not** a participant in the file-based task queue
  (`scan_analysis.task_queue`: no status records, claim locks,
  heartbeats or Google Doc uploads) — "just run it"; since the
  GEECS-MCP rebuild its `run_analysis` IS this route (the agent's runs
  queue behind an operator's on the same worker thread) — the server
  runs no analysis of its own.  A run writes wherever
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
  The image comes from exactly one of two sources: `image`, the Plot
  tab's data URL, or `path` (GEECS-MCP arc, 2026-10), an analysis
  artifact named as the run listing names it — read off the analysis
  tree under the artifact route's containment rule, raster images only,
  so a client holding only a reference (the MCP server posting an
  agent's figures) moves no bytes; it posts to the RESOLVED folder's
  tag (day + number), not the start document's day, so across midnight
  a figure joins the scan it came from.  The reply carries the attachment
  `link` the body now shows.  `create_app(logbook_client=…)` injects the
  httpx client the send uses (tests: an in-process logbook); production
  makes one per send.
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
- **On the GeecsWebTheme kit** (2026-10-28, the author's override of
  the "portal last" ruling — below, "Open, for the maintainer").  Every
  page is `body class="kit"`: the kit's top bar, shell and rail, panels,
  `.chip` status words (a run's `exit_status` through `app.run_state`, an
  analysis job's state through the page's `JOB_STATES`), the five pane
  states (`paneState(kind, text)` in the page script), its `.tabs` (the
  selected tab `aria-selected`, every other panel `[hidden]`), its toast
  (`GeecsKit.toast`), and the overlay ladder: the filter, bin, display,
  send-to-log and config-editor panels are rung-2 drawers, the
  destructive confirmation a rung-3 `<dialog>`.  What the kit does not
  settle lives in `static/portal.css`, never in an inline style block; a
  component another surface would want goes into the kit through
  `kit.html` instead.  The glue is `geecs_web_theme.web`
  (`ForwardedPrefixMiddleware`, `make_templates`, `mount_theme`,
  `root_of`), never a copy (`tests/test_page.py`).
- **No build chain.**  Server-rendered Jinja2 templates; page behaviour
  is plain inline JS.  No npm, no CDN assets
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
  `X-Forwarded-Prefix` via the theme's `ForwardedPrefixMiddleware`,
  else `--root-path`), page JS builds fetches from `const ROOT`, and
  redirect endpoints prefix with `root_of(request)`.  Pinned by
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
  app.py         # create_app(catalog, default_experiment=…): builds the
                 #   PortalState, mounts static/theme, includes the routers
  state.py       # PortalState on app.state (catalog, caches, config, the
                 #   helpers that need them); get_state = the one Depends
  routes/        # one APIRouter per family: browse (health + browsing
                 #   JSON), pages (HTML), plot_api (plot and grid JSON + bin
                 #   membership), images (shot/bin images, traces,
                 #   plot.png), runs (analysis runs + artifacts), logbook
                 #   (plot send), config_editor (mount + previews);
                 #   common = stateless request helpers
  analysis.py    # /api boundary chores: filters/bincfg/display parsing,
                 #   JSON shaping (NaN→null), "show the code" snippets
  processing.py  # core compilation/execution and previews (every draw is the core's)
  analysis_runs.py  # analysis runs: AnalysisRunner (one worker thread,
                 #   one job per scan, thread-scoped log capture), the
                 #   ScanAnalysis factory seam, artifact containment
  logbook_send.py   # the four-call conversation that puts a rendered
                 #   plot into a scan's logbook entry (HTTP only)
  figures.py     # server-side figure authoring (plotly.py) for the /api routes and the 2-D map:
                 #   palette, base layout, multi-axis ladder, display
  plotter/       # the Plot tab: GEECSplotter's page, mounted at /plot (its
                 #   app, routes, settings sets, figures, image strip)
  resources.py   # figure_png (a shot's pixels, traces and PNG are
                 #   geecs_data_utils.io.shot_images / .display_png)
  static/        # the vendored Plotly bundle (the ONE committed JS asset)
                 #   and portal.css (what the kit does not settle)
  __main__.py    # CLI (geecs-data-portal): real TiledScanCatalog + uvicorn;
                 #   --config-editor mounts scan_analysis.config_editor at /configs
                 #   --logbook-url  the logbook service's base URL (its own process,
                 #                  :8400); the run page's "log" link; empty = no link
  templates/     # base.html / day.html / run.html (Jinja2; every colour a GeecsWebTheme token)
tests/
  test_app.py        # TestClient over FakeCatalog/StubCatalog (+ /api)
  test_resources.py  # tmp scan trees: gallery routes + union (the tier
                     #   ladder's own tests: Data-Utils' test_shot_images.py)
  test_analysis_runs.py  # the run ladder over an injected fake analyzer
  test_logbook_link.py   # the run page's link to the logbook service; /log is not served here
  test_logbook_send.py   # the send: the conversation (fake logbook over an
                         #   httpx transport) and the route's gate ladder
  test_page.py       # on the kit: status words, the theme's glue
```

**Canonical day view (owner ruling 2026-09-13):** this package's `/day/`
is the Tiled catalog's list of runs — plus, since 0.39.0, every scan
folder no Tiled run claims (`geecs_data_utils.folder_catalog`'s
`MergedScanCatalog`, wired in `__main__` under `--tiled`, which skips an
unreachable Tiled for five minutes after it fails), so Master Control scans and
other experiments on the share (`?experiment=Thomson`) browse through the
same pages, their scalars from the s-file alone; the logbook's `/day/` is the scan
folders on the share. When they disagree, **the folders are canonical for
now** (parity with LabVIEW Master Control) — Tiled is expected to take
over later, and the ruling and its consequences live in
`GeecsLogbook/CLAUDE.md`'s deferred-items table. Do not rule the other
way here.

Routes: `/` (redirect to today) · `/day/{iso}` (run list; `?experiment=`)
· `/run/{uid}` (the scan page: rail + Overview/Plot/Images/Analysis
tabs — Plot and Images are links to the plotter's pages `/plot/` and
`/plot/images`, Analysis only when runs are possible, see below;
`?tab=` is the page's state, and `tab=images` redirects to the Images
page) · `/run/jump/{iso}` (day-step
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
`/api/run/{uid}/bin-images?device=&filters=&bincfg=` (per-bin
membership: bins/counts/member shots over Data-Utils' `bin_rows` (the
filter as its `mask`, never a filtered frame — the plotter's own call),
the same bins as `/binned`; no page reads it since the Images pane
moved, see the debt below)
· `/run/{uid}/plot.png?y=&x=` (server-rendered scalar PNG, kept for
embedding) · `/run/{uid}/image.png?device=&shot=` (one rendered device
shot) · `/run/{uid}/bin-image.png?device=&bin=<index>&filters=&bincfg=`
(one bin's `nanmean`-averaged device image — `bin` is the INDEX in
`bin-images` order, both endpoints run the same membership call;
member shots load and average through Data-Utils'
`shot_images.average_rows` (with `processing`: `load_row_array` per
shot, processed, then `average_frames`), windowed once after
averaging; the per-shot refusals are that module's; always `no-cache` — bin membership comes off the union
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
The page and JSON routes share `PortalState.list_day` / `.neighbours` /
`routes.common._resolved_folder` — add a page-visible fact to BOTH surfaces through
those helpers, never to one template.  `/openapi.json` stays on (the
docs UI is off) so a client can discover the deployed routes; the OSPREY
deployment's `skills/geecs-data-portal` reference (htu-assistant repo)
mirrors every payload shape and must move with any change here.

Both image endpoints also take `?display=` (the shared display state's
image slice: `cmap` matplotlib-colormap name + `plo`/`phi` percentile
window — types 400 at parse, values degrade to grayscale/defaults;
edited via the Images plotbar's "display…" drawer — plus `mode`
(0.18.0): `"rendered"` serves the analyzer's own figure via
`geecs_portal.processing.render_diagnostic_ephemeral` (object-API
`Figure`, never pyplot — the one sanctioned matplotlib path on a
request thread; per-bin = the base renderer over the averaged image,
overlays dropped; the colormap defaults to `gray` like the pixel
view); only meaningful with `processing`, ignored without it. `mode`
is an enum: an unknown value is a **400**, the `bincfg`-enum
precedent and a deliberate exception to "values degrade" — a wrong
mode is a broken link, not a cosmetic) and
`?processing=<diagnostic id>` (the `processing` URL state): the named analysis diagnostic runs
**ephemerally** on the served pixels via
`geecs_portal.processing.process_images` — the write-free
router: the document compiles once per request batch and runs on
`geecs-analysis`; `UnsupportedRecipe` at compilation is a 400 naming the
reason (no second backend since #1039), and runtime or rendering errors
never retry either. The core accepts arrays only.  Per-shot renders the `processed_image`; per-bin
processes each member THEN averages (nonlinear-correct).  The feature
is **explicit-opt-in**: `--processing-configs <tree>` /
`create_app(processing_config_dir=…)` names the configs tree — the
portal deliberately never falls back to the global config resolution
(two competing resolution paths exist, so the portal names its tree
explicitly). The `analysis` extra installs the core runtime. A
missing config root or core runtime hides the selector and 404s the param.
Errors map honestly: unknown diagnostic 404,
denylisted/miswired 400, analyzer failure 400 — never a 500.
**Analysis runs** (0.16.0): `GET /api/analysis/catalog` (GEECS-MCP
arc, 2026-10) lists the tree **scan-free** — every loadable diagnostic
(`id`, `device`, `output_dir`, `destructive`, `kind` ∈
`recipe|job`) and every loadable group with its resolved
`members` in run order (ScanAnalysis's own `discover_groups` +
`load_analysis_group`, so a file that would fail a run — bad YAML, a
ref no diagnostic answers to — is dropped with a log line; the group walk
is uncached by choice: groups are few and the route is called when a
plan is made, not per page) — the planning surface
for an agent or a picker; the portal still runs one diagnostic per job,
so a group is a plan the caller expands.  `GET /api/run/{uid}/analysis`
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
**Destructive kinds and progress** (0.38.0): each listed analyzer
carries `destructive` (the document's `destructive` — true only for the
`.himg` compaction job, a `HimgJob` with `kind: himg_compact`; a recipe
is never destructive).  For such a kind the tab's row says *deletes files*
and its run button opens a real `<dialog>` (`showModal()`: focus trap,
`Esc`, inert background; focus on the input, never the delete button)
that asks the user to type **this scan's number** (`const SCAN_NUMBER`);
the POST carries it as `confirm=<number>` and the endpoint refuses a
destructive kind without exactly that value (**400**) — the gate is
server-side so no client, script or curl runs a delete unasked.  Past
that check the run's factory is called with `allow_destructive=True`
(`run_scan_analyzer(..., allow_destructive=)` → `create_scan_analyzer`,
which refuses a destructive kind without it — the same gate the task
queue and MCP never pass), so the confirmation and the opt-in are one
decision.  The listing carries `scan_number`, the value the check
compares against (from the resolved folder, like the run's tag), and
the dialog asks for that one.  The
job record also carries `progress` (`{done, total, phase}` — the last
thing the analyzer reported through `ScanAnalyzer.progress`, the
base-class hook `run_scan_analyzer` sets when the analyzer has it; most
never report, the `.himg` kinds report per frame from their child
process), and the tab shows it in the state badge while the run lasts
(`running · verifying 320/1806`).
**The config editor drawer** (0.21.0, `config_editor` in the page and
`/api/run/{uid}` payload): the Analysis tab's per-analyzer **edit** button
opens `scan_analysis.config_editor`'s form in a drawer over the page
(`openConfigEditor` in `run.html` loads `/configs/static/editor.js` on
first use); the preview renders the drawer's shot of the **recipe's own
data folder** (`diag.data_folder` — the server ignores any device in the
params; there is no device picker, 0.37.0: it once drew one camera
through another's recipe), on demand via its `preview` button or per
edit with `auto`; **duplicate as** copies the open recipe for the same
device with `output_name` set to the new id; a save refreshes the analyzer list.  The editor's own API is documented
in `ScanAnalysis/CLAUDE.md`.
`GET /run/{uid}/artifact?path=<relative>` serves one produced file —
the resolved path must stay inside the scan's own analysis folder
(`analysis_runs.contained_artifact`; symlinks resolved), else 404.
Build + run + `cleanup()` all happen on the worker thread, so config
and analyzer failures become `failed` records, never 500s; every run ends
with `release_memory()` (gc + glibc `malloc_trim`) before its record flips
to finished, because the allocator otherwise keeps a run's peak resident
in this long-lived, memory-capped process (2.8 GB after one scope scan);
`run_analysis` returning `None` and `DataUnavailableWarning` map to
`no_data` (the worklist runner's own mapping).  The factory seam
(`create_app(analysis_factory=…)`) is how the tests drive the whole
ladder without ScanAnalysis; the default is `load_diagnostic` +
`create_scan_analyzer`, needing the `analysis` extra (which since
0.16.0 also carries ScanAnalysis).  The factory (and `__main__`,
earlier) pins `matplotlib.use("Agg")` before importing ScanAnalysis:
its base class uses pyplot, and the one worker thread is what
serialises pyplot *in this process* — never add a second pyplot user
on a request thread.  Known and accepted: the 2D wrapper's per-shot
stage forks a `ProcessPoolExecutor` from this thread-rich process
(asyncio loop, request threadpool, cache warmers) — the classic
fork-with-threads hazard, pre-existing in every embedding of
ScanAnalysis (the MCP included) and not fixable portal-side; the live
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

## The Plot tab: GEECSplotter's page (`geecs_portal/plotter/`)

GeecsPlotter folded into the portal (the author's ruling, 2026-10-28,
below, "Open, for the maintainer"): its package moved to
`geecs_portal/plotter/` nearly unchanged, with its tests
(`tests/plotter/`), and `__main__` mounts its app at **`/plot`**
(`create_app(plotter=…)`; `None`, as the tests pass, mounts nothing and
the scan page has no Plot tab). It opens a scan by day and number
in the portal's own experiment (`--experiment`): a run of another
experiment, or one without a day or a scan number, has no Plot tab
(the author's ruling, 2026-10-29) and its day-list link goes to
`/run/{uid}`. The scan page has no plot of its own: its old plot pane
was deleted once its tools were the page's (below). The page is the Plot tab: the day
list opens a scan there, the scan page's Plot tab links there, and the
page's own tab bar reaches the scan's other tabs through
`/run/jump/{day}?prefer={scan}&tab=…` (so a Tiled run and a folder scan
both resolve). It keeps its own routes under `/plot` (`/plot/`,
`/plot/save-as`, `/plot/api/…`); the portal's `/api/run/{uid}/…` routes
and payloads are untouched. GeecsPlotter itself was deleted at the end
of the arc (2026-10-29); its design rules are below ("The Plot pages'
design rules"). The rules that are the portal's own:

- **Each viewer saves into their own set.** The page asks the viewer's
  name once (a URL without `set=` shows only that question, and a
  shared link asks before its first change; the browser keeps the
  answer, and `plotter.js` sends it as `owner` with every save). Opening someone else's set
  writes nothing; the viewer's first change copies it, with the change,
  into their own set, **replacing what their own set held** (it is the
  viewer's working copy; Save as keeps a named one), and switches to it
  (`pages._store`, and
  `copy_from` on `/plot/api/settings/{name}` for the saves that do not
  reload). The owner's set changes only by Save as onto its name, which
  asks before replacing. A change that names no viewer is refused
  (422), never saved where it is: the page asks the name before a first
  change on a shared link. `/plot/api/settings/{name}` names its target
  in the URL, like Save as, and so takes no owner. The name is kept under
  its own browser key,
  first borrowing the logbook author box's name when it fits. A link
  carries `set=`, which opens that set as
  it is now (ruling 3 of the brief: the portal's old URL-carried plot
  state is retired for this page).
- **The settings folder** is chosen once at start
  (`plotter.settings.resolve_store`): `--settings-dir`, else
  `{config.ini [Paths] geecs_data}/DataPortal/settings/` (the shared
  `user data` folder), else `<main checkout>/user data/DataPortal/settings/`
  (gitignored). On its first start beside GeecsPlotter's folder
  (`…/GeecsPlotter/settings/`) it copies those sets once
  (`adopt_plotter_sets`: never overwriting, never deleting). On the
  service host, name the folder through `config.ini` or
  `--settings-dir` in `GEECS_PORTAL_EXTRA_ARGS` (`DEPLOYMENT.md`).
- **The old Plot tab's tools** (the author's list, 2026-10-28) are the
  page's too, each a field of the set as the JSON text the scan page
  carried in its URL, checked with the same parsers (`analysis.parse_*`):
  the **shot filters** (`shot_filters`; the rail's chips and a drawer,
  live count from `/plot/api/filter-count`) narrow the shots before
  anything is drawn, the **binned view** (`xy_view`, `bin_config`: the
  scan page's `/binned` on this page's three-slot axes,
  `plotter.figures.binned_figure`; no fit is made to bin centres), the
  **display settings** (`display`: server-side through the portal's own
  helpers, the raw `layout` patch merged in the browser only) and the
  toolbar's **copy** and **send to the scan's log entry**, which posts to
  the portal's `/api/run/{uid}/logbook` with the run's uid (only with an
  absolute `--logbook-url`). `plot_tools.js` holds them; `/plot/api/figure`
  takes `view`, `filters`, `bincfg` and `display`, and its `code` rebuilds
  either view.
- **One figure code.** `plotter/figures.py` keeps only what is the
  page's own (GEECSplotter's axis ladder, its layout without a legend,
  the fits) and imports the rest from `figures.py`: the palettes,
  `page_figure`, the display cosmetics (`apply_display`, `marker_size`,
  `trace_color`) and the binned trace (`add_binned_trace`).
- **The `/api/run/{uid}/frame` and `/binned` figures have no page now.**
  The routes stay (the `/api` layer is the data contract: GEECS-MCP's
  `get_scan_scalars` reads `/binned`'s numbers, scripts read either),
  and so do their figures and snippets (`figures.shots_figure`,
  `binned_figure`, `base_layout`, `_multi_y_layout`;
  `analysis.frame_code`, `binned_code`): they are the notebook view of
  those payloads, no longer what a page draws. `/columns`' `default_x`
  and `grid_axes` have no reader in the portal either, nor `/grid`'s
  figures and snippet (the Plot page's map has its own route,
  `/plot/api/map`, over the same `figures.grid_document`), nor
  `/bin-images` and `bin-image.png` since the scan page's Images pane
  went (the Images page's bins are the strip's own tiles). Deleting any
  of it is an `/api` change for the owner, not a cleanup.
- **Accepted debt, for a later PR of the arc:** the other
  duplicates the move brought in (`plotter/scans.default_x` is
  `routes/plot_api._default_x`; `scans.display_name` overlaps
  `_pretty_names`; fold toward the plotter's, which a page uses). That PR also hands the plotter the portal's catalog
  (`plotter/app.py` builds its own `FolderScanCatalog`, so the page is
  folder-only even under `--tiled`; the folder catalog caches finished
  scans itself). The filter and setting rows the plotter page takes from
  `portal.css` can become kit components (through `kit.html`) now that
  the scan page's rail filters are gone. A native camera's shots were
  background-warmed into the data cache when the scan page opened its
  Images pane (`data_cache.warm_native`, now without a production
  caller); the Images page's processed preview asks `image.png` one
  shot at a time, uncached until something warms it again.

## The Plot pages' design rules (from GeecsPlotter, 2026-10-29)

GEECSplotter, the LabVIEW tool, as two pages of the portal (`/plot/`,
`/plot/images`). Its decisions are settled — do not reopen them here.

- **Numerics are imported, never written here.** Column picks, coercion and
  the shot axis come from `geecs_data_utils.tiled_schema`; the frame from
  `geecs_data_utils.scan_frame` (the s-file alone — no Tiled run). Binning
  and filters are `geecs_data_utils.data.binning` / `.row_filters`, fits
  `data.curve_fit.fit_curve`. New math goes into GEECS-Data-Utils first,
  in its own PR.
- **Read-only on the share.** A scan not on the share is a 404 or the
  page's error state, never created
  (`tests/plotter/test_app.py::test_a_lookup_never_creates_a_scan_folder`).
- **The server authors the figure; the browser draws it.** Endpoints return
  `figure` (built with `THEMED_PALETTE`, serialized by `page_figure`, whose
  `$tok:--name` sentinels `plot_tools.js`' `themed` resolves against the
  theme) and `code`, the snippet that rebuilds the frame and the figure with
  the same functions (`tests/plotter/test_app.py::test_the_code_rebuilds_the_same_numbers`).
- **Every control lives in a named settings set, not the URL.** The URL
  is `?day=&scan=&set=`. The set keeps the last `day` and `scan` posted,
  so a URL with neither (a bare start, or switching sets) redirects to that
  scan; a URL's own day and scan win, a switch to a set with none keeps the
  open scan (`here_day` / `here_scan`), and otherwise it is the empty form.
  `extra="allow"` keeps fields a newer copy wrote, so every copy on the
  shared folder can load and save every set. The form POSTs its controls;
  the server merges them into the set's file (`settings.merged`: a field
  the page does not send keeps its saved value), saves it, and redirects
  back (303) to the page the form came from (`page`). A list filter saves
  as it is typed through `POST /plot/api/settings/{name}`. A new control
  is a new `PlotterSettings` field with a default, so older files still
  load; a change of meaning bumps `schema_version`. With `x_auto` on, X is
  the open scan's variable ("autofill par1"); a hand pick saves it off,
  and then X `""` is the shot number.
- **Set names** are lower-cased letters, digits, spaces, `-`, `_` and
  `.`, starting with a letter or digit and never ending in a space or a
  dot. A refused Save-as name is `NameRefused`, shown on the box (the
  kit's `aria-invalid` + `.err`); a store problem is the rail's error
  state. **Replacing a saved set** is Save as onto its name: `plotter.js`
  asks in a rung-3 `<dialog>` (focus on "Keep it"), whose Replace posts
  `overwrite=true`; without it the server refuses a taken name.
- **Y2 and Y3** are GEECSplotter's extra Y lists. `y_count` (the rail's
  "Y plots 1 | 2 | 3") says how many show and draw; lists past it keep
  their picks. Each list owns a slot (colour, axis id), so Y3 is Y3 with
  Y2 off (`shots_figure(…, ["a", None, "c"])`). `plotter.figures._y_axes`:
  Y1 left with the grid; every other axis on the right with a coloured
  title, the second one free at the plot's edge with `autoshift` (plus
  `_OUTER_AXIS_GAP`), so it sits just past the first's labels and title at
  any width (its title moves with it on the bundled plotly.js 3.1.1). The
  rail's swatch is the same `--trace-N` token. On a scan without a Y2/Y3
  column it draws nothing, but the list keeps the pick ("not in this
  scan") so the next click does not erase it.
- **Fits** are GEECSplotter's Fit menu, one per Y list (`y_fit`,
  `y2_fit`, `y3_fit`, plus `*_fit_order` for a polynomial; `"none"` by
  default). `plotter.figures.FITS` is the menu (LabVIEW's names, only the
  models data-utils has), `fit_columns` fits the drawn per-shot points,
  and `shots_figure(…, fits=)` draws each fit dashed on its slot's axis.
  `/plot/api/figure` returns the numbers as `fits` (each term
  `value ± stderr`, `api.plus_minus`), and the snippet rebuilds them. A
  fit name is stored as a plain string, so a set naming a fit a newer copy
  added still loads ("not in this version"). A new model is a data-utils
  PR first (its curve and `CurveFit.formula` live there), then a `FITS`
  entry.
- **Several scans** (the XY plot only; the author's design, 2026-10-29):
  the set's `extra_scans` (`[{day, scan}]`, from any day) beside the
  main `day`/`scan`, which the steppers, the day picker, the 2-D map and
  the Images page act on alone; the link stays `?day=&scan=&set=`. The
  rail's Scans group is a chip per scan and **Add scan…**, the kit's
  `GeecsKit.picker` over `/plot/api/day` (the plotter's folder catalog).
  `/plot/api/figure?scans=YYYY-MM-DD:N,…` reads every scan into one frame
  (`geecs_data_utils.data.scan_merge.merge_scan_frames`: rows tagged
  `scan` and `day`, the union of columns, each shot its own number; an
  s-file's own `scan` column naming the same scan is the tag), so a shot
  filter can name `scan`. `scan_view` picks the view: **overlay** (the
  default) draws one Y list (`overlay_y`), a trace, a fit and bins per
  scan, coloured `--trace-1..8` by its place in the list, at most
  `MAX_OVERLAY_SCANS` (8); **merged** (LabVIEW's "merge s files") plots
  the pooled frame as one scan, Y1–Y3, one fit per Y, and bins by the X
  value where the bin column is `Bin #` (it restarts in every scan:
  `plotter.multi.merged_bins`; X = shot # is refused). The pick lists
  are the union of the scans' columns, a column some lack marked "in 2
  of 3 scans". `plotter/multi.py` holds the frame and the groups the
  route and the snippet share; the `code` rebuilds either view.
- **The image strip** (the Images page): a tile is one shot's image or
  its bin's average (`geecs_data_utils.data.binning.bin_rows` over the
  WHOLE s-file, an empty `Bin #` a bin of its own; a filter would go in as
  its `mask`, never as a filtered frame, whose positions name other
  shots), laid out as a grid or one row (`image_layout`, `image_columns`;
  one row spreads that many tiles evenly over the scan, `plotter.js`). The
  ladder, the refusals, the trace check and the per-row average are
  `geecs_data_utils.io.shot_images` (`average_rows`), the PNG
  `io.display_png`. A row that says the device missed its shot, or a
  missing file, gives no image and is left out of its bin's average,
  **never a neighbour's image**. The picker lists every device folder
  with ONE listing and probes only the picked one
  (`images.is_image_device`): probing ~70 folders costs tens of seconds
  cold over the share. Each tile is an `<img loading="lazy">` on
  `/plot/api/images/tile.png`; tiles across, the colour map and the levels
  redraw in place and save as typed; device, tiles and layout reload. The
  **z scale** (`image_zscale`) is LabVIEW's autoscale: each tile from its
  own darkest to brightest pixel, one scale shared by every tile
  (`images.shared_levels` via `/plot/api/images/levels`, which reads EVERY
  tile first — the author's ruling, and the page says so), or black and
  white typed in ("fixed"). There is **no percentile window**: a laser
  mode is a small spot and a percentile white saturates it (the author,
  2026-10-28). The **ROI** is kept per camera (`image_rois`, `[x0, y0,
  x1, y1]`, end excluded), drawn in a rung-2 drawer, applied to every tile
  and to the shared scale. Tiles are kept in memory
  (`images.kept_tile_array`, at most `_KEPT_BYTES`) so the shared scale and
  the draw read a file once; a tile with no image is kept as its reason
  for `_REFUSAL_SECONDS` only, so a passing failure is never made
  permanent. The big view's colour bar is `tile.png?bar=true` (matplotlib's
  object API, the same levels).
- **The theme and kit are imported** (`geecs_web_theme.web`); every
  colour is a token. `GeecsWebTheme/tests/test_no_literal_colours.py`
  walks the plotter's template, CSS and scripts.
- **`--browser`** (the portal's CLI): once the server is up it opens the
  portal in the system browser (`default`), a named one, or `none` (what
  the service unit passes); a Linux host with no display opens nothing.
- No experiment, host or path in code: `config.ini`, or `--experiment`.

## Open, for the maintainer (from the merge's briefs, 2026-10-29)

The arc's two briefs (the plotter's and the merge's, under Planning)
were deleted with GeecsPlotter; what
they left open is here, for Sam at the promotion PR to `master`.

- **The author's three overrides of Sam's rulings** (2026-10-28):
  portal/plotter integration now rather than later (and no separate
  package); the portal onto the kit now rather than last; view state in
  named settings sets rather than the URL (old links with `y=`, `x=`,
  `filters=` … open the scan but no longer reproduce the plot, including
  links stored in logbook entries, and new entries link the live set).
- **`Planning/epics_estate.md`** plans this merge differently (a React
  `geecs-web` app that deletes the kit layer). Neither plan binds; this one
  is built.
- Whether `PlotterSettings` stays in the portal or moves to GEECS-Schemas.
- **A run's status word has three tables**: the scanner (`EXIT_WORDS`:
  abort → failed), the logbook (`KIT_STATE`: aborted → degraded) and the
  portal (`app.run_state`: abort → degraded, no stop document → unknown);
  one table in GEECS-Data-Utils needs a ruling on abort first.
- Settings sets are shared files with no revisions: two people editing one
  set overwrite each other, as with LabVIEW's settings files.
- **For the author:** the LabVIEW settings file format — sets could be
  imported from it if it is readable.
- **Still to build** (the author's lists): clicking a plot point to open
  its shot, several cameras in one layout, GIF export; deferred: gauss
  prop, peak/valley, xcorr-aligned bin averages, a trace waterfall;
  several scans on the 2-D map and the Images page (the XY plot has
  them: "Several scans", above); and the strip honouring the shot filters.

## Adding an analysis tab (the checklist)

The W2-must-be-dramatically-cheaper checkpoint is the point of this
list — a new tab should be exactly these steps:

1. **Primitive** — the numerics live in GEECS-Data-Utils (or
   GEECS-Analysis), pure and hermetically tested there.  If the tab
   needs new math, that PR comes first.
2. **Endpoint** — one `/api/run/{uid}/…` route in `routes/plot_api.py`: parse
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
4. **Tab** — a `<section role="tabpanel" class="tabpane">` + a
   `role="tab"` button in `run.html`'s `.tabs`; `Plotly.react` over the served figure (vendored
   bundle); keep all view state in the URL (extend
   `readState`/`writeState`); expansion UI goes behind a drawer/⚙,
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
`geecs-data-portal --experiment <name>` serves directly for development,
over the share's scan folders alone; `--tiled` adds the Tiled catalog
(the unit passes it). Off by default since 2026-10-28: an unreachable
Tiled cost every page that lists a day about a minute of connect
retries, on a laptop off the lab network.

## The resource viewer (arc phase 4)

(run, device, shot) → displayable image, per the Tier A/B/C ladder in
`geecs_data_utils.io.shot_images` (moved down from `resources.py` so the
plotter resolves a shot the same way; the event-row joins are its
`row_acq_timestamp` / `row_file_shot`, the PNG `io.display_png`).
Since 0.12.0 the ladder resolves to raw pixels first
(`load_shot_array` → `ShotArray`); `load_shot_image` is the
render-one-shot wrapper over it, and per-bin averaging consumes the
arrays directly — one resolution path, one cache ride:
capture HDF5 stacks first (`geecs_data_utils.io.scan_stack` — joined by
`read_stack_timestamps` + `native_files` millisecond keys against the
event row's `acq_timestamp`, ScanAnalysis parity; NEVER by frame
ordinal when a timestamp is available — pre-scan extra frames shift
ordinals, FORMAT.md caveat a), else native files via `native_files`
exact stat-probes (legacy `ScanNNN_device_shot` names tried first, numbered
by the row's announced shot on a scan-mode run, from the s-file when the
folder catalog loaded no event table — `routes.common._file_shot`;
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
one layout. `shot_images.stack_content(probe)` (Data-Utils) names which a device has,
from the stack's own declaration (`stack_content_kind` in
`geecs_data_utils.io.scan_stack`: the plugin writes `wave_*` attributes
for an array variable and never for an image one, and **rank alone
cannot tell `(N, H, W)` pixels from `(N, n, 2)` rows**). Anything not a
readable array stack reads as `"image"`, so a damaged or unfamiliar
stack degrades to the path the portal has always taken.

For an array device the tab serves a **figure, not a PNG**:
`shot_images.load_shot_trace` → `ShotTrace` → `figures.trace_figure` →
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
  ruling: resampling belongs to whoever averages): a trace device shows
  one shot at a time whatever the per-shot/binned switch says.

## The 2-D map (the Plot page's view; the Grid tab until 2026-10-29)

The Grid tab moved onto the Plot page as its **2-D map** view (the
author's ruling: one tab per job; the scan page no longer has a Grid
tab, and a `tab=grid` link opens its Overview). The switch reads *XY
plot | 2-D map*; the map shares the rail's shot filters
and saves into the viewer's set: `map_value`, `map_x`, `map_y` (each
`None` = the default — Y1, the scan's variables — so a set that follows
the viewer maps each scan's own axes), `map_filter`, and `map_config`
(the statistics as `GridConfig` JSON, its column fields ignored;
checked with `analysis.parse_gridcfg`). `plotter/maps.py` resolves the
columns and writes the snippet; `/plot/api/map` answers with
`figures.grid_document`, the one payload shape it shares with the
scan's `GET /api/run/{uid}/grid` (kept: the `/api` contract);
`plotter/static/map.js` draws both maps square, steps the cells (the
selected cell is the browser tab's, in `sessionStorage`), checks the
number boxes before saving, and links a cell to the Images page under
the same set, its big view on that cell's bin (`?bin=`: the bin's tile,
or on a shots strip its first shot).
Numerical work stays in `scan_grid`.

**Measured positions only, for now** (the author's ruling,
2026-10-29): the Plot page reads scans through the folder catalog,
whose start document names one scan variable, so a folder scan's map
has X = that variable, Y the viewer's pick, and each bin at its
measured position (`kind == "points"`). A Bluesky 2-D scan's planned
grid (both axes picked, unacquired cells drawn) needs the scan's Tiled
record: it comes back when the plotter is handed the portal's catalog
(the debt below). The map offers no "Equal cells · indexed" spacing
until then: it needs a recorded grid, and a saved one would fail every
scan's map.

The map's geometry is resolved before filters. Both maps share coordinates and selection
but use independent color scales. Figure edges preserve nonuniform/log spacing;
equal-cell mode labels indices with physical values. Square refers to pixel area,
not equal physical units. No interpolation is applied to arbitrary trajectories.
The Images link uses the same acquisition-bin column and filters as the map, keeping
original bin indices for the image endpoint. Higher-dimensional slicing is deferred.

The Plot page's copy and send-to-log act on the clicked plot, a map
included: map captions name the scalar, statistic, axes and visit, and a
sent map keeps its plotting area square (`plot_tools.js`).

## The Images tab: the plotter's second page (`/plot/images`)

The author's ruling (2026-10-28): Images is the image strip with the big
single-shot view under it, reading the viewer's set. The Plot page's
"Image strip" view moved here, and the scan page's Images pane and its
filter rail went (a `tab=images` link redirects here). One set, two
pages: the page form's `page` field says which one a save reopens
(`pages._opened`), and its `tab` field carries the Plot page's view (XY
plot or 2-D map), so a save here never moves the Plot page; a set an
older copy saved on its strip (`tab="images"`) opens the XY plot.
The rail is the strip's (camera, bins or shots, layout, colour map, z
scale, ROI). `plotter/static/images.js` draws the **big view**: the tile
clicked (or stepped to with ‹ ›) at full size: the tile's own picture
(its ROI, levels and colour map), redrawn whenever the tile's picture
changes (a shared z scale read late, a new colour map) through a
`MutationObserver`. With a recipe picked, a shot is the portal's
processing preview instead (`image.png?processing=`, the whole frame on
its own levels; the choices are `/api/run/{uid}/device`'s
`processing_options`). A bin has no processing (the author's choice:
it always matches its tile). The strip shows every shot: the shot
filters narrow the Plot page's plots and map, not the strip, and a map
cell's link says so. A device whose stack holds traces
(`images.trace_kind`) has no strip and draws one shot's trace from the
portal's `/api/run/{uid}/trace`. Tiles carry their bin label
(`images.Tile.bin`), which is how a 2-D map cell opens on its bin. The
selection is the browser tab's (`sessionStorage`), never the set's. Like
the Plot page, it opens the portal's own experiment only (the author:
one experiment at a time).

## Processing migration (0.31.0)

`processing.py` owns the temporary backend choice and the portal display style,
not numerical algorithms. Discovery/reads delegate to Data Utils, schema
validation to GEECS-Schemas, execution/rendering to geecs-analysis. The config
editor still uses ScanAnalysis ConfigStore; explicit scan runs still use its
factory. Bin images process each shot before averaging and omit shot overlays.
The image preview source ladder is unchanged; trace rendering accepts already
loaded Nx2 data but does not add native trace discovery to the editor yet.

Camera file backgrounds use `scan_analysis.core_inputs.prepare_recipe`, shared with
future scan execution: one load per prepared run, legacy constant fallback on
read failure, hard errors for loaded geometry mismatches. Unsaved previews use
their supplied document without saving it. No device-directory context is
invented for previews, so `{scan_dir}` remains literal there as before.
