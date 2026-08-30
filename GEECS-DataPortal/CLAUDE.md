# GEECS-DataPortal — Developer Context for Claude

The read-only scan-browsing web service: a zero-install browser view over
the Tiled catalog (and, from phase 4, the data share's image files) for
anyone on the lab network.  **The scope document is the spec**:
`Planning/data_portal/01_data_portal_scope.md` — read it before extending
this package; the architecture rules below are its distillation.

## Architecture rules

- **Read-only by doctrine.**  No write verbs, no analysis triggering, no
  annotations.  Nothing on the scans path is ever created (repo
  scan-folder invariant — see root `CLAUDE.md`); the data share, when
  mounted for the resource viewer, is mounted read-only.
- **The ScanCatalog seam.**  `create_app(catalog)` takes any
  `geecs_data_utils.tiled_catalog.ScanCatalog`; this package never
  imports `tiled` directly and never talks to the catalog server except
  through that protocol.  The shared front-end helpers
  (`resolve_scan_folder`, `metadata_rows`, `daily_scan_folder`) come
  from GEECS-Data-Utils — never re-implement them here (the console
  scan browser shares them; import-identity is pinned console-side for
  `resolve_scan_folder` and `metadata_rows`).
- **Column semantics live in `geecs_data_utils.tiled_schema`.**  The
  plot pick list is `plottable_columns`, coercion is `numeric_series` —
  shared with the console's B4 so the two front-ends cannot drift.
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
- **The three-layer analysis contract** (03 design doc): pure
  primitives live in GEECS-Data-Utils (`scan_frame`, `row_filters`,
  `binning`) → the portal's `/api` endpoints are **one-liners over
  those primitives** (parsing/JSON chores live in
  `geecs_portal/analysis.py`) → tabs are client-side Plotly views of
  the JSON.  Every endpoint response carries a `code` field: the
  notebook snippet that reproduces it exactly (the reproducibility
  doctrine — never put numerics in an endpoint that a notebook can't
  import).  All view state is URL-carried (`tab`/`y`/`x`/`view`/
  `filters`/`bincfg`/`display` — filters/bincfg are the
  Pydantic/dataclass models' JSON, `display` the plot-cosmetics JSON):
  a link IS the analysis, which is also the multi-user story
  (statelessness; shared caches are a feature).  Every `/api` fetch
  additionally carries `v=<portal version>` — completed-run responses
  cache immutable, and the version key rolls browser caches over when
  a release changes the payload shape.
- **Hermetic tests.**  `tests/` drives the app through
  `fastapi.testclient.TestClient` over fake catalogs — no network, no
  data root, no config.ini.  Catalog failures must surface in the page
  (or as 404s), never as 500s or hangs.
- **Blocking catalog calls are fine here** (unlike the Qt console):
  FastAPI runs sync endpoints on a threadpool.  Keep endpoints sync
  unless something genuinely needs async.
- **Eager WITHIN a scan, lazy ACROSS scans** (owner amendment,
  2026-08-29, superseding the scope doc's blanket "lazy loading is a
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
  analysis.py    # /api boundary chores: filters/bincfg parsing, JSON
                 #   shaping (NaN→null), "show the code" snippets
  resources.py   # (folder, device, shot) → PNG bytes / tiered refusal
  static/        # the vendored Plotly bundle (the ONE committed JS asset)
  __main__.py    # CLI (geecs-data-portal): real TiledScanCatalog + uvicorn
  templates/     # base.html / day.html / run.html (Jinja2, dark palette)
tests/
  test_app.py        # TestClient over FakeCatalog/StubCatalog (+ /api)
  test_resources.py  # tmp scan trees: gallery routes + tier ladder + union
```

Routes: `/` (redirect to today) · `/day/{iso}` (run list; `?experiment=`)
· `/run/{uid}` (the scan page: rail + Overview/Plot/Images tabs;
`?tab=&y=&x=&view=&filters=&bincfg=&display=` is the Plot-tab state,
`?device=&shot=` the Images selection) · `/run/jump/{iso}` (day-step
redirect: `?prefer=<scan number>`, all other params carried to the
target day's matching/newest run; empty day → the day page) ·
`/api/run/{uid}/columns` (union pick list with provenance, a
`timestamp` flag on ts_ columns, + default X) ·
`/api/run/{uid}/frame?cols=&x=&filters=` (per-shot series, filtered;
timestamp columns arrive as host-local ISO datetimes with a `kinds`
map — presentation-side conversion the `code` snippet mirrors) ·
`/api/run/{uid}/binned?cols=&filters=&bincfg=` (centers + asymmetric
error bands — served RAW, no datetime conversion: a timestamp bin
column keeps its epoch-second labels) ·
`/api/run/{uid}/filter-count?filters=` (live pass count)
· `/run/{uid}/plot.png?y=&x=` (server-rendered scalar PNG, kept for
embedding) · `/run/{uid}/image.png?device=&shot=` (one rendered device
shot) · `/health` (catalog probe — the fleet-map health check).
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
   `jsonable_values`, attach a `code` snippet, serve with the
   completed-run cache headers.
3. **Tab** — a `<section class="pane">` + a `.tab` button in
   `run.html`; render with the vendored Plotly from the JSON; keep all
   view state in the URL (extend `readState`/`writeState`); expansion
   UI goes behind a popup/⚙, never into the rail.
4. **Tests** — endpoint tests against the fakes (hand-computed numbers,
   the 400/404 ladder, NaN→null, immutable headers on completed runs).

## Deployment

Runs on the queueserver worker host next to the capture daemon (default
port **8200**; Tiled is :8000, GEECS-MCP :8100).  The systemd unit is
`deploy/geecs-data-portal.service`; the runbook is `DEPLOYMENT.md`; the
fleet map (`docs/platform/fleet_map.md`) carries the service row and
must move in the same PR as any deployment change.
`geecs-data-portal --experiment <name>` serves directly for development.

## The resource viewer (`resources.py`, arc phase 4)

(run, device, shot) → displayable image, per the scope doc's tiering:
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
