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
  scan browser shares them; import-identity is pinned console-side).
- **Column semantics live in `geecs_data_utils.tiled_schema`.**  The
  plot pick list is `plottable_columns`, coercion is `numeric_series` —
  shared with the console's B4 so the two front-ends cannot drift.
  Never interpret event-schema column names or dtypes in this package.
- **No build chain.**  Server-rendered Jinja2 templates + minimal inline
  CSS; plots are server-side matplotlib PNGs via the **object API**
  (`matplotlib.figure.Figure`, never pyplot — no global figure registry
  on FastAPI's threadpool).  No npm, no CDN assets (control-room
  machines may lack internet).
- **Hermetic tests.**  `tests/` drives the app through
  `fastapi.testclient.TestClient` over fake catalogs — no network, no
  data root, no config.ini.  Catalog failures must surface in the page
  (or as 404s), never as 500s or hangs.
- **Blocking catalog calls are fine here** (unlike the Qt console):
  FastAPI runs sync endpoints on a threadpool.  Keep endpoints sync
  unless something genuinely needs async.

## Package layout

```
geecs_portal/
  app.py         # create_app(catalog, default_experiment=…) — all routes
  __main__.py    # CLI (geecs-data-portal): real TiledScanCatalog + uvicorn
  templates/     # base.html / day.html / run.html (Jinja2, dark palette)
tests/
  test_app.py    # TestClient over FakeCatalog/StubCatalog
```

Routes: `/` (redirect to today) · `/day/{iso}` (run list; `?experiment=`)
· `/run/{uid}` (metadata + column links; `?y=` selects the plotted
column) · `/run/{uid}/plot.png?y=&x=` (server-rendered scalar plot) ·
`/health` (catalog probe — the fleet-map health check).

## Deployment

Runs on the queueserver worker host next to the capture daemon (default
port **8200**; Tiled is :8000, GEECS-MCP :8100).  The systemd unit +
runbook land in `deploy/` (phase 5 of the arc); until then
`geecs-data-portal --experiment <name>` serves directly.

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
