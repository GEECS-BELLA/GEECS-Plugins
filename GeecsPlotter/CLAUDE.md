# GeecsPlotter — Developer Context for Claude

GEECSplotter (the LabVIEW tool) as a web app: a scan's s-file, one column
against another, per shot (M1) and binned (M2). The brief and its
milestones are `Planning/geecs_plotter.md` until M5 lands; its decisions
are settled — do not reopen them here.

## Layout

```
geecs_plotter/
  __main__.py   # geecs-plotter --port 8600 (argparse + uvicorn, binds 127.0.0.1)
  app.py        # create_app(experiment, base_directory=…): theme glue, routes, /health
  scans.py      # find a scan folder, read its s-file: GEECS-Data-Utils calls only
  figures.py    # pure: columns in, plotly Figure out (copied from the portal)
  routes/       # Context (+ the 404 for a missing scan); pages.py (HTML), api.py (JSON)
  templates/    # index.html, on the kit
  static/       # plotter.js, plotter.css, the vendored plotly-cartesian-3.1.1.min.js (MIT)
tests/          # pytest over a fake share in tmp_path (conftest.py)
```

## Rules

- **Numerics are imported, never written here.** Column picks, coercion and
  the shot axis come from `geecs_data_utils.tiled_schema`; the frame from
  `geecs_data_utils.scan_frame` (the s-file alone — no Tiled run). Binning
  and filters (M2) are `geecs_data_utils.data.binning` / `.row_filters`.
  New math goes into GEECS-Data-Utils first, in its own PR.
- **Nothing needs Tiled.** Data-utils is installed without its `tiled` extra.
- **Read-only.** A scan not on the share is a 404 or the page's error
  state, never created (`tests/test_app.py::test_a_lookup_never_creates_a_scan_folder`).
- **The server authors the figure; the browser draws it.** Endpoints return
  `figure` (built with `THEMED_PALETTE`, serialized by `page_figure`, whose
  `$tok:--name` sentinels `plotter.js` resolves against the theme) and
  `code`, the snippet that rebuilds the frame and the figure with the same
  functions (`tests/test_app.py::test_the_code_rebuilds_the_same_numbers`).
- **The URL is the selection** (`?day=&scan=&x=&y=`). `x` absent is the
  scan variable ("autofill par1"); `x=` empty is the shot number.
- **The theme and kit are imported** (`geecs_web_theme.web`); every colour
  is a token. `GeecsWebTheme/tests/test_no_literal_colours.py` walks this
  package's template, CSS and script, and `figures.py` in the token check.
- No experiment, host or path in code: `config.ini`, or `--experiment`.

## Accepted debt

- **`figures.py` is a copy** of the portal's `geecs_portal/figures.py`
  (`Palette`, the sentinel walk, `base_layout`, `shots_figure`), cut to one
  Y column. The maintainer wants a separate package; the two copies may
  drift until portal/plotter integration, a later effort. Copy further
  portal code (the multi-Y ladder, `binned_figure`) the same way when a
  milestone needs it; never import `geecs_portal`.
- `plotter.js` carries its own copy of the portal's `resolveThemeTokens`.

## Out of scope

See the brief: Tiled, images, fitting, cross-scan merging, deployment.
