# GeecsPlotter — Developer Context for Claude

GEECSplotter (the LabVIEW tool) as a web app: a scan's s-file, one column
against another, per shot (M1) and binned (M2). The brief and its
milestones are `Planning/geecs_plotter.md` until M5 lands; its decisions
are settled — do not reopen them here.

## Layout

```
geecs_plotter/
  __main__.py   # geecs-plotter --port 8600 (argparse + uvicorn, binds 127.0.0.1; opens --browser once up)
  app.py        # create_app(experiment, base_directory=…): theme glue, routes, /health
  scans.py      # find a scan folder, read its s-file: GEECS-Data-Utils calls only
  figures.py    # pure: columns in, plotly Figure out (copied from the portal)
  settings.py   # named settings sets: PlotterSettings, one JSON file per set, the folder
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
- **Read-only on the share.** A scan not on the share is a 404 or the page's error
  state, never created (`tests/test_app.py::test_a_lookup_never_creates_a_scan_folder`).
- **The server authors the figure; the browser draws it.** Endpoints return
  `figure` (built with `THEMED_PALETTE`, serialized by `page_figure`, whose
  `$tok:--name` sentinels `plotter.js` resolves against the theme) and
  `code`, the snippet that rebuilds the frame and the figure with the same
  functions (`tests/test_app.py::test_the_code_rebuilds_the_same_numbers`).
- **Every control lives in a named settings set, not the URL.** The URL
  is `?day=&scan=&set=` (`set` defaults to `default`). The set also keeps
  the last `day` and `scan` posted, so a URL with neither (a bare start,
  or switching sets) redirects to that scan; a URL's own day and scan
  win, a switch to a set with none keeps the open scan (`here_day` /
  `here_scan`), and otherwise it is the empty form. `extra="allow"` keeps
  fields a newer copy wrote, so every copy on the shared folder can load
  and save every set. The form POSTs its
  controls; the server merges them into the set's file
  (`settings.merged`: a field the page does not send keeps its saved
  value), saves it, and redirects back (303). A filter saves as it is
  typed through `POST /api/settings/{name}`. A new control is a new
  `PlotterSettings` field with a default, so older files still load; a
  change of meaning bumps `schema_version`. With `x_auto` on, X is the
  open scan's variable ("autofill par1"); a hand pick saves it off, and
  then X `""` is the shot number.
- **The settings folder** is chosen once at start
  (`settings.resolve_store`): `--settings-dir`, else
  `{config.ini [Paths] geecs_data}/GeecsPlotter/settings/` (the LabVIEW
  `user data` folder, usually shared), else `<repo>/user data/…`, where
  `<repo>` is the main checkout even from a `.claude/worktrees/` worktree
  (`settings.checkout_root`), so every copy shares one folder; a local save
  writes `user data/.gitignore` (`*`) besides the root rule. The page
  shows which folder. Settings writes are the only writes this package
  makes, and never under the data share's scan tree.
- **Set names** are lower-cased letters, digits, spaces, `-`, `_` and `.`,
  starting with a letter or digit and never ending in a space or a dot. A
  refused Save-as name is `NameRefused`, shown on the box (the kit's
  `aria-invalid` + `.err`); a store problem is the rail's error state.
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
