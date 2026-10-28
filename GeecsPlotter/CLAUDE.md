# GeecsPlotter — Developer Context for Claude

GEECSplotter (the LabVIEW tool) as a web app: a scan's s-file, up to three
columns (Y1–Y3, each on its own colour-coded axis) against one X, per shot
(M1) and binned (M2). The brief and its
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
  images.py     # the image strip: devices, tiles (per bin / per shot), a tile's pixels (data-utils calls)
  routes/       # Context (+ the 404 for a missing scan); pages.py (HTML), api.py (JSON),
                #   images.py (the strip's JSON and each tile's PNG)
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
- **Y2 and Y3** are GEECSplotter's extra Y lists. `y_count` (the rail's
  "Y plots 1 | 2 | 3") says how many show and draw; lists past it keep
  their picks. Each list owns a slot (colour, axis id), so Y3 is Y3 with
  Y2 off (`shots_figure(…, ["a", None, "c"])`). `figures._y_axes`: Y1
  left with the grid; every other axis on the right with a coloured title,
  the second one free at the plot's edge with `autoshift` (plus
  `_OUTER_AXIS_GAP`), so it sits just past the first's labels and title at
  any width. Its title moves with it on the bundled plotly.js 3.1.1
  (re-checked 2026-10-27), so the portal's #725 "titles don't shift" note
  does not hold on this same bundle. The
  rail's swatch is the same `--trace-N` token. On a scan
  without a Y2/Y3 column it draws nothing, but the list keeps the pick
  ("not in this scan") so the next click does not erase it.
- **Fits** are GEECSplotter's Fit menu, one per Y list (`y_fit`,
  `y2_fit`, `y3_fit`, plus `*_fit_order` for a polynomial; `"none"` by
  default). The math is `geecs_data_utils.data.curve_fit.fit_curve`;
  `figures.FITS` is the menu (LabVIEW's names, only the models data-utils
  has), `figures.fit_columns` fits the drawn per-shot points, and
  `shots_figure(…, fits=)` draws each fit dashed on its slot's axis in its
  slot's colour. `/api/figure` returns the numbers as `fits` (each term
  `value ± stderr`, rounded by `api.plus_minus`) for `plotter.js` to list
  under the plot, and the snippet rebuilds them. A fit name is stored as a
  plain string, so a set naming a fit a newer copy added still loads
  ("not in this version"); the API reports it instead of fitting. A new
  model is a data-utils PR first (its curve and `CurveFit.formula` live
  there), then a `FITS` entry here.
- **Images** are GEECSplotter's image strip, the Images tab (`tab`;
  the brief's "Images"). A tile is one shot's image or its bin's average
  (`Bin #`, `geecs_data_utils.data.binning.compute_bin_key`), laid out as
  a grid or one row (`image_layout`, `image_columns`). The ladder,
  the refusals and the trace check are `geecs_data_utils.io.shot_images`,
  the average `io.images.average_frames`, the PNG `io.display_png` (the
  colour map needs matplotlib, a dependency here). A shot is an s-file
  row: a row that says the device missed it, or a missing file, gives no
  image, and is left out of its bin's average, **never a neighbour's
  image**. The picker lists every device folder with ONE listing, and only
  the picked one is probed (`images.is_image_device`): probing all of a
  scan's ~70 folders costs tens of seconds cold over the share. Each tile
  is an `<img loading="lazy">` on `/api/images/tile.png`, so a scan's
  images are never loaded all at once; tiles across, the colour map and
  the levels redraw in place and save as typed; device, tiles and layout
  reload. `/api/images/strip` returns the tiles and the `code` snippet.
  The **z scale** (`image_zscale`) is LabVIEW's autoscale: each tile
  from its own darkest to its brightest pixel, every tile on one scale
  from the darkest to the brightest pixel of all ("shared":
  `images.shared_levels`, from `/api/images/levels`, which reads every
  tile first), or black and white typed in ("fixed"); the last two reach
  the tile as `zmin`/`zmax`, drawn by `to_display_png(levels=)`. There is
  **no percentile window** (the portal's 1–99.7): a laser mode is a small
  spot, the top fraction of its ROI, and a percentile white saturates it
  (the author's ruling, 2026-10-28). The **ROI** is kept per camera
  (`image_rois`, `[x0, y0, x1, y1]` pixels, end excluded), drawn in a
  rung-2 drawer on the first tile, and applies to every tile and to the
  shared scale; `merged` takes it with the camera's name (`roi` + `device`),
  so picking another camera never carries one camera's ROI to the next.
  Tiles read for the shared scale are kept in memory
  (`images.kept_tile_array`, keyed by the camera folder's mtime) so
  drawing them does not read every file twice.
- **Replacing a saved set** is Save as onto its name (the box offers the
  saved names): `plotter.js` asks in a rung-3 `<dialog>` (focus on "Keep
  it"), whose Replace posts `overwrite=true`; without it the server
  refuses a taken name (`NameRefused`).
- **The theme and kit are imported** (`geecs_web_theme.web`); every colour
  is a token. `GeecsWebTheme/tests/test_no_literal_colours.py` walks this
  package's template, CSS and script, and `figures.py` in the token check.
- No experiment, host or path in code: `config.ini`, or `--experiment`.

## Accepted debt

- **`figures.py` is a copy** of the portal's `geecs_portal/figures.py`
  (`Palette`, the sentinel walk, `base_layout`, `shots_figure`, the multi-Y
  ladder as `_y_axes`), cut to three Y columns. The maintainer wants a separate package; the two copies may
  drift until portal/plotter integration, a later effort. Copy further
  portal code (the multi-Y ladder, `binned_figure`) the same way when a
  milestone needs it; never import `geecs_portal`.
- `plotter.js` carries its own copy of the portal's `resolveThemeTokens`.

## Out of scope

See the brief: Tiled, cross-scan merging, deployment. (Fitting and
images left this list on the author's rulings, 2026-10-27: the brief
records them.)
