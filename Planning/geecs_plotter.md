# GEECSplotter as a web app — the brief

For the developer who wrote GEECSplotter in LabVIEW and will rebuild it here
with Claude Code. Start a session at the repo root and say: *"read
Planning/geecs_plotter.md and start"*. This file is scaffolding
(`Planning/README.md`): the PR that lands M5 deletes it.

## Goal

Rebuild GEECSplotter — the LabVIEW tool that opens a scan's s-file and plots
one column against another, per shot and binned — as a small web app in this
repo. The browser is the front panel. A Python server reads the s-file,
computes the numbers and the figure, and sends the figure as JSON. Plotly
draws it. Every plot can be reproduced in a notebook from a copied snippet.
It runs on a laptop against the data share first.

## What GEECSplotter does today

The repo digested the tool in PR #710 (panel notes) and PR #717 (feature
inventory with proposed keep/adapt/drop). Both files were deleted later;
read them with
`git show e8df33f9d:Planning/data_portal/labview_peruser/notes.md` and
`git show 9277b608d:Planning/data_portal/02_labview_peruser_inventory.md`.
The rulings below are the ones recorded there.

| GEECSplotter feature | Recorded | In this app |
|---|---|---|
| Folder tree → year/month/day/scan → s-file | drop (portal navigation) | a day + scan-number form; no Tiled |
| "merge s files" | defer | later (cross-scan) |
| Listbox par 1–4 | adapt | one X picker + a multi-Y list |
| "autofill par1" (scan variable as X); "use aliases" | keep / defer | default X = the scan variable; raw column names first |
| Substring name filter on columns | keep | a filter box over the column list |
| Save/load selections and filter sets | adapt | the selection lives in the URL |
| Data filters: per-column bounds, include/exclude, outer OR / inner AND | keep | `RowFilters` in GEECS-Data-Utils |
| "filter max L2 each bin" (best N shots per bin) | keep | no primitive exists yet; add one to data-utils |
| FileType to Analyze (2 Parameter Plot, traces, …) | adapt | scalar plots only; traces later |
| User field, color table, Save png, copy tdms, loop counter, clipboard; "pretty PDF" | drop; defer | Plotly's own PNG export |
| Scan-info text | keep | a metadata table on the page |
| xy plots: pars 2–4 vs par 1, per shot; zoom, cursor, autoscale, same-scale | keep (top priority) | M1; Plotly built-ins; same-scale = linked axes |
| Sub-tabs Fit · L2L3 · All3 · histogram | adapt | any column as X covers L2L3; histogram = a trace type; fit deferred |
| "cf file" sub-tab | unknown | you know what it does — say so in the first session |
| Binned plot: bin by `Bin #`, median centre, quartile bars | keep (flagship) | M2; `BinningConfig(agg="median", err="iqr")` |
| Fixed/auto bin interval; use-bin#, same-scale, subtract-mean | keep / adapt | `bin_width` / `bin_edges`; small toggles, no per-knob code |
| Fitting, "gauss prop", peak/valley | defer | not in M1–M5 |
| triple/double/single/errors/**data** sub-tabs; export to Origin / txt | adapt | multi-Y; the data sub-tab = a table + CSV + the notebook snippet |
| Overview: per-bin averaged images (xcorr align); Intensity: trace waterfall | keep (later); xcorr defer | images out of scope here; M3's heatmap is the trace step |
| Setup · 3D graph · stats · XY Graphs · MultiScan · Analysis tabs | unknown | one line each from you: used or not |

Two numbers to pin from the LabVIEW source, which you have: the binned centre
statistic (median?) and the exact quartile error-bar definition. Never commit
LabVIEW source or screenshots with personal paths: this repo is public.

## Decisions already made (do not reopen)

- New package `GeecsPlotter/`, import name `geecs_plotter`, own
  `pyproject.toml`, Python `>=3.11,<3.12`, Poetry, in this repo.
- Depends on GEECS-Data-Utils and GeecsWebTheme (`web` extra), plus
  `fastapi`, `uvicorn`, `jinja2`, `plotly`, `pandas`. It never imports
  `geecs_portal` or any other app, never LogMaker4GoogleDocs, never the
  deleted legacy packages (root `CLAUDE.md` § "Deleted legacy packages").
- Duplication is allowed inside the package. Copy what helps from the
  portal: `GEECS-DataPortal/geecs_portal/figures.py` (`shots_figure`,
  `binned_figure`, `Palette`, `page_figure`), the parsing in `analysis.py`,
  the routes in `routes/plot_api.py`. Portal/plotter integration is a later
  effort. Numerics are not copied: they are imported from GEECS-Data-Utils.
- The server returns figure JSON; Plotly JS draws it in the browser, at a
  pinned version: copy the portal's `static/plotly-cartesian-3.1.1.min.js`
  (MIT) or load that build from an allowed CDN. The portal vendors because
  control-room machines may have no internet. Use its `plotly` Python pin.
- Every figure endpoint also returns `code`: the notebook snippet that
  reproduces the frame and the figure with the same functions.
- Data: s-files through GEECS-Data-Utils. Tiled is optional and you do not
  have it; nothing in M1–M5 may need it.
- Localhost first; deployment is a later PR. Kilohertz and other
  high-throughput systems are out of scope.

## The package

```
GeecsPlotter/
  pyproject.toml  README.md  CLAUDE.md  CHANGELOG.md
  geecs_plotter/
    __init__.py  __main__.py   # `geecs-plotter --port 8500` (argparse + uvicorn)
    app.py                     # FastAPI app factory: mount_theme, make_templates
    scans.py                   # find a scan folder, read the s-file (data-utils calls)
    figures.py                 # pure: columns in, plotly Figure out (copied from the portal)
    routes/  templates/  static/   # pages.py + api.py; Jinja2 pages; the Plotly bundle
  tests/                       # pytest; fixtures write a fake s-file in tmp_path
```

Model the files on `GeecsLogbook/` (same shape and deps). Run locally:
`cd GeecsPlotter && poetry install && poetry run geecs-plotter`, then open
`http://localhost:8500` (portal 8200, scanner 8300, logbook 8400 are taken).

A new package is registered in six places; the skeleton PR does all of
them: a row in root `CLAUDE.md`'s repository map, an entry in its
dependency graph and in its "Release & Versioning" changelog list;
`.importlinter` (`root_packages`, the top layer, and a `plotter-edges`
contract: data-utils and the theme only); `scripts/check.sh`
`OWN_ENV_PKGS` and `scripts/ci_select.py`; the templates and CSS in
`GeecsWebTheme/tests/test_no_literal_colours.py::_SURFACES`.

## Data access

Everything here is in GEECS-Data-Utils and needs pandas only, no Tiled.

- Config: `~/.config/geecs_python_api/config.ini` with
  `[Paths] GEECS_DATA_LOCAL_BASE_PATH = <data root>` and
  `[Experiment] expt = <experiment>`. `GeecsPathsConfig` reads it. The folder
  layout is in root `CLAUDE.md` § "GEECS Data Folder Convention"; the s-file
  the readers open is `{day}/analysis/sN.txt`.
- A scan: `ScanData.from_date(year=, month=, day=, number=, source="sfile")`
  → `sd.data_frame`, `sd.paths.get_folder()`, `sd.paths.load_scan_info()`.
- The frame the figure code expects: `scan_frame(scan_folder=folder).frame`
  (`geecs_data_utils.scan_frame`). With no Tiled run it is the s-file alone;
  a Tiled `detail` can be passed later without changing anything else.
- A day's scans without Tiled: `FolderScanCatalog`
  (`geecs_data_utils.folder_catalog`).
- Column list and coercion: `plottable_columns`, `numeric_series`
  (`geecs_data_utils.tiled_schema`). Never guess a column's meaning.
- Filters: `RowFilters`, `apply_filters` (`geecs_data_utils.data.row_filters`).
- Binning: `BinningConfig`, `bin_frame` (`geecs_data_utils.data.binning`);
  read `GEECS-Data-Utils/CLAUDE.md` § "Binning System".
- New math goes into GEECS-Data-Utils as a pure function with its own test,
  in a PR of its own, before the plotter uses it.

## Rules that bind

- `GeecsWebTheme/CLAUDE.md` first: the shell, status words, containers and
  the overlay ladder are decided; import `geecs_web_theme.web`, never copy it.
- Colours through tokens, never a literal; `body class="kit"` scopes the kit
  (`GeecsWebTheme/tests/test_no_literal_colours.py` walks your templates).
- Analysis code never creates a scan folder: read-only paths, no `mkdir`
  through `scans/ScanNNN/` (root `CLAUDE.md` § "Cross-package invariants").
- Facility values have one home: no experiment name, host, path or timezone
  in code; `config.ini` only (root `CLAUDE.md` § "Facility values").
- Pydantic v2 models for every request and response shape; no free dicts.
- NumPy docstrings, type hints on every public function, `ruff` decides
  formatting (root `CLAUDE.md` § "Code Style Conventions").
- No version bump or CHANGELOG entry in an ordinary PR (root `CLAUDE.md`
  § "Release & Versioning").

## Skills to run and when

- `/get-started` in the first session: environment check, orientation, a first win.
- `/env-doctor` when `poetry`, `pytest` or an import fails for setup
  reasons, or `config.ini` is missing.
- `/check` before every push: lint and the tests the way CI runs them.
- `/land` to open each PR: scope check, `scripts/commit.sh`, PR body,
  review, CI watch. Master merges are the maintainer's.

## Milestones

- **M1** — skeleton + first plot. Done when `geecs-plotter` serves a page
  that takes a day and scan number, lists the s-file's columns and plots one
  against another with Plotly; package registered; tests on a fake s-file pass in CI.
- **M2** — binned statistics. Done when the page shows the binned view:
  median centres, quartile error bars, bin by `Bin #` or by value interval,
  row filters (bounds, OR of ANDs) applied, and a table + CSV of the bins.
- **M3** — 2D and multi-Y. Done when several Y columns plot against one X
  (per shot and binned), and a heatmap view shows one column over two axes
  (or per-shot traces as a waterfall), all from one URL state.
- **M4** — copy to notebook. Done when every figure endpoint returns a
  `code` snippet that, pasted into a notebook, rebuilds the frame and the
  identical figure; one test executes the snippet.
- **M5** — an API an agent can drive. Done when the JSON routes are
  documented (FastAPI `/docs`, Pydantic models), take every view as query
  parameters, and a script fetches columns and both figures without the page.

## Size guidance

- First PR = the skeleton + M1, nothing else. Later PRs one milestone or
  less each; split M2 into filters and binning if either passes ~400 lines.
- When a PR bundles concerns, the body lists lines per concern. A new
  data-utils primitive is its own PR, landed first.
- Tests only where they pin behaviour: numbers computed by hand on a tiny
  frame, a template guard, one route test per endpoint.

## Out of scope

Tiled and the Bluesky side; images and per-bin averaged images; traces
beyond M3's heatmap; fitting; cross-scan merging; Origin export; logbook
posting; portal integration; deployment and systemd units; kHz data.

## The prompt to paste

```
Read Planning/geecs_plotter.md and start. I am the developer who wrote
GEECSplotter in LabVIEW; I know the tool and the data, not this repo or
Python packaging. Run /get-started first. Then build milestone M1 only:
the GeecsPlotter package skeleton, registered everywhere the brief lists,
and a page that opens one scan's s-file and plots one column against
another with Plotly. Follow the brief's decisions without reopening them.
Read GeecsWebTheme/CLAUDE.md before writing a template. Import numerics
from GEECS-Data-Utils; copy figure code from the portal where it helps.
Nothing may need Tiled. Use a worktree under .claude/worktrees/, run
/check before pushing, and use /land to open the PR against master with
a per-concern line count in the body. Do not merge. Ask me the "cf file"
and unknown-tabs questions from the table in the first session.
```
