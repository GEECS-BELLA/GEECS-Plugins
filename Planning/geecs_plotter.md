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

## What GEECSplotter does today

The repo digested the tool in PR #710 (panel notes) and PR #717 (feature
inventory, then the owner's "Post-mockup rulings"). Both files were deleted
later; read their final versions with
`git show 9d668b33a^:Planning/data_portal/labview_peruser/notes.md` and
`git show 9d668b33a^:Planning/data_portal/02_labview_peruser_inventory.md`
(`03_analysis_tabs_design.md` beside them is the portal's design of record).
The table reports what is recorded there.

| GEECSplotter feature | Recorded | In this app |
|---|---|---|
| Folder tree → year/month/day/scan → s-file | adapt: a landing page with day/scan entry; steppers in the rail | a day + scan-number form; no Tiled |
| "merge s files" | defer | later (cross-scan) |
| Listbox par 1–4 | adapt | one X picker + up to 4 Y columns |
| "autofill par1" (scan variable as X); "use aliases" | keep / defer | default X = the scan variable; raw column names first |
| Substring name filter on columns | keep | a filter box over the column list |
| Save/load selections and filter sets | adapt | the selection lives in the URL |
| Data filters: per-column bounds, include/exclude, outer OR / inner AND | keep | `RowFilters` in GEECS-Data-Utils |
| "filter max L2 each bin" (best N shots per bin) | keep, but evicted to a later filters popup | not M1; the primitive goes into data-utils first |
| FileType to Analyze (2 Parameter Plot, traces, …) | adapt | scalar plots only; traces later |
| User field, color table, Save png, copy tdms, loop counter, clipboard; "pretty PDF" | drop; defer | Plotly's own PNG export |
| Scan-info text | keep | a metadata table on the page |
| xy plots: pars 2–4 vs par 1, per shot; zoom, cursor, autoscale, same-scale | keep (2-Parameter mode = top priority); one Plot tab, per-shot ⇄ binned toggle | M1, then M2; Plotly built-ins; same-scale = linked axes |
| Sub-tabs Fit · L2L3 · All3 · histogram | adapt | any column as X covers L2L3; histogram = a trace type; fit deferred |
| "cf file" sub-tab | drop (owner, 2026-10-26) | not built |
| Binned plot: bin by `Bin #`, median centre, quartile bars | keep (flagship) | M2; `BinningConfig(agg="median", err="iqr")` |
| Fixed/auto bin interval; use-bin#, same-scale, subtract-mean | keep / adapt | `bin_width` / `bin_edges`; toggles behind a settings popup; subtract-mean = a data-utils transform |
| Fitting, "gauss prop", peak/valley | defer | not in M1–M5 |
| triple/double/single/errors/**data** sub-tabs; export to Origin / txt | adapt | multi-Y; the data sub-tab = a table + CSV + the notebook snippet |
| Overview: per-bin averaged images (xcorr align); Intensity: trace waterfall | keep (later), xcorr defer; waterfall rastered, colour = intensity | images out of scope here; M3's heatmap is the trace step |
| Setup · 3D graph · stats · XY Graphs · MultiScan · Analysis tabs | owner, 2026-10-26: Setup, 3D graph, stats, XY Graphs, MultiScan used; Analysis not | the used five need a milestone each after M5; Analysis dropped |

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
- The portal's Plot tab already does M1–M5 over the same s-files, Tiled or
  not (portal 0.39.0 lists scan folders itself). The maintainer still wants
  a separate package; run the portal once to see the target, then copy.
- Duplication is allowed inside the package. Copy what helps from the
  portal: `GEECS-DataPortal/geecs_portal/figures.py` (`shots_figure`,
  `binned_figure`, `Palette`, `page_figure`), the parsing in `analysis.py`,
  the routes in `routes/plot_api.py`; name the copy as accepted debt in
  `GeecsPlotter/CLAUDE.md`. Numerics are not copied: they are imported from
  GEECS-Data-Utils. Portal/plotter integration is a later effort.
- The server returns figure JSON; Plotly JS draws it in the browser. Vendor
  the portal's `static/plotly-cartesian-3.1.1.min.js` (MIT) and use its
  `plotly` Python pin. A pinned CDN build is allowed by the maintainer, but no
  repo policy names a CDN and control-room machines may lack internet.
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
    __init__.py  __main__.py   # `geecs-plotter --port 8600` (argparse + uvicorn)
    app.py                     # FastAPI app factory: mount_theme, make_templates
    scans.py                   # find a scan folder, read the s-file (data-utils calls)
    figures.py                 # pure: columns in, plotly Figure out (copied from the portal)
    routes/  templates/  static/   # pages.py + api.py; Jinja2 pages; the Plotly bundle
  tests/                       # pytest; fake s-file as in GEECS-Data-Utils/tests/test_scan_frame.py
```

Model the files on `GeecsLogbook/` (same shape; deps per the list above).
Run locally: `cd GeecsPlotter && poetry install && poetry run geecs-plotter`,
then open `http://localhost:8600` (8200–8400 are live; `Planning/device_panels.md` claims 8500).

A new package is registered in these places; the skeleton PR does all of
them: a row in root `CLAUDE.md`'s repository map, an entry in its
dependency graph and in its "Release & Versioning" changelog list;
`.importlinter` (`root_packages`, the top layer, the contracts that list
every package, and a `plotter-edges` contract: data-utils and the theme
only); `scripts/check.sh` `OWN_ENV_PKGS` and `scripts/ci_select.py`; in
`GeecsWebTheme/tests/test_no_literal_colours.py`, the templates and CSS in
`_SURFACES` and the copied `figures.py` beside the portal's in the token check.

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
- A day's scans, no Tiled: `FolderScanCatalog` (`geecs_data_utils.folder_catalog`).
- Columns and coercion: `plottable_columns`, `numeric_series` (`geecs_data_utils.tiled_schema`).
- Filters: `RowFilters`, `apply_filters` (`geecs_data_utils.data.row_filters`).
- Binning: `BinningConfig`, `bin_frame` (`geecs_data_utils.data.binning`);
  read `GEECS-Data-Utils/CLAUDE.md` § "Binning System".
- New math is a pure, tested function in GEECS-Data-Utils, in its own PR first.

## Rules that bind

- `GeecsWebTheme/CLAUDE.md` first: the shell, status words, containers and
  the overlay ladder are decided; import `geecs_web_theme.web`, never copy it.
- Colours through tokens, never a literal; `body class="kit"` scopes the kit
  (`GeecsWebTheme/tests/test_no_literal_colours.py` walks your templates).
- Analysis code never creates a scan folder: read-only paths, no `mkdir`
  through `scans/ScanNNN/` (root `CLAUDE.md` § "Cross-package invariants").
- Facility values have one home: no experiment name, host, path or timezone
  in code; `config.ini` only (root `CLAUDE.md` § "Facility values").
- Pydantic v2 models for every request and response shape; the Plotly figure
  dict inside a response is the one free dict.
- NumPy docstrings, type hints on every public function, `ruff` decides
  formatting (root `CLAUDE.md` § "Code Style Conventions").
- No version bump or CHANGELOG entry in an ordinary PR (root `CLAUDE.md`
  § "Release & Versioning").

## Skills to run and when

- `/get-started` in the first session: environment check, orientation, a first win.
- `/env-doctor` when `poetry`, `pytest` or an import fails for setup reasons.
- `/check` before every push: lint and the tests the way CI runs them.
- `/land` to open each PR: scope check, `scripts/commit.sh`, body, review, CI watch.

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
- Tests pin behaviour: hand-computed numbers on a tiny frame, one route test per endpoint.

## Out of scope

Tiled and the Bluesky side; images and per-bin averaged images; traces
beyond M3's heatmap; fitting; cross-scan merging; Origin export; logbook
posting; portal integration; deployment, systemd units and the docs page; kHz data.

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
