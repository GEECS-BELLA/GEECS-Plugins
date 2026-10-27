# GeecsPlotter

GEECSplotter rebuilt as a small web app. The browser is the front panel:
pick a day and a scan, and the page plots one column of that scan's s-file
against another, with Plotly. The server reads the s-file (through
GEECS-Data-Utils), builds the figure, and sends it as JSON.

## Run it

```bash
cd GeecsPlotter
poetry install
poetry run geecs-plotter
```

The page opens in your default browser once the server is up
(<http://localhost:8600>); `--browser chrome` (or `safari`, `firefox`) picks
one, and `--browser none` opens nothing. The data root and experiment come from
`~/.config/geecs_python_api/config.ini`
(`[Paths] GEECS_DATA_LOCAL_BASE_PATH`, `[Experiment] expt`); pass
`--experiment` to read another experiment's folder.

Every control (X, Y, the filters and the toggles) is kept in a named
settings set, one `<name>.json` file each, saved as you pick, along with
the last scan opened from the page: starting the plotter, or switching to
a set, reopens that scan (a set is one file, so everyone using it shares
its last scan; a set with none keeps the scan you have open). The URL names
only what is open: `/?day=2026-10-02&scan=3&set=default`. Pick another set,
or "Save as" a new one (names may have spaces), in the rail's Settings
group. The sets live in `{[Paths] geecs_data}/GeecsPlotter/settings/` (the
GEECS `user data` folder, usually on the shared drive), else in
`user data/GeecsPlotter/settings/` in your main checkout (shared by all its
worktrees, and never committed) when that folder is not reachable;
`--settings-dir` picks another. The page shows which folder it is using.

Up to three Y columns plot at once: "Y plots 1 | 2 | 3" shows Y, Y2 and
Y3. Each has its own axis (Y on the left, Y2 and Y3 on the right), coloured
like its points and like the swatch beside its list. To replace a saved
set, pick its name in Save as and confirm.

X follows the scan's variable by default ("X = scan variable: auto"); a
hand-picked X turns that off. Column names show as their LabVIEW aliases
unless an axis's toggle says "name".

## Reproduce a plot in a notebook

Every figure carries a "Notebook code" snippet under it. Paste it into a
notebook in this package's environment to rebuild the same numbers and figure.

## Tests

```bash
cd GeecsPlotter
poetry run pytest tests
```
