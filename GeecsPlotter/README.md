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

Then open <http://localhost:8600>. The data root and experiment come from
`~/.config/geecs_python_api/config.ini`
(`[Paths] GEECS_DATA_LOCAL_BASE_PATH`, `[Experiment] expt`); pass
`--experiment` to read another experiment's folder.

A plot's URL is the plot, filters and toggles included; bookmark it to
save a setup. X follows the scan's variable by default ("X = scan
variable: auto"); a link to a hand-picked X carries `xauto=false`:
`/?day=2026-10-02&scan=3&xauto=false&x=<column>&y=<column>`, with `x=`
empty for the shot number. Column names show as their LabVIEW aliases
unless an axis's toggle says "name" (`xalias=false` / `yalias=false`).

## Reproduce a plot in a notebook

Every figure carries a "Notebook code" snippet under it. Paste it into a
notebook in this package's environment to rebuild the same numbers and figure.

## Tests

```bash
cd GeecsPlotter
poetry run pytest tests
```
