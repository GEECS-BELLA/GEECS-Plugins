# GEECS Data Portal

A read-only web view of GEECS scan data: pick a day, pick a scan, see its
metadata and scalar plots — from any browser on the lab network, nothing
to install.  Built on the `ScanCatalog` layer in GEECS-Data-Utils
(`geecs_data_utils.tiled_catalog`).

```bash
poetry install
poetry run geecs-data-portal --experiment Undulator   # serves on :8200
```

Requires the `[tiled]` section of `~/.config/geecs_python_api/config.ini`
(the same file every GEECS-Plugins package reads).  Design and arc plan:
`Planning/data_portal/01_data_portal_scope.md`; developer rules:
`CLAUDE.md` here.

## Two-dimensional scans

Open **Grid**, choose a scalar and two scan axes, then choose the average
(mean or median) and error statistic independently. The homepage shot filters
apply to both maps. Missing, fully filtered and unacquired cells remain visible;
errors below the minimum sample count are withheld while averages remain visible.

Each plotting area is square. Axis spacing can be linear, logarithmic, or equal
cells labelled with the original grid coordinates. The latter is available only
for recorded rectangular grids. Spiral/together trajectories show measured points
without interpolation. Repeated visits have a separate selector.

Select a cell to inspect readbacks, sample counts and member shots, or open its
bin images with the same filters. The URL preserves these choices; **show the
code** reproduces the calculation and figures in a notebook. SEM is always the
standard error of the mean, even when the displayed average is a median.

This release supports two-axis scans; higher-dimensional slicing is deferred.
ScanInfo and ScanAnalysis's first-motor labels are unchanged.

Each Grid map has the same **send to scan log** toolbar button as Plot. It sends
the selected map with its statistic caption and a link back to the filtered view;
you can append both maps to the same log entry.
