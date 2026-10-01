# Data Portal

The Data Portal shows what a scan recorded, in any browser on the lab
network: pick a day, pick a scan, then look at its metadata, plot its
scalars, map a 2D scan, page through camera images and traces, and run a
saved analysis. It never changes the data. The only thing it writes is
analysis output, and only when you press **Run** on the Analysis tab.

It runs on the worker host on port **8200**
(`http://<worker-host>:8200/`), beside the [scanner](scanner.md) (8300)
and the [logbook](logbook.md) (8400).

## Open a day

The front page lists days. Open one to get that day's scans in a table:
number, mode (no-scan, 1D, grid), shot count, status, description and
save set. The **filter** box narrows the table by any of those words.
The arrows either side of the date step to the previous and next day.

Click a scan to open its run page.

## Find your way around a scan

The left rail of the run page says where you are (experiment, day, scan
number, mode, shot count, status, start time) and how to move:

- **‹ day / day ›** go to the same scan number on the neighbouring day
  (or its latest scan); the date itself goes back to the day list.
- **‹ scan / scan ›** and the scan dropdown step through the day.
- **log ↗** opens this scan's entry in the [logbook](logbook.md), when
  the portal knows where the logbook is.
- **Filters** hold conditions on the scalar columns (AND inside a group,
  groups OR together), edited with **edit filters…**. They apply to the
  plots, the grid maps and the images, and they follow you from scan to
  scan and day to day, as do the columns you picked.

Everything you choose (tab, columns, filters, bins, the selected cell or
shot) is in the page's URL, so a link you copy opens the same view.

## The tabs

- **Overview** — the scan's metadata and settings as recorded.
- **Plot** — any scalar against the shot number or another scalar, per
  shot or **binned** by the scan variable. **show the code** gives the
  notebook cell that reproduces the figure from Python. The toolbar's
  **send to scan log** posts the figure into this scan's logbook entry.
- **Grid** — for a two-variable scan, a map of any scalar over the two
  scan axes, with the average (mean or median) and the error (standard
  error of the mean, or others) chosen separately. Click a cell for its
  readbacks, sample count and member shots, or open that cell's images.
  It also has **show the code** and **send to scan log**.
- **Images** — one device's shots, one at a time or averaged per bin. A
  camera shows its image; a spectrometer or scope shows its trace as an
  interactive plot. The **processing** selector runs a saved recipe's
  processing on the shown image as a preview, without writing anything.
- **Analysis** — shown when the portal has an analysis configuration.
  See below.

## Run an analysis on a scan

The Analysis tab lists every saved analysis recipe that applies to this
scan (its device has data in the scan); the rest are folded away.

1. Press **run** on a recipe. The badge goes *queued* → *running* →
   *done* (or *failed*, with the error and the run's log). Only one run
   per scan at a time.
2. The results appear in the tab: summary figures inline, per-bin
   figures behind a stepper, other files as links. They are saved under
   the day's `analysis/ScanNNN/` folder, and the scalars the recipe
   computes are added to the scan's `analysis/sNNN.txt`, so they show up
   as new columns on the Plot and Grid tabs.
3. Re-running replaces those outputs; the raw data is never touched.

A missing scan folder is reported, never created.

## Edit a recipe

Each recipe on the Analysis tab has an **edit** button. It opens the
config editor in a drawer over the page, with this scan's data as the
preview input.

- Change the steps (background, crop, filters, masks) and the measure,
  then press **preview** (or turn on **auto**) to see the result on a
  shot. Preview saves nothing.
- **Save** writes the recipe to the configs repository. If someone else
  saved it since you opened it, the editor says so rather than
  overwriting their change.
- **duplicate as** copies the recipe under a new name, for a second
  analysis of the same device.

What each step and measure does, what every scalar it writes means, and
worked examples are in the
[analysis recipe reference](../sites/analysis_recipes/index.html). The
[analysis tutorial](../tutorials/analysis.md) walks the whole path from
a recorded shot to s-file columns, including how to do the same from
Python without the portal.

## When something is wrong

- **No Analysis tab** — the portal was started without an analysis
  configuration, or its analysis extra is not installed. That is a
  deployment question; see the [fleet map](../platform/fleet_map.md).
- **An image says "no frame" for a shot** — that device did not record
  that shot. The portal never shows a neighbouring shot in its place.
- **A run fails** — the tab shows the error and the run's captured log.
  Fix the recipe with **edit**, preview it on a shot, and run again.
