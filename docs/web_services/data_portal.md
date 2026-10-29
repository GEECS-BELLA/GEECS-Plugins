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
The open tab is in the page's URL, so a link you copy opens the same tab.
The Plot and Images tabs keep your choices in your settings set instead.

## The tabs

- **Overview** — the scan's metadata and settings as recorded.
- **Plot** — its own page, GEECSplotter's: up to three scalars against
  the shot number or another scalar, per shot or **binned**, with shot
  filters, fits and display settings, all kept in your named settings
  set. The page gives the notebook cell that reproduces the figure from
  Python, and its toolbar can copy the plot or **send** it into this
  scan's logbook entry. Scans from another experiment than the portal's
  own have no Plot tab.

  Its **2-D map** view (the switch above the plot) maps any scalar over
  two scan variables: each cell shows the scalar's average (mean or
  median) and, beside it, its error (standard error of the mean, or
  others), chosen separately. X starts as the scan variable; pick the
  Y axis in the rail. Click a cell for its readbacks, sample count and
  member shots, or open that cell's images on the Images tab.
- **Images** — its own page beside Plot, reading the same settings set:
  one camera's strip of tiles, one per bin (its shots' average) or one
  per shot, with the colour map, levels and region of interest you
  chose. Click a tile to see it full size underneath, and step with
  ‹ ›. On a single shot, the **processing** choice runs a saved recipe's
  processing as a preview, without writing anything. A spectrometer or
  scope has no strip: it shows one shot's trace at a time, as an
  interactive plot. A 2-D map cell's **View this cell's images** opens
  this page on that cell's bin.
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
   as new columns on the Plot page.
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
