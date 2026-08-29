# GEECSplotter ("HPD mode") — the LabVIEW online data peruser

*Source: owner's slide deck, 2026-08-29 (screenshots `01`–`05` in this
folder; description text transcribed and lightly cleaned below).  This
is the raw spec material for the analysis-tabs arc — the formal
feature inventory with keep/adapt/drop dispositions is the next
document (`02_labview_peruser_inventory.md`, to be drafted).  The tool
itself: a single LabVIEW executable grown organically by one developer
from operator requests; no documentation exists; not all options are
understood even by daily users.*

## Owner framing (the goal — verbatim intent)

> The goal at this point isn't an identical surface to the LabVIEW
> one.  We should start by discussing how to build something with this
> level of access into the data for rapid inspection, built scalably
> to add new analysis tabs with new features — and, as much as
> possible, the actual core analysis pushed down into the other
> projects (data-utils, image-analysis, …).

Priorities (owner): **"2 Parameter Plot" is the top priority by far**;
"MagSpec lineout" and "Scope Trace" next; the other file types are not
first-order concerns.

## 01 — Main panel

- **Data selection**: data-folder root (e.g. `Z:\data\Undulator`; other
  experiments like Thomson likewise) → Year / Month / Date / Scan tree →
  an **s-file** (`analysis/sN.txt`).  A "merge s files" button exists.
  (Portal equivalent: the day/experiment picker + run list — already
  built, catalog-backed.)
- **Listbox par 1–4**: column selectors over the s-file. Par 1 ≈ the
  independent variable, usually auto-populated with the scan variable
  ("autofill par1" checkbox; "use aliases" checkbox).  A per-listbox
  **name filter** box narrows columns by substring (e.g. `UC_amp`).
  Save/load listbox selections.
- **Data filters** (the big panel, bottom right): per-column numeric
  boundary filters — Value 1 / Value 2, include/exclude ("Exclusive"),
  per-row enable; **outer indexes OR together, inner AND**.  Sets the
  data used when "Analyze" is clicked.  Additional selection tooling:
  "filter max L2 each bin" with a "top x each bin for filtering" count
  and ascending/descending selector (best-N-shots-per-bin selection).
  Filter sets can be saved/loaded/appended.
- **FileType to Analyze** (options exploded in `02`): None ·
  **2 Parameter Plot** · Optical Spectrum · Scope Trace ·
  MagSpecLineout · MagSpecLineoutAngle · Spreadsheet · g1/g2/g3
  lineouts.  Owner note: this toggle could conceivably become the
  s-file ↔ Tiled source toggle in a port — needs discussion.
- Misc: user field, color-table choice, "tab to switch to" after
  analyze, Save png, copy tdms, scan-info text display, loop counter.
- Tab strip across the top: Main · Setup · 3D graph · xy plots ·
  Binned plot · Intensity plot · stats · XY Graphs · MultiScan ·
  overview · Analysis · Program info.

## 03 — "xy plots" tab (2 Parameter Plot output, per-shot)

Every data point of pars 2–4 plotted vs the independent variable
(par 1), unbinned.  Sub-tabs: Fit · par L2 · L2L3 · All3 · cf file ·
histogram.  Interactive: zoom, cursor readout, reset xy bounds (not
visible in the slide — from the owner's description),
"autoscale/both same scale", per-axis lock toggles, plot-width control,
Clipboard buttons, "pretty PDF" export.  (Portal equivalent today: the
run view's scalar plot — single-line, no zoom; multi-line overlay and
the L2-vs-L3 cross-plot are the gaps.)

## 04 — "Binned plot" tab (the flagship)

Pars 2/3/4 binned by the **`Bin #` column** with **quartile error
bars** ("Error bar type xy: Quarter Percentiles"; owner believes the
bin's center point is the **median** — verify against source).
Controls: fixed/auto bin interval, use-bin#, same-scale, subtract-mean,
axis format/precision, autoscale modes.  Sub-tabs: triple · double ·
single · errors · data (the numbers behind the plot).  The right-hand
panel has **Export** · **Fitting** · **peak/valley** sub-tabs; a
separate "xy analysis: gauss prop" + analyze control group sits at the
tab's bottom-left.  The Export sub-tab (send binned data to Origin with
graph/worksheet templates, axis names/units; "pretty PDF or export
txt").  This is the workflow the portal's analysis tab must nail
first.
- **Pin against source**: the exact binned-point statistic (median?)
  and the quartile error-bar definition; the "gauss prop" fit.

## 05 — "overview" tab (binned image browser)

Displays **image artifacts averaged over the bins** (after the main
panel's filters are applied): a grid/strip of per-bin averaged images
with bin-value captions, stitched or individual, xcorr alignment
option ("align images in a bin Xcorr (slow)"), grid save, caption/size
controls, image-to-display selector ("binned image"), overview source
(g1…).  (Portal equivalent today: the per-shot gallery; per-bin
*averaged* images are new — note the capture-stack HDF5 makes bin
averaging cheap server-side.)

## Not shown in slides

- **Intensity tab**: waterfall stacking of per-shot y-vs-x traces
  (magspec lineouts etc.) — the "Scope Trace"/"MagSpecLineout" file
  types land here.  This is the trace-analysis tab of a port.

## Porting notes (session digest — proposals, not decisions)

- The portal already covers: data navigation (catalog beats the folder
  tree), per-shot xy plot (single-line), per-shot images.  The three
  big adds, in priority order: (1) **binned plot with quartile error
  bars + Bin # semantics**, (2) column pickers/name filters +
  numeric-boundary data filters (OR-of-ANDs), (3) multi-line overlay /
  cross-plots; then per-bin averaged images; then traces/waterfall.
- Core numerics (binning statistic, quartile bars, gauss fit) go DOWN
  into `geecs_data_utils` (binning already lives there — reconcile,
  don't duplicate) per the owner's explicit instruction; tabs stay
  thin views.  The Correlations-integration design (library primitives
  in data-utils) is the sibling precedent.
- The s-file is the tool's substrate; the portal's substrate is the
  catalog (same rows, via `RunDetail.data`).  A source toggle is
  likely unnecessary — the catalog's event table already carries the
  s-file columns (`Bin #` ↔ `bin_number`).
- LabVIEW source: to be shared out-of-band (NOT committed — public
  repo); needed only to pin the 2–3 load-bearing numerics above.
