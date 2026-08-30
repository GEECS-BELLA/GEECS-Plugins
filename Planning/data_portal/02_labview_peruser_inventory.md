# GEECSplotter feature inventory — keep / adapt / drop

*The decision surface for the analysis-tabs arc, per the
cutover-strategy precedent: every GEECSplotter capability from
`labview_peruser/notes.md` (the digested spec source), each with a
proposed disposition **for the owner to rule on**. Proposals are
informed by the 2026-08-29/30 vision session; the settled architecture
below is decided, the per-feature rulings are not until marked.*

*Legend — **keep**: port the capability as-is (behavior, not surface);
**adapt**: the need is real, the mechanism changes; **drop**: not
ported (rationale given; reversible later); **defer**: real, but not in
the first analysis-tab wave. Owner column: fill with ✓ / a ruling /
notes.*

## Settled architecture (decided 2026-08-29/30 — not up for per-row relitigating)

1. **Surface = a scan.** Identity is experiment + day + number (+ uid
   when cataloged). Column **providers** attach to it: the Tiled event
   table and the s-file, joined by shot number as a **union with
   provenance** — both name-spaces kept verbatim, each column badged by
   source, near-duplicates accepted, no name reconciliation ever.
2. **Numerics live in GEECS-Data-Utils** (and ImageAnalysis where
   image-shaped). Portal analysis endpoints are JSON one-liners over
   public functions; anything the web tab computes is reproducible
   verbatim in a notebook (`BinningConfig` already covers
   median-center + quartile-band binning).
3. **Interactivity = vendored Plotly.js** (one checked-in,
   version-pinned `.min.js`; still no npm, no CDN — the doctrine
   amendment lands in the portal CLAUDE.md with the first plotly PR).
   Server sends data, browser draws: zoom, pan, log axes, autoscale,
   hover readout, PNG export come free.
4. **Single-scan analysis surface first.** Cross-scan = a later tab
   over a data-utils concat-with-scan-column util, same primitives.
5. **Historic MC data enters via an INGESTER** (one-shot batch writing
   legacy scans into Tiled with a provenance marker), not via a
   permanent s-file-only browsing surface. Owner: near-required.
   Separate work item; rows below that depend on legacy data note it.

## 01 — Main panel

| # | Feature (GEECSplotter) | Proposal | Port shape | Owner |
|---|---|---|---|---|
| M1 | Data-folder root → Year/Month/Date/Scan tree → s-file picker | **drop** (superseded) | The portal's day → run navigation over the catalog already is this, minus the file-picking indirection. Legacy days arrive via the ingester (§5 above). | |
| M2 | "merge s files" | **defer** | This is cross-scan analysis (settled §4): a concat util in data-utils + a later multi-scan tab. Not wave 1. | |
| M3 | Listbox par 1–4 (independent variable + up to 3 dependents) | **adapt** | Becomes the analysis tab's X picker + multi-Y picker (no fixed count of 4 — a chip list). Pick list = `plottable_columns` over the union-with-provenance frame, columns badged by source. | |
| M4 | "autofill par1" (scan variable auto-selected as X) | **keep** | Already portal behavior (stepped-scan default X, console parity). Extends unchanged to the analysis tab. | |
| M5 | "use aliases" (friendly column names) | **defer** | The event schema's `geecs_scalar_headers` prettification exists in `tiled_schema`; wire it as a display-name layer once the tab works with raw names. Never a join key (union-with-provenance forbids reconciliation). | |
| M6 | Per-listbox substring name filter (e.g. `UC_amp`) | **keep** | A filter box over the column picker — cheap, high daily value with hundreds of columns. Client-side. | |
| M7 | Save/load listbox selections | **adapt** | Becomes shareable state: the analysis tab's full selection lives in the URL query (the portal's sticky-query pattern) — a bookmark IS a saved selection. Named server-side presets only if URLs prove insufficient. | |
| M8 | Data filters: per-column numeric boundaries, Value 1/Value 2, include/exclude, per-row enable, **outer OR / inner AND** | **keep** | THE filter model, ported semantically: a data-utils `FilterSpec` (list of AND-groups, OR of groups) → boolean mask over the frame. Note: `scans_database/filter_models.py` already has a FilterSpec vocabulary — reuse/extend it, don't mint a second one. | |
| M9 | "filter max L2 each bin" + top-x-per-bin + asc/desc | **keep** | Best-N-shots-per-bin selection — a data-utils primitive (`top_n_per_bin(frame, bin_col, value_col, n, descending)`) composed after M8's mask. Flagship-adjacent (feeds the binned plot). | |
| M10 | Filter sets save/load/append | **adapt** | Same answer as M7: filters serialize into the URL; named presets deferred until wanted. | |
| M11 | FileType to Analyze (None · 2 Parameter Plot · Optical Spectrum · Scope Trace · MagSpecLineout ·-Angle · Spreadsheet · g1/g2/g3 lineouts) | **adapt** | The mode toggle dissolves: scalar analysis is the 2-Parameter tab; trace types become the trace tab (§Intensity); per-type file loaders become data-utils readers as each trace type is ported. The owner's "could this become the s-file ↔ Tiled toggle" question is answered by settled §1: no toggle — union with provenance. | |
| M12 | User field, color table, "tab to switch to", Save png, copy tdms, loop counter | **drop** (mostly) | User field: no sessions in a read-only portal. Color table: Plotly template + per-plot colormap picker where it matters (overview images). Save png: Plotly's export button. copy tdms / loop counter: no equivalent need identified — say if wrong. | |
| M13 | Scan-info text display | **keep** (done) | The run page's metadata table already shows it. | |

## 03 — xy plots tab (2 Parameter Plot — **top priority**)

| # | Feature | Proposal | Port shape | Owner |
|---|---|---|---|---|
| X1 | Pars 2–4 vs par 1, unbinned per-shot scatter | **keep** | THE first analysis tab: multi-Y scatter vs X over the filtered frame, Plotly. Data endpoint = filtered frame → JSON series. | |
| X2 | Zoom, cursor readout, reset bounds, autoscale/same-scale, per-axis locks, plot width | **keep** (free) | Plotly built-ins; "both same scale" = a linked-axes toggle. | |
| X3 | Sub-tabs Fit · par L2 · L2L3 · All3 · histogram | **adapt** | L2L3 (Y-vs-Y cross-plot) = X picker accepts any column, so it falls out of X1. Histogram = a second trace type on the same tab. Fit: see B4 (one fitting story, not per-tab). All3/par L2 layout variants dissolve into the multi-Y chip list. | |
| X4 | cf file sub-tab (compare against a reference file?) | **unknown → owner** | Not legible from the slides/notes. What does it do, and is it used? | |
| X5 | Clipboard buttons | **drop** | Plotly PNG export + the browser covers it. | |
| X6 | "pretty PDF" export | **defer** | Publication-grade export is real but not wave 1; Plotly's SVG/PNG export bridges. Revisit with B6 (Origin export). | |

## 04 — Binned plot tab (**the flagship**)

| # | Feature | Proposal | Port shape | Owner |
|---|---|---|---|---|
| B1 | Bin by `Bin #`, quartile error bars, believed-median centers | **keep** | `ScanData.bin` / `BinningConfig` already does `agg="median"`, `err="percentile"`, `(0.25, 0.75)` — near 1:1. **Pin against the LabVIEW source before first release** (owner shares source out-of-band): exact center statistic + exact quartile definition. | |
| B2 | Fixed/auto bin interval (value-binning, not `Bin #`) | **keep** | `BinningConfig` extension: bin by X-value intervals when `Bin #` is absent/unwanted (noscan data, telemetry X). | |
| B3 | use-bin#, same-scale, subtract-mean, axis format/precision, autoscale modes | **adapt** | use-bin# = B1/B2 selector. same-scale = linked axes. subtract-mean = a per-series normalize toggle (data-utils transform, not a plot hack). Axis format/precision = Plotly defaults; add controls only on demand. | |
| B4 | Fitting panel + "xy analysis: gauss prop" + peak/valley analysis | **defer** (wave 2) | One fitting story for all tabs: a data-utils `fit` module (gaussian first — pin "gauss prop" against source), applied to binned or raw series, params rendered beside the plot. Deliberately after the plotting tabs stand. | |
| B5 | triple/double/single/errors/**data** sub-tabs | **adapt** | Layout variants dissolve into multi-Y. The **data** sub-tab (the numbers behind the plot) survives as a table view + CSV download of the binned frame — cheap and honest. | |
| B6 | Export panel: send to Origin (graph/worksheet templates, axis names/units), "pretty PDF or export txt" | **adapt** | Origin templates were personal-drive artifacts (redacted in the screenshots — effectively unportable). Port the *need*: download the binned data as CSV (B5) + "show the code" snippet so any plotting environment reproduces it. Direct Origin integration: drop unless someone actually asks. | |

## 05 — overview tab (binned image browser)

| # | Feature | Proposal | Port shape | Owner |
|---|---|---|---|---|
| O1 | Per-bin AVERAGED images (post-filter), grid/strip with bin captions | **keep** (wave 2) | Server-side: bin the shot list (M8/M9 filters apply), average each bin's frames, render a grid. Cheap for stack devices (the prefetch cache already holds whole frames arrays); native devices pay a one-time read. Numerics = ImageAnalysis/data-utils averaging, not portal code. | |
| O2 | xcorr alignment within a bin ("slow") | **defer** | Real analysis (ImageAnalysis territory), explicitly slow even in LabVIEW. Wave 3+, if wanted. | |
| O3 | Grid save, caption/size/font controls, browser size, frame style | **adapt** | Minimal: captions on, PNG export, sensible defaults. The long tail of layout knobs drops until missed. | |
| O4 | "Image to display" selector / overview source (g1…) | **unknown → owner** | Which image products can feed the overview besides the raw camera (g1/g2/g3 lineouts? analysis outputs?) — and which matter? | |

## Intensity tab (not screenshotted — owner-verbal)

| # | Feature | Proposal | Port shape | Owner |
|---|---|---|---|---|
| I1 | Waterfall stacking of per-shot y-vs-x traces (magspec lineouts, scope traces) | **keep** (wave 2, second priority per owner) | The trace tab: per-shot 1D arrays (from files or stacks) → Plotly heatmap/waterfall vs shot or bin. Loaders = data-utils readers per trace type (M11). Needs: which file formats first — magspec lineout + scope trace per the owner's priority. | |

## Tabs with no spec material (Setup · 3D graph · stats · XY Graphs · MultiScan · Analysis · Program info)

| # | Feature | Proposal | Owner |
|---|---|---|---|
| U1 | All seven un-screenshotted tabs | **unknown → owner** | One line each from the owner: used / not used / what it does. MultiScan is presumptively M2/§4 (cross-scan, deferred). Anything unused: drop. | |

## Proposed wave plan (falls out of the rulings)

- **Wave 1** — the 2 Parameter Plot tab: union-with-provenance frame
  (Tiled + s-file providers in data-utils), FilterSpec (M8) +
  top-N-per-bin (M9), multi-Y scatter + histogram over vendored Plotly
  (X1–X3), column filter (M6), URL-state selections (M7/M10),
  "show the code" snippet.
- **Wave 2** — the Binned plot tab (B1–B3, B5, CSV/data view) + the
  overview per-bin images (O1) + the trace waterfall (I1).
- **Wave 3** — fitting (B4), cross-scan (M2), and whatever U1 surfaces.
- **Parallel track** — the historic-MC ingester (settled §5), sized
  separately.

## To pin against the LabVIEW source (owner shares out-of-band, never committed)

1. Binned-point center statistic (median?) and the exact quartile
   error-bar definition (B1).
2. The "gauss prop" fit (B4).
3. The `cf file` sub-tab's meaning (X4).
4. Overview "source"/"image to display" options (O4).
