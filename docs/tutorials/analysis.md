# Tutorial — Configure & Run Live Analysis

This tutorial walks the full GEECS-Plugins analysis loop end to end, using
only the GUIs. By the end you'll have:

1. Tuned a per-camera analyzer config in the **web config editor**, watching
   the result on a real shot
2. Added it to a group config that LiveWatch can dispatch
3. Run that group against a real scan with **LiveWatch**

It's the canonical workflow most users adopt for live shift analysis. No
Python required.

## Before you start

You should already have:

- **`~/.config/geecs_python_api/config.ini`** set up. Copy from a working
  teammate if needed — the full key-by-key reference (and how to create it
  from scratch) is in [Getting started](getting_started.md).
  Minimal contents look like:

    ```ini
    [Paths]
    geecs_data = Z:\path\to\experiment\user data
    scan_analysis_configs_path = Z:\path\to\GEECS-Plugins-Configs\scan_analysis_configs
    image_analysis_configs_path = Z:\path\to\GEECS-Plugins-Configs\image_analysis_configs

    [Experiment]
    expt = Undulator
    rep_rate_hz = 1
    ```

    LiveWatch resolves data paths from this file: `geecs_data` is the
    experiment data root it walks looking for new scan folders, and
    `scan_analysis_configs_path` is what its **Analyzer Group** dropdown
    discovers groups from.

- A **scan_analysis_configs/** directory checked out and writable. This
  usually lives in the sister `GEECS-Plugins-Configs` repo alongside this
  one.
- At least one **completed scan folder** under the data root for whatever
  date you want LiveWatch to process. If you don't have one handy, use a
  previous day's folder — LiveWatch can back-date freely.

If those three are in place, you're ready.

## 1. Author the analyzer config

The config editor is part of the data portal (the scan browser on the worker
host, port 8200). Open a recent scan of the camera you want to tune, switch
to its **Analysis** tab and click **edit** next to the diagnostic. The editor
opens in a drawer over the scan page, with a **preview** of that diagnostic
rendered on the drawer's device and shot.

There is also a standalone page — the portal's **edit configs** link, or on a
laptop with a clone of the configs repo:

```bash
poetry run scan-config-editor --configs /path/to/GEECS-Plugins-Configs/scan_analysis_configs
```

Either way the form is generated from the diagnostic schema, so every field
carries its description. The key sections of a diagnostic:

- **`name`**, `output_name`, `description` — which device folder is analyzed
  and what the outputs are called.
- **`analyzer`** — `kind` picks the analyzer (`beam`, `standard`, `magspec`,
  …) and the form swaps in that kind's parameters. There are no class paths
  and no free-form `kwargs`: every parameter is typed.
- **`image`** — the per-shot processing. `type: camera` is the 2D pipeline,
  `type: line` the 1D one. Each step (ROI, background, thresholding, …) is a
  section you tick on, **and** an entry in the ordered `pipeline` list — a
  step runs only if it is listed there.
- **`scan`** — how ScanAnalysis invokes it: priority, mode (`per_shot` vs
  `per_bin`), the Google Doc slot, renderer cosmetics.

Edit any field. The validator runs as you type; the YAML pane on the right
shows exactly what will be written, and the **preview** button (or **auto**)
re-renders the current shot through the unsaved document — dial an ROI in
here, not by saving and re-running.

A representative camera-analyzer YAML looks like:

```yaml
schema_version: 2
name: UC_TopView
analyzer:
  kind: beam
image:
  type: camera
  bit_depth: 16
  roi: {x_min: 0, x_max: 650, y_min: 350, y_max: 650}
  background: {method: constant, constant_level: 5.0}
  thresholding: {method: constant, value: 0.0, mode: to_zero}
  pipeline: [background, roi, thresholding]
scan:
  priority: 50
  mode: per_shot
```

**Save** writes the file into the configs tree on the share (the portal's
`--processing-configs` root) in canonical form. It is a normal uncommitted
change in that checkout — commit it when you are happy with it. The
Analysis tab picks the new configuration up immediately.

## 2. Add the analyzer to a group

Groups are the unit LiveWatch dispatches. A group is a named list of
analyzer refs, each optionally overridden per-group.

In the standalone editor page, pick a group under `groups/` (e.g.
`HTU/baseline.yaml`). The group form shows:

- **`name`** and **`description`** — the group's human-readable identity.
- **`upload_to_scan-log`** — when ticked, each member's display files go to
  the Google Doc e-log on completion.
- **`analyzers`** — the roster. Each entry is a diagnostic id (with
  type-ahead over every analyzer in the tree) and an optional per-group
  `priority` override; unknown ids are flagged before you can save.

Save the group when you're done.

A representative group YAML reads:

```yaml
name: HTU_baseline
description: standard HTU shift analysis
upload_to_scan-log: true
analyzers:
  - Amp4Input
  - Amp4Output
  - UC_TopView
  - {ref: GaiaMode, priority: 5}     # explicitly bumped vs the bare entry below
```

The bare-string form `UC_TopView` and the dict form `{ref: Foo, priority: N}`
both work; the editor preserves whichever you used.

## 3. Run the group with LiveWatch

Launch LiveWatch:

```bash
poetry run python ScanAnalysis/LiveWatchGUI/main.py
```

The main window fills in defaults from your `config.ini`:

![LiveWatch with HTU/baseline selected as the analyzer group, the scan
config dir auto-detected, and today's date](
assets/livewatch_02_group_selected.png)

Field-by-field for our purpose:

1. **Experiment (for Google Docs)** — leave at `Undulator` (or whichever
   experiment matches the group you're about to run; `(none)` to disable
   e-log upload entirely).
2. **Namespace** — `(all)` shows every group; pick a namespace to filter.
3. **Analyzer Group** — pick the group you just edited (e.g.
   `HTU/baseline`). The dropdown auto-populates from
   `scan_analysis_configs/groups/`.
4. **Date** — defaults to today. Back-date if you want to reprocess a
   previous day.
5. **Start Scan #** — `0` for "every scan from the start of the day,"
   otherwise the first scan number to consider.
6. **Enable GDoc Upload** — only when you actually want results in the
   e-log.

### Sanity-check with a dry run

Before letting it loose on real data, tick **Dry Run** in the Runtime
Options box and click **▶ Start**. The runner walks the day's scans and
reports what it *would* dispatch for each, without running anything. The
log panel shows you exactly which analyzers got matched to which scans.

Look at the output. If a scan that should be processed is being skipped,
the log tells you why (already completed, marked failed, no matching
device, etc.). When the dry run is clean, untick **Dry Run** and Start
again — this time for real.

### Watching it work

While the runner is alive, the status pip flips to `Running` (green) and
the log panel streams the runner's logs:

- Discovery: which scans match the date / start-number filter.
- Dispatch: which analyzers from the group are being kicked off for each
  scan.
- Completion: each analyzer's exit state and the display files it
  produced.

For per-task detail, click **Status…** to open the per-scan, per-analyzer
grid. Failures show their traceback inline.

When everything's been processed, the runner idles, watching for new
scans. Stop it with **⏹ Stop** (replaces Start while running).

## 4. Where the output lives

LiveWatch writes results into a sibling `analysis/` tree next to
`scans/`:

```
{geecs_data}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/
├── scans/
│   ├── Scan001/
│   ├── Scan002/
│   └── …
└── analysis/
    ├── Scan001/
    │   ├── UC_TopView/
    │   │   ├── summary_figure.png
    │   │   └── …
    │   └── …
    └── …
```

Inside each `analysis/ScanNNN/<analyzer>/` you'll find the analyzer's
display files (typically PNGs), any derived scalars, and a status YAML the
task queue uses to track completion. If GDoc upload was enabled, the same
display files will appear in the experiment's Google Doc.

## What to do next

You now have the full loop. The places to go from here:

- **Author a new analyzer from scratch.** In the drawer, **duplicate as**
  copies the open diagnostic under a new id for the previewed device; the
  standalone page's **new** starts from the schema defaults.
- **Build a custom group.** Same flow as Step 2, from **new** on the
  standalone page.
- **Inspect the underlying API.** Everything the editor and LiveWatch
  do is also available headlessly via Python — see
  [Image Analysis overview](../image_analysis/overview.md) for the
  per-image API and
  [Scan Analysis overview](../scan_analysis/overview.md) for the
  `LiveTaskRunner` that LiveWatch wraps.
- **Diagnose a recurring failure.** The `/triage` skill parses scan logs
  into a markdown summary that classifies errors by source — see
  [Skills](../skills/overview.md).
