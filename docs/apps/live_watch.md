# LiveWatch

**Automated per-scan analysis runner.**

LiveWatch watches a data directory for new scans and dispatches a configured
analyzer group as each scan completes. It's the canonical way to run
analysis alongside live data taking: configure once at the start of a shift,
hit Start, and every subsequent scan produces summary figures and (optionally)
e-log uploads without anyone touching a notebook.

Under the hood it's a `LiveTaskRunner` driven by a heartbeat-based task queue
— multiple LiveWatch instances can co-operate over the same data directory
without double-processing any scan.

## Prerequisites

LiveWatch resolves data paths from your shared **`config.ini`** (the same
file the Scanner GUI and the Python API read). It must exist before launch:

```
~/.config/geecs_python_api/config.ini
```

Minimal contents:

```ini
[Paths]
geecs_data = Z:\path\to\experiment\user data
scan_analysis_configs_path = Z:\path\to\GEECS-Plugins-Configs\scan_analysis_configs
image_analysis_configs_path = Z:\path\to\GEECS-Plugins-Configs\image_analysis_configs

[Experiment]
expt = Undulator
rep_rate_hz = 1
```

`geecs_data` is the experiment data root — LiveWatch walks
`{geecs_data}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/scans/` looking for
new scan folders. `scan_analysis_configs_path` is what the **Analyzer Group**
dropdown discovers groups from. The Scanner GUI's first-launch wizard
creates this file; if you're running LiveWatch on a machine that never had
the Scanner installed, copy `config.ini` from a working teammate.

## Launch

```bash
poetry run python ScanAnalysis/LiveWatchGUI/main.py
```

Optional flag: `--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL` (default
`INFO`).

## The main window

![LiveWatch initial view with the Configuration, Runtime Options, Control,
and Log Output sections](
assets/livewatch_01_initial.png)

Four group boxes top-to-bottom:

- **Configuration** — what to run and where
- **Runtime Options** — how to run
- **Control** — Start / Stop / Status
- **Log Output** — live log feed from the runner

## Configuration

![LiveWatch with HTU/baseline selected as the analyzer group and the scan
config dir auto-detected](
assets/livewatch_02_group_selected.png)

Field by field:

- **Experiment (for Google Docs)** — selects which experiment's e-log gets
  the upload (Undulator, Thomson, or *(none)* to disable e-log entirely).
  Independent from the GEECS experiment in `config.ini` because they're
  often the same but need not be.
- **Namespace** — filters the **Analyzer Group** dropdown to one
  sub-directory of `scan_analysis_configs/groups/` (e.g. `HTU`). Set to
  `(all)` to see every group.
- **Analyzer Group** — the canonical key for the group config to run
  (`HTU/baseline`, `PW/standard`, etc.). The dropdown is auto-populated from
  `scan_analysis_configs/groups/` on launch; the refresh button next to it
  re-scans without restarting. The combo is also editable — type a key
  directly if the discovery missed it.
- **Date** — which day's scan folder to watch. Defaults to today; back-date
  to re-process a previous day.
- **Start Scan #** — the first scan number to process. Set to `0` for
  "every scan from the start of the day"; bump it if you only want to
  process from some midpoint forward.
- **Enable GDoc Upload** — when ticked, each completed analyzer's display
  files are uploaded to the experiment's Google Doc. Requires
  `LogMaker4GoogleDocs` + a valid Document ID below.
- **Document ID** — Google Doc destination. Auto-detected from
  `config.ini`'s experiment block when possible; manual entry otherwise.
- **Scan Config Dir / Image Config Dir** — typically auto-detected from
  `config.ini`. Override per-launch with `Browse…` if you want to point at
  a different config tree (e.g. a draft).

## Runtime options

- **Max Items / Cycle** — how many tasks the runner claims per polling
  cycle. `1` is the steady-state default; bump it if you're back-filling a
  large day and want to process several scans in parallel.
- **Dry Run** — walks the queue and reports what *would* run without
  actually invoking analyzers. Useful for sanity-checking a new group config
  before letting it loose on real data.
- **Rerun Completed** / **Rerun Failed** — by default the task queue skips
  tasks that have already succeeded or failed. Tick the relevant box to
  force re-processing.

## Control & status

- **▶ Start** — kicks off the runner in a background thread. The status
  pip turns from `Idle` (grey) to `Running` (green) once the watcher is
  alive.
- **⏹ Stop** (replaces Start when running) — asks the runner to drain
  current work and stop cleanly. Force-quit by closing the window.
- **Status…** — opens a per-scan, per-analyzer grid showing task state
  (pending / running / completed / failed), heartbeat freshness, and
  display files produced. Useful for diagnosing which task in a group
  failed when the log is too noisy to scan.

## Log output

The Log Output box receives the runner's log stream live, color-coded by
level. The **Level** dropdown filters what gets shown (drop it from `INFO`
to `WARNING` for a quieter run; `DEBUG` for full firehose). **Clear** wipes
the panel without affecting the on-disk log files the runner also writes.

## Where files land

LiveWatch reads from and writes to the standard GEECS scan-folder layout:

```
{geecs_data}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/
├── scans/
│   └── ScanNNN/
│       ├── ScanNNN.tdms              # raw acquisition (Scanner-written)
│       ├── ScanData_scan.txt         # s-file (Scanner-written)
│       └── analysis_status/          # ← LiveWatch's task-queue YAML files
│           ├── <analyzer_id>.yaml
│           └── …
└── analysis/
    └── ScanNNN/
        └── …                         # ← Analyzer output (figures, scalars)
```

LiveWatch never creates a `scans/ScanNNN/` folder — those are produced by
the Scanner GUI when a scan runs. If you point LiveWatch at a date with no
scan folders yet, the runner simply idles until folders appear.
`analysis/ScanNNN/` subdirectories *are* created by LiveWatch as analyzers
produce output.

## See also

- The [end-to-end tutorial](tutorial.md) walks through building a config in
  ConfigFileGUI, referencing it from a group, and running the group here.
- [ConfigFileGUI](config_file_gui.md) — what produces the configs LiveWatch
  consumes.
- [Scan Analysis overview](../scan_analysis/overview.md) — the
  `LiveTaskRunner` engine LiveWatch wraps; useful if you want to drive it
  headlessly from a script.
- [Python API Scripting Guide](../geecs_python_api/scripting_guide.md) —
  details on `config.ini` and the experiment/path resolution model.
