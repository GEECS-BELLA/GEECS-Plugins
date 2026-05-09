# Scan Output Structure

This page documents the file layout that the scanner produces and how to load each file from Python. If you've just run a scan and want to find your data, you're in the right place.

## The folder convention

Every scan writes to a folder named after the experiment, the date, and the scan number:

```
{base_path}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/scans/Scan{NNN}/
```

Concrete example:

```
Z:/data/Undulator/Y2026/05-May/26_0508/scans/Scan042/
```

`base_path` is configured per machine in `~/.config/geecs_python_api/config.ini`. On a Windows acquisition workstation it's typically a mapped network share; on Linux/Mac it's the equivalent mount path. `Scan{NNN}` is zero-padded to three digits and increments through the day.

Don't hardcode this path in scripts. Use `geecs_data_utils.ScanPaths` — it handles every machine's path resolution from one config file.

## What's inside a scan folder

A typical scan folder looks like this:

```
Scan042/
├── Scan042.tdms                  ← binary TDMS log (LabVIEW-compatible)
├── ScanData_scan.txt             ← s-file (per-shot scalars, TSV)
├── scan_info.ini                 ← scan metadata
├── analysis/                     ← created by ScanAnalysis if it runs
│   └── analysis_status/
│       └── *.yaml                ← task queue files
├── U_ModeImager/                 ← per-device subfolder for save_nonscalar devices
│   ├── Scan042_U_ModeImager_001.png
│   ├── Scan042_U_ModeImager_002.png
│   └── ...
└── U_HiResMagCam/
    └── ...
```

The contents depend on what was in your save element. Devices with `save_nonscalar_data: true` get a subfolder; devices that only provide scalars are recorded in `ScanData_scan.txt` and `Scan042.tdms` only.

## File-by-file reference

### `Scan{NNN}.tdms`

National Instruments TDMS format — the binary scalar log. Every variable in your save element gets a TDMS channel; every shot is a row. This is the file LabVIEW analysis tools read directly; from Python it's most often loaded indirectly via the s-file.

Load with:

```python
from nptdms import TdmsFile
tdms = TdmsFile.read("Scan042.tdms")
df = tdms.as_dataframe()
```

### `ScanData_scan.txt` — the s-file

Tab-separated values with a one-line header. Each row is one shot. Columns are:

- `Shotnumber` — integer, one-indexed.
- `Timestamp` — wall-clock time of the shot (Unix-style float seconds).
- `acq_timestamp` — hardware timestamp of acquisition (used for synchronization checks).
- One column per recorded variable, named `{DeviceName}.{variable}` — for example `U_ModeImager.exposure`.
- For 1D and optimization scans: a `scan_var` column holding the value of the scanned variable at each shot.

The s-file is the canonical "what happened during this scan" record. It's the input to ScanAnalysis and the most common starting point for ad-hoc analysis.

Load with `geecs_data_utils`:

```python
from geecs_data_utils import ScanPaths, ScanTag

tag = ScanTag(year=2026, month=5, day=8, number=42)
paths = ScanPaths(tag=tag, experiment="Undulator", read_mode=True)
df = paths.load_sfile()
```

Or directly with pandas:

```python
import pandas as pd
df = pd.read_csv("ScanData_scan.txt", sep="\t")
```

The `ScanPaths` route is preferred because it works regardless of which machine you're on and which mount points are in use.

### `scan_info.ini`

Plain INI file with scan metadata. Contains the experiment name, scan number, mode (NOSCAN / 1D / optimization / background), the scanned variable and its range, repetition rate, the operator description, and a copy of the relevant ScanOptions.

Read it with `configparser`:

```python
import configparser

cfg = configparser.ConfigParser()
cfg.read("scan_info.ini")
mode = cfg["Scan Info"]["scan_mode"]
description = cfg["Scan Info"]["description"]
```

`scan_info.ini` is what tells downstream tools whether this folder represents a scan worth analyzing. ScanAnalysis reads it before running; it's also what the Google Doc uploader uses to title each entry.

### `analysis/`

Created by ScanAnalysis if you run it. Contains rendered summary figures and the task-queue YAML files that record which analyzers have processed this scan.

If you didn't run ScanAnalysis (live or offline), this folder doesn't exist.

### Per-device subfolders

For every device in your save element with `save_nonscalar_data: true`, the scanner creates a subfolder named after the device and pulls every per-shot file into it. The naming convention is `Scan{NNN}_{DeviceName}_{ShotNumber}.{ext}`.

Files are matched to shots by the device's per-shot timestamp. The scanner's FileMover runs in parallel with acquisition; if the network share is slow or a device finishes writing late, files queue and get drained at scan teardown. If a file never arrives within the orphan-sweep timeout, it's logged at WARNING but the scan still completes — the s-file row exists, only the binary file is missing.

## Loading per-shot files

Image files in particular are most cleanly loaded via the analyzer framework. For ad-hoc loading:

```python
from pathlib import Path
from geecs_data_utils import ScanPaths, ScanTag

tag = ScanTag(year=2026, month=5, day=8, number=42)
paths = ScanPaths(tag=tag, experiment="Undulator", read_mode=True)
device_folder = paths.get_folder() / "U_ModeImager"

# Match to a row in the s-file by shot number:
shot = 17
candidates = list(device_folder.glob(f"*_{shot:03d}.png"))
if candidates:
    image_path = candidates[0]
```

For LabVIEW-saved PNGs, use `image_analysis.utils.read_imaq_png_image` instead of standard PNG loaders — LabVIEW saves with a bit-shift that needs unpacking.

## Where the layout is enforced

The folder layout is enforced by `ScanDataManager` in `geecs_scanner/engine/scan_data_manager.py` (not by the scanner GUI). The GUI just hands a `ScanExecutionConfig` to the engine; the engine asks `ScanPaths` to claim the next scan number and create the folder. If you're scripting scans without the GUI, you get the same layout for free.

If you're loading data and the path resolution is going wrong, the question is almost always "is `geecs_paths_config.py` looking at the right base path?" Check `ScanPaths.paths_config` and reload it with the right experiment if needed.

## Where to go next

- **[Save Elements](save_elements.md)** — controls what variables and per-device files end up here.
- **[Scan Analysis package](../scan_analysis/overview.md)** — the standard way to process scan folders.
- **[Image Analysis package](../image_analysis/overview.md)** — analyzers for the per-device image files.
- **[Data Utils package](../geecs_data_utils/overview.md)** — full API for path resolution and s-file loading.
