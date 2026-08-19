# ScanAnalysis — Developer Context for Claude

Post-scan analysis framework. Watches for new scans, runs configurable chains of
image/1D analyzers, and optionally uploads summary figures to Google Docs.

## Package Layout

```
scan_analysis/
  base.py                          # ScanAnalyzer abstract base class
  live_task_runner.py              # LiveTaskRunner: watches for s-files → drives queue
  task_queue.py                    # Task claiming, heartbeat, YAML status system
  gdoc_upload.py                   # GDoc upload integration (optional logmaker dep)
  config/
    diagnostic_models.py           # Scan-side runtime + group Pydantic models
    diagnostic_factory.py          # create_scan_analyzer(DiagnosticAnalysisConfig)
    analysis_group_loader.py       # discover_analyzers/groups + load_analysis_group
  analyzers/
    common/
      single_device_scan_analyzer.py   # SingleDeviceScanAnalyzer base
      array2D_scan_analysis.py         # Array2DScanAnalyzer
      array1d_scan_analysis.py         # Array1DScanAnalyzer
      scatter_plotter_analysis.py      # ScatterPlotterAnalysis + PlotParameter
```

## Config System (YAML → Pydantic → Factory → Instances)

Scan analysis is driven by YAML config files stored in the
**GEECS-Plugins-configs** repository (not this repo). Image-analyzer-driven
scan analyzers (Array2D / Array1D) use the **unified diagnostic schema**:
one YAML per diagnostic under `analyzers/<namespace>/<id>.yaml`, carrying
both an `image:` section (consumed by ImageAnalysis) and a `scan:` section
(consumed by ScanAnalysis). Diagnostics are assembled into analysis groups
under `groups/<namespace>/<group>.yaml`, which `LiveWatch` and the task
queue consume directly. Scatter analyzers sit outside the YAML config
system entirely — they are plain Python subclasses of
`ScatterPlotterAnalysis` (see below) because they don't consume images.

### Public loader / factory API

```python
from scan_analysis.config import (
    load_analysis_group, create_scan_analyzer,
    discover_analyzers, discover_groups,
)

group = load_analysis_group("baseline", config_dir=...)
analyzers = [
    create_scan_analyzer(r.diagnostic, id=r.id, priority=r.priority)
    for r in group.analyzers
]
for a in analyzers:
    a.run_analysis(scan_tag)
```

`task_queue.load_analyzers_from_config(group_name, config_dir=...)` is a
thin wrapper around the same two calls.

### Unified diagnostic schema

The top-level `DiagnosticAnalysisConfig` lives in **`image_analysis.config`**
(it owns the `image_analyzer` + `image:` shape and carries `scan:` as a
weakly-typed dict). `scan_analysis.config` re-exports it and owns the
scan-side models in `diagnostic_models.py`:

```
DiagnosticAnalysisConfig          # One YAML per diagnostic (image_analysis.config)
  name: str                       # Device/channel name for input-data discovery
  image_analyzer: ImageAnalyzerSpec  # Analyzer class path (+ optional kwargs)
  image: CameraConfig | Line1DConfig | None  # Routed by `type: camera | line`
  output_name: Optional[str]      # Output stem override (defaults to name)
  metric_suffix: Optional[str]    # Scalar-key-only suffix (no dir/file effect)
  scan: dict                      # Validated by ScanAnalysis into ScanRuntimeConfig

ScanRuntimeConfig                 # Validates the scan: dict (diagnostic_models.py)
  priority: int                   # Lower = runs first (100 default)
  mode: Literal["per_shot", "per_bin"]  # default per_shot
  save: bool                      # Write per-shot/bin outputs to the analysis tree
  gdoc_slot: Optional[int]        # 0-3 → table cell; None → hyperlink upload
  device: Optional[str]           # Data-subfolder override (defaults to name)
  file_tail: Optional[str]        # Filename suffix matching this device's files
  renderer_kwargs: dict           # Extra renderer options (colormap mode, ...)
  background_source: Optional[BackgroundSource]
                                  # scan_number | from_current_scan | autodetect

AnalysisGroupConfig               # One YAML per group under groups/
  analyzers: List[AnalyzerRef]    # Bare stem strings or {ref, enabled, priority}

ResolvedDiagnosticConfig          # What the loader hands the factory
  id: str                         # Diagnostic filename stem (task-queue ID)
  enabled: bool                   # Refs with enabled: false are excluded
  priority: int                   # Group override, else the diagnostic's own
  diagnostic: DiagnosticAnalysisConfig
```

There is no `scan.type` field: the factory picks the wrapper class from
the type of `diag.image` — `Line1DConfig` → `Array1DScanAnalyzer`,
anything else → `Array2DScanAnalyzer`.

**Output-naming contract (#412)** — image analyzers emit **bare** scalar
keys (`x_fwhm`, not `UC_TopView_x_fwhm`); the ScanAnalyzer wrapper applies
the diagnostic's `output_name` prefix (defaults to `name`) and
`metric_suffix` when storing per-shot results. `output_name` also names
the per-analyzer output directory under `analysis/Scan<NNN>/`; override
it to run two analyzer variants over the same camera with distinct
output trees and s-file columns (`output_name: UC_TopView_left` /
`UC_TopView_right`). `metric_suffix` affects scalar keys only, never
directory or file names. This keeps ImageAnalysis reusable standalone —
`ImageAnalysis/CLAUDE.md` points here for the full contract.

### The `image_analyzer` field (`image_analysis.config`)

`ImageAnalyzerSpec` and `resolve_image_analyzer_value` live in
`image_analysis.config` and are re-exported by `scan_analysis.config`.
The field accepts two forms (the former alias registry — `beam`,
`standard`, … — was removed along with `aliases.py`):

```yaml
image_analyzer: image_analysis.analyzers.beam_analyzer.BeamAnalyzer  # bare class path
image_analyzer:                       # verbose, for constructor kwargs
  class_path: image_analysis.analyzers.HASO_himg_has_processor.HASOHimgHasProcessor
  kwargs: {mask_top: 125}
```

### Scatter (`analyzers/common/scatter_plotter_analysis.py`)

Scatter analyzers read scalar columns from the s-file and produce a
summary plot — no image data flows through them, so the unified
`image:` / `scan:` shape doesn't apply. They are configured directly in
Python: subclass `ScatterPlotterAnalysis` and pass one or more
`PlotParameter` named tuples (sfile key, legend/axis labels, color).
`analyzers/Undulator/ict_plot_analysis.py` (`ICTPlotAnalysis`) is the
reference example. The former `analyzer_config_models.py` +
`analyzer_factory.py` scatter-config module pair was removed.

## Analyzer Class Hierarchy

```
ScanAnalyzer  (base.py)
  ├── SingleDeviceScanAnalyzer  (single_device_scan_analyzer.py)
  │     ├── Array2DScanAnalyzer  (array2D_scan_analysis.py)
  │     │     └── HIMGWithAveraging  (Undulator/HIMG_with_average_saving.py)
  │     └── Array1DScanAnalyzer  (array1d_scan_analysis.py)
  └── ScatterPlotterAnalysis  (scatter_plotter_analysis.py)
        └── ICTPlotAnalysis  (Undulator/ict_plot_analysis.py)
```

### `ScanAnalyzer.run_analysis(scan_tag) -> Optional[list[Path | str]]`

The main entry point. Returns a list of **display files** (paths to summary
figures) that the task queue stores and optionally uploads to GDocs, or
`None` when there was nothing to analyze.

### `SingleDeviceScanAnalyzer`

- Holds an `ImageAnalyzer` instance
- `_run_analysis_core()` → resolves the device data folder, then dispatches
  to one of two streaming pipelines based on `analysis_mode` (the
  constructor kwarg fed from `scan.mode` in the diagnostic YAML):
  - **`per_shot`** (default): fused per-shot tasks call
    `ImageAnalyzer.analyze_image_file(path, aux)` atomically. One image
    is loaded and analyzed per task; per-shot data never has to shuttle
    between separate load and analyze phases through analyzer-instance
    state. This is the correctness property enforced after the shot-by-shot
    refactor (1.5.0) — it eliminates a whole class of bugs (aux-columns
    regression, stale `data_metadata`, etc.).
  - **`per_bin`**: streams bin-by-bin. For each bin, parallel-load that
    bin's files, average, run `analyze_image` once on the averaged image,
    store result, release. Memory bounded by one bin's image count. Use
    this for analyzers where running on the bin-average is scientifically
    distinct from per-shot + post-hoc result averaging (nonlinear measures,
    threshold-based metrics, etc.).
- Both pipelines call `_postprocess_noscan()` or `_postprocess_scan()` once
  the per-task work is done.
- `DataUnavailableWarning` — raised when device data dir is missing or empty;
  caught with `logger.warning()` only (no traceback). Separate from real errors
  which still log with traceback.

#### Adding a new analyzer

Implement `analyze_image_file(path, aux)` if your analyzer needs to
coordinate load and analyze (rare). Otherwise, just implement
`analyze_image(image, aux)` and `load_image(path)` and rely on the base
class composition. **Do not** rely on instance state being preserved
between a separate `load_image` call and a later `analyze_image` call —
the per-shot pipeline runs them inside one atomic task per shot, but
shared instance state across tasks is undefined under parallelism.

### `Array2DScanAnalyzer`

- Wraps any 2D `ImageAnalyzer` (StandardAnalyzer, BeamAnalyzer, etc.)
- `_postprocess_noscan()` → averaged image
- `_postprocess_scan()` → grid montage of per-bin averages
- Uses `Image2DRenderer` for consistent figure rendering
- `renderer_kwargs` from config: colormap mode (sequential/diverging), vmin/vmax

### `Array1DScanAnalyzer`

- Wraps any 1D `ImageAnalyzer` (Standard1DAnalyzer, LineAnalyzer, etc.)
- `_postprocess_noscan()` → averaged line plot
- `_postprocess_scan()` → waterfall plot (one trace per bin)
- Uses `Line1DRenderer`
- `renderer_kwargs`: colormap mode for waterfall coloring

## Task Queue System (`task_queue.py`)

Enables multiple `LiveTaskRunner` processes to divide work without conflicts.

### How It Works

1. A **status YAML** is created per scan per analyzer:
   `<scan_folder>/analysis_status/<analyzer_id>.yaml`
2. States: `queued → claimed → done / failed`
3. When a runner picks up a task it writes a **heartbeat** (timestamp) every 30s
4. A claimed task is considered **stale** after 180s without a heartbeat update
5. Other runners can re-claim stale tasks — safe parallelism without a central
   coordinator

### `TaskStatus` fields

- `status: str` — queued / claimed / done / failed
- `claimed_by: Optional[str]` — runner identifier
- `heartbeat: Optional[float]` — unix timestamp of last ping
- `display_files: Optional[List[str]]` — populated when analyzer completes

## Live Watching (`live_task_runner.py`)

`LiveTaskRunner` watches a data directory for new s-files (scan summary files),
enqueues analysis tasks, and drives `run_worklist()`.

```python
runner = LiveTaskRunner(
    analyzer_group="baseline",          # group name under groups/<namespace>/
    date_tag=ScanTag(year=..., experiment="Undulator", ...),
    config_dir=None,                    # None → uses paths_config default
    document_id=None,                   # None → reads from INI (live mode);
                                        # explicit string → historical doc (backtest)
)
runner.start()
```

Multiple `LiveTaskRunner` instances can run concurrently — the heartbeat
staleness system handles contention.

## GDoc Upload (`gdoc_upload.py`)

Called by `run_worklist()` after an analyzer completes, if `gdoc_slot is not None`.

```python
upload_summary_to_gdoc(
    scan_tag,               # ScanTag; carries scan number + experiment
    display_files,          # List of paths; uploads display_files[-1]
    gdoc_slot,              # 0=row0/col0, 1=row0/col1, 2=row1/col0, 3=row1/col1
    document_id=None,       # None → reads from experiment INI
)
```

- **Per-day folder:** If `ImageParentFolderID` is set in the experiment INI,
  images land in a date-named subfolder under it (persistent). Otherwise falls
  back to `_FALLBACK_IMAGE_FOLDER` (may be purged).
- **logmaker optional:** If `logmaker_4_googledocs` is not installed, calls are
  silently skipped.

## Key Design Decisions

- **`priority`** — Lower number runs first. Default 100; use low numbers
  for fast diagnostics.
- **`gdoc_slot`** — Set 0-3 to insert into a 2×2 table cell. Omit (None) to
  upload display files as hyperlinks instead, when the runner has gdoc
  upload enabled.
- **`enabled: false`** on a group ref — Disable an analyzer without
  removing it from the group config.
- **`image_analyzer`** — Fully qualified class path; resolved at runtime.
  Adding a new analyzer class requires no factory changes.
- **Embedded `image:` config** — The per-device image-processing config
  (ROI, background, pipeline) lives inside the diagnostic YAML itself;
  there is no separate camera-config lookup.

## Filesystem invariants

**ScanAnalysis never creates `scans/ScanNNN/`.** It is a consumer of scan
folders that the scanner already wrote. This rule is load-bearing — see
[Cross-package invariants](../CLAUDE.md#cross-package-invariants) in the root
for the full background and the production incident that motivated it.

In practice:

- All `ScanPaths(...)` calls in this package use the default `read_mode=True`
  (which raises on a missing folder). Never pass `read_mode=False` from
  analysis code.
- `task_queue.init_status_for_scan` and `task_queue.update_status` verify
  `scan_folder.is_dir()` and bail with an `ERROR` log if it's missing — they
  do **not** auto-create. LiveWatch keeps running other work; if the scan
  folder later reappears, discovery can pick it up on a later processing pass
  or after relaunch.
- `analysis_status/` is the only directory ever auto-created by this package,
  and only via `mkdir(exist_ok=True)` — no `parents=True`.
- Analyzers write their outputs to `<date>/analysis/Scan<NNN>/...`, the
  *sibling* of `scans/Scan<NNN>/`. Never write back into the scans tree.

Do not treat a missing entire scan folder as `no_data`. `no_data` means the
scan exists but a specific device/analyzer has no usable data. If the scan
folder itself is absent, `analysis_status/` is unavailable because it lives
inside that folder; logging and skipping is the safe behavior.

When writing a new analyzer, **do not** use `Path.mkdir(parents=True, ...)` on
any path that could traverse through a `scans/` folder. The invariant is
pinned by tests in `tests/test_task_queue.py::TestScanFolderCreationInvariant`.

## Adding a New Scan Analyzer

1. (Optional) Add or reuse an `ImageAnalyzer` subclass in `ImageAnalysis`.
2. Create the diagnostic YAML in the configs repo under
   `analyzers/<namespace>/<id>.yaml`, carrying both an `image:` and a
   `scan:` section:

   ```yaml
   name: MyDevice                      # device/channel name for data discovery
   image_analyzer: image_analysis.analyzers.beam_analyzer.BeamAnalyzer
   image:                              # consumed by ImageAnalysis
     type: camera                      # camera → Array2D; line → Array1D
     roi: {x_min: 0, x_max: 650, y_min: 350, y_max: 650}
     background: {method: constant, constant_level: 5.0}
     pipeline:
       steps: [background, roi]
   scan:                               # consumed by ScanAnalysis
     priority: 50
     mode: per_shot                    # or per_bin
     gdoc_slot: 0                      # optional
     # background_source:              # optional, for cross-scan or dynamic bg
     #   scan_number: 42
   ```

3. Add the diagnostic to one or more groups under
   `groups/<namespace>/<group>.yaml` — bare filename stem, or a dict
   with per-group overrides:

   ```yaml
   analyzers:
     - MyDevice
     - {ref: OtherDevice, priority: 5}
   ```

4. No Python changes needed in ScanAnalysis itself. The factory
   (`create_scan_analyzer`) resolves the `image_analyzer` class path,
   builds the inner `ImageAnalyzer` via
   `image_analysis.config.create_image_analyzer`, and wraps it in
   `Array1DScanAnalyzer` or `Array2DScanAnalyzer` based on the type of
   the `image:` section.
