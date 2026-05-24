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
    aliases.py                     # ImageAnalyzerSpec + ALIAS_REGISTRY (string / dict / verbose)
    diagnostic_models.py           # Unified diagnostic + group Pydantic models
    diagnostic_factory.py          # create_diagnostic_analyzer(ResolvedDiagnosticConfig)
    analysis_group_loader.py       # discover_analyzers/groups + load_analysis_group
    analyzer_config_models.py      # Scatter-only: PlotParameterConfig + ScatterAnalyzerConfig
    analyzer_factory.py            # Scatter-only: create_analyzer dispatch
  analyzers/
    common/
      single_device_scan_analyzer.py   # SingleDeviceScanAnalyzer base
      array2d_scan_analysis.py         # Array2DScanAnalyzer
      array1d_scan_analysis.py         # Array1DScanAnalyzer
```

## Config System (YAML → Pydantic → Factory → Instances)

Scan analysis is driven by YAML config files stored in the
**GEECS-Plugins-configs** repository (not this repo). Image-analyzer-driven
scan analyzers (Array2D / Array1D) use the **unified diagnostic schema**:
one YAML per diagnostic under `analyzers/<namespace>/<id>.yaml`, carrying
both an `image:` section (consumed by ImageAnalysis) and a `scan:` section
(consumed by ScanAnalysis). Diagnostics are assembled into analysis groups
under `groups/<namespace>/<group>.yaml`, which `LiveWatch` and the task
queue consume directly. Scatter analyzers stay on their own
`ScatterAnalyzerConfig` shape because they don't consume images.

### Public loader / factory API

```python
from scan_analysis.config import (
    load_analysis_group, create_diagnostic_analyzer,
    discover_analyzers, discover_groups,
)

group = load_analysis_group("baseline", config_dir=...)
analyzers = [create_diagnostic_analyzer(r) for r in group.analyzers]
for a in analyzers:
    a.run_analysis(scan_tag)
```

`task_queue.load_analyzers_from_config(group_name, config_dir=...)` is a
thin wrapper around the same two calls.

### Unified diagnostic schema (`diagnostic_models.py`)

```
DiagnosticAnalysisConfig          # One YAML per diagnostic
  id: str                         # Filename stem; unique per namespace
  image: ImageAnalyzerConfig      # Consumed by ImageAnalysis (camera/line + processing)
  scan: ScanRuntimeConfig         # Consumed by ScanAnalysis (priority, gdoc, etc.)

ScanRuntimeConfig
  type: Literal["array2d", "array1d", "scatter"]
  analysis_mode: Literal["per_shot", "per_bin"]  # default per_shot
  priority: int                   # Lower = runs first (200 default)
  gdoc_slot: Optional[int]        # 0-3 → table cell; None → hyperlink
  is_active: bool
  background_source: Optional[BackgroundSource]   # scan_number | from_current_scan

AnalysisGroupConfig               # One YAML per group under groups/
  analyzers: List[AnalyzerRef]    # Each ref points at a diagnostic id

ResolvedDiagnosticConfig          # What the loader hands the factory
  diagnostic: DiagnosticAnalysisConfig
  ref: AnalyzerRef                # Group-level overrides (priority etc.)
```

### Alias registry (`aliases.py`)

`image_analyzer:` values in a diagnostic accept three forms (resolved by
`resolve_image_analyzer_value`):

```yaml
image_analyzer: beam                              # alias
image_analyzer: {beam: {camera_config_name: U_Cam}}  # alias-with-overrides
image_analyzer:                                   # verbose, escape hatch
  class_path: image_analysis.analyzers.beam.BeamAnalyzer
  kwargs: {camera_config_name: U_Cam}
```

`ALIAS_REGISTRY` holds the production aliases (beam, standard, mode_imager,
…) keyed to `ImageAnalyzerSpec(class_path, default_kwargs)`.

### Scatter (`analyzer_config_models.py` + `analyzer_factory.py`)

Scatter analyzers use a separate `ScatterAnalyzerConfig` and the
scatter-only `create_analyzer` function. They read scalar columns from
the s-file and produce a single summary plot — no image data flows
through them, so the unified `image:` / `scan:` shape doesn't apply.

## Analyzer Class Hierarchy

```
ScanAnalyzer  (base.py)
  └── SingleDeviceScanAnalyzer  (single_device_scan_analyzer.py)
        ├── Array2DScanAnalyzer  (array2d_scan_analysis.py)
        └── Array1DScanAnalyzer  (array1d_scan_analysis.py)
```

### `ScanAnalyzer.run_analysis(scan_folder) -> list[Path | str]`

The main entry point. Returns a list of **display files** (paths to summary
figures) that the task queue stores and optionally uploads to GDocs.

### `SingleDeviceScanAnalyzer`

- Holds an `ImageAnalyzer` instance
- `_run_analysis_core()` → resolves the device data folder, then dispatches
  to one of two streaming pipelines based on `analysis_mode`:
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
    display_files,          # List of paths; uploads display_files[-1]
    scan_number,
    gdoc_slot,              # 0=row0/col0, 1=row0/col1, 2=row1/col0, 3=row1/col1
    document_id,            # None → reads from experiment INI
    experiment,
)
```

- **Per-day folder:** If `ImageParentFolderID` is set in the experiment INI,
  images land in a date-named subfolder under it (persistent). Otherwise falls
  back to `_FALLBACK_IMAGE_FOLDER` (may be purged).
- **logmaker optional:** If `logmaker_4_googledocs` is not installed, calls are
  silently skipped.

## Typical Config YAML

```yaml
experiment: Undulator
description: Standard Undulator analysis
version: "1.0"

analyzers:
  - type: array2d
    device_name: UC_GaiaMode
    priority: 0
    gdoc_slot: 0
    image_analyzer:
      analyzer_class: image_analysis.offline_analyzers.beam_analyzer.BeamAnalyzer
      camera_config_name: UC_GaiaMode

  - type: array1d
    device_name: U_BCaveICT
    priority: 100
    image_analyzer:
      analyzer_class: image_analysis.offline_analyzers.ict_1d_analyzer.ICT1DAnalyzer
      camera_config_name: U_BCaveICT
```

## Key Design Decisions

- **`priority`** — Lower number runs first. Use 0 for fast diagnostics, 100 for
  slow ones.
- **`gdoc_slot`** — Omit to use hyperlink mode (future PR). Set 0-3 to insert
  into 2×2 table cell.
- **`is_active: false`** — Disable an analyzer without removing it from config.
- **`analyzer_class`** — Fully qualified path; resolved at runtime. Adding a new
  analyzer class requires no factory changes.
- **`camera_config_name`** — Points to a YAML in the configs repo. The name
  (without extension) is used for lookup.

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
   id: MyDevice                       # filename stem; unique per namespace
   image:                              # consumed by ImageAnalysis
     image_analyzer: beam              # alias, alias-dict, or {class_path, kwargs}
     camera_config_name: MyDevice
   scan:                               # consumed by ScanAnalysis
     type: array2d                     # or array1d / scatter
     device_name: MyDevice
     priority: 50
     gdoc_slot: 0                      # optional
     # background_source:              # optional, for cross-scan or dynamic bg
     #   scan_number: 42
   ```

3. Add the diagnostic to one or more groups under
   `groups/<namespace>/<group>.yaml`:

   ```yaml
   analyzers:
     - id: MyDevice
   ```

4. No Python changes needed in ScanAnalysis itself. The diagnostic
   factory (`create_diagnostic_analyzer`) resolves the `image_analyzer`
   spec, instantiates the `ImageAnalyzer`, and wraps it in the right
   `Array{1,2}DScanAnalyzer`.
