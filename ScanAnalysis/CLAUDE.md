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
    analyzer_config_models.py      # Pydantic models for YAML configs
    analyzer_factory.py            # Instantiates analyzers from config objects
  analyzers/
    common/
      single_device_scan_analyzer.py   # SingleDeviceScanAnalyzer base
      array2d_scan_analysis.py         # Array2DScanAnalyzer
      array1d_scan_analysis.py         # Array1DScanAnalyzer
```

## Config System (YAML → Pydantic → Factory → Instances)

All scan analysis is driven by YAML config files stored in the
**GEECS-Plugins-configs** repository (not this repo). The YAML is validated
against Pydantic models and then fed to a factory.

### Config Models (`analyzer_config_models.py`)

```
ExperimentAnalysisConfig          # Top-level: experiment name + list of analyzers
  analyzers: List[ScanAnalyzerConfig]   # Union[Array2DAnalyzerConfig, Array1DAnalyzerConfig]
  include: List[IncludeEntry]           # Optional: include refs from a library YAML

Array2DAnalyzerConfig             # Wraps a 2D image analyzer
  type: Literal["array2d"]
  device_name: str
  priority: int                   # Lower = runs first (0 = highest)
  image_analyzer: ImageAnalyzerConfig
  gdoc_slot: Optional[int]        # 0-3 → table cell; None → hyperlink mode
  is_active: bool

Array1DAnalyzerConfig             # Wraps a 1D line/spectrum analyzer
  type: Literal["array1d"]
  device_name: str
  priority: int
  image_analyzer: ImageAnalyzerConfig
  gdoc_slot: Optional[int]
  is_active: bool

ImageAnalyzerConfig               # Specifies which ImageAnalyzer class to use
  analyzer_class: str             # Fully qualified: "image_analysis.offline_analyzers...."
  camera_config_name: Optional[str]   # Name of CameraConfig/Line1DConfig YAML
  kwargs: Dict[str, Any]          # Extra constructor args
```

### Factory (`analyzer_factory.py`)

`create_analyzer(config, ...)` → `ScanAnalyzer` instance with `.id`, `.priority`,
`.gdoc_slot` stamped on it. Dynamic import via `_import_class()` — the
`analyzer_class` string is resolved at runtime, so new analyzer classes are
picked up without changing factory code.

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
- `_run_analysis_core()` → fetches device data files, runs per-shot analysis
  via `ImageAnalyzer.analyze_image()`, then calls `_postprocess_noscan()` or
  `_postprocess_scan()` depending on scan mode
- `DataUnavailableWarning` — raised when device data dir is missing or empty;
  caught with `logger.warning()` only (no traceback). Separate from real errors
  which still log with traceback.

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
    experiment="Undulator",
    config_path="path/to/experiment_config.yaml",
    document_id=None,   # None → reads from INI file (live mode)
                        # explicit string → targets specific historical doc (backtest)
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

## Adding a New Scan Analyzer

1. Create (or reuse) an `ImageAnalyzer` subclass in `ImageAnalysis`
2. Create a `CameraConfig` or `Line1DConfig` YAML in the configs repo
3. Add an entry to the experiment's analysis YAML:
   ```yaml
   - type: array2d   # or array1d
     device_name: MyDevice
     priority: 50
     image_analyzer:
       analyzer_class: image_analysis.offline_analyzers.my_analyzer.MyAnalyzer
       camera_config_name: MyDevice
   ```
4. No Python changes needed in ScanAnalysis itself.
