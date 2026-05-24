# ImageAnalysis — Developer Context for Claude

Per-image analysis framework. Provides a pipeline-based processing system (ROI,
background subtraction, filtering, transforms) and a hierarchy of typed analyzer
classes. The output of every analyzer is a standardized `ImageAnalyzerResult`.

## Scope Note

The `analyzers/` directory contains older LabVIEW-era analyzers that are
**not** the focus of active development. The `offline_analyzers/` directory
contains the modern, actively maintained classes. Focus here.

## Package Layout

```
image_analysis/
  base.py                          # ImageAnalyzer abstract base; ImageAnalyzerResult in types.py
  types.py                         # ImageAnalyzerResult, Array1D, Array2D type aliases
  config_loader.py                 # load_camera_config(), load_line_config() — YAML entry points
  processing/
    array2d/
      config_models.py             # CameraConfig and all sub-models (ROI, Background, etc.)
      background.py                # Background subtraction implementations
      background_manager.py        # BackgroundManager orchestrator
      filtering.py                 # Gaussian / median filters
      masking.py                   # Crosshair and circular masking
      transforms.py                # Rotation, flip, distortion
      pipeline.py (or inline)      # apply_camera_processing_pipeline()
    array1d/
      config_models.py             # Line1DConfig and sub-models (ROI1D, Filtering, etc.)
      background.py                # 1D background subtraction
      roi.py                       # 1D ROI (x-value range, not pixel)
      thresholding.py              # 1D thresholding
  offline_analyzers/
    __init__.py                    # Exports: StandardAnalyzer, Standard1DAnalyzer, BeamAnalyzer, LineAnalyzer
    standard_analyzer.py           # StandardAnalyzer — 2D foundation
    standard_1d_analyzer.py        # Standard1DAnalyzer — 1D foundation
    beam_analyzer.py               # BeamAnalyzer(StandardAnalyzer)
    line_analyzer.py               # LineAnalyzer(Standard1DAnalyzer)
    ict_1d_analyzer.py             # ICT1DAnalyzer(Standard1DAnalyzer)
    density_from_phase_analysis.py # DensityFromPhaseAnalyzer(ImageAnalyzer)
    ...
```

## Core Abstractions

### `ImageAnalyzer` (base.py)

Abstract base class all analyzers implement.

```python
class ImageAnalyzer:
    run_analyze_image_asynchronously: bool = False

    def load_image(self, file_path: Path) -> Array1D | Array2D:
        # Default: read_imaq_image() — override for custom formats (TDMS, HDF5, CSV)

    def analyze_image(self, image, auxiliary_data=None) -> ImageAnalyzerResult:
        # Must implement — the main per-shot analysis method

    def analyze_image_file(self, file_path, auxiliary_data=None) -> ImageAnalyzerResult:
        # Canonical entry point for scan-level pipelines:
        # load_image() then analyze_image() — atomically, in one task.
        # Override only if your analyzer needs to thread state between
        # load and analyze (rare; the base composition is correct for
        # almost all cases).
```

### `ImageAnalyzerResult` (types.py)

Pydantic model. All analyzers return this.

```python
class ImageAnalyzerResult(BaseModel):
    data_type: Literal["1d", "2d", "scalars_only"] = "scalars_only"
    processed_image: Optional[NDArray] = None    # 2D array
    line_data: Optional[NDArray] = None          # Nx2 array (col0=x, col1=y)
    scalars: Dict[str, float] = {}               # Named scalar metrics
    metadata: Dict[str, Any] = {}                # Config, context, parameters
    render_data: Dict[str, RenderDataValue] = {} # Projections, overlays, etc.
    render_function: Optional[Callable] = None   # Custom rendering hook
```

Key methods:
- `get_primary_data()` → image or line_data regardless of type
- `has_image_data()` → True for "1d" or "2d"
- `set_xy_projections(horiz, vert)` → standard pattern for beam analysis
- `ImageAnalyzerResult.average(results)` → nanmean over a list of results

## Config System

### 2D Image Configs (`CameraConfig`)

YAML files live in the **GEECS-Plugins-configs** repo. Loaded via:

```python
from image_analysis.config_loader import load_camera_config
cfg = load_camera_config("UC_GaiaMode")  # finds UC_GaiaMode.yaml in configs repo
```

`CameraConfig` fields (all processing sections are `Optional` — omit to skip):

```python
class CameraConfig(BaseModel):
    name: str
    description: Optional[str]
    bit_depth: int = 16          # 8, 10, 12, 14, 16, 32

    roi: Optional[ROIConfig]                    # x_min, x_max, y_min, y_max (pixels)
    background: Optional[BackgroundConfig]      # method, file_path, constant_level, additional_constant
    crosshair_masking: Optional[CrosshairMaskingConfig]
    circular_mask: Optional[CircularMaskConfig]
    vignette: Optional[VignetteConfig]          # radial_polynomial or map_file method
    thresholding: Optional[ThresholdingConfig]
    filtering: Optional[FilteringConfig]        # gaussian_sigma, median_kernel_size
    normalization: Optional[NormalizationConfig]
    transforms: Optional[TransformConfig]       # rotation_angle, flip_horizontal, flip_vertical
    pipeline: Optional[PipelineConfig]          # Ordered list of ProcessingStepType
    analysis: Optional[Dict[str, Any]]          # Analyzer-specific (validated per-analyzer)
```

`BackgroundConfig.method` options: `constant`, `from_file`. Scan-context
backgrounds (cross-scan dark via `scan.background_source.scan_number`, or
dynamic from-current-scan via `scan.background_source.from_current_scan`)
are expressed at the diagnostic config layer in ScanAnalysis. The scan
analyzer computes / caches the resulting `.npy`, then rewrites this
config to a static `FROM_FILE` background pointing at the cache before
the per-shot pipeline runs.

### 1D Line Configs (`Line1DConfig`)

```python
from image_analysis.config_loader import load_line_config
cfg = load_line_config("U_BCaveICT")
```

```python
class Line1DConfig(BaseModel):
    name: str
    description: str
    data_loading: Data1DConfig      # data_type (tek_scope_hdf5, tdms_scope, csv, tsv, npy),
                                    # trace_index, x_column, y_column, delimiter
    x_scale_factor: float = 1.0    # Applied FIRST before any processing
    y_scale_factor: float = 1.0
    x_units: Optional[str]
    y_units: Optional[str]

    roi: Optional[ROI1DConfig]              # x_min, x_max in data units (not pixels)
    background: Optional[BackgroundConfig]
    filtering: Optional[FilteringConfig]
    thresholding: Optional[ThresholdingConfig]
    interpolation: Optional[InterpolationConfig]  # Resample to uniform x-axis
    pipeline: Optional[PipelineConfig]
    analysis: Optional[Dict[str, Any]]
```

## Offline Analyzers

### `StandardAnalyzer` (2D foundation)

```python
analyzer = StandardAnalyzer(
    camera_config_name="UC_GaiaMode",
    name_suffix=None,       # Appended to camera name
    metric_suffix=None,     # Appended to all scalar metric keys
)
```

Key methods:
- `preprocess_image(image) -> np.ndarray` — applies full processing pipeline
- `analyze_image(image, auxiliary_data) -> ImageAnalyzerResult` — data_type="2d"
- `analyze_image_file(path, auxiliary_data)` — canonical scan-pipeline entry
- `render_image(result, vmin, vmax, cmap, ...) -> (Figure, Axes)` — static method

### `Standard1DAnalyzer` (1D foundation)

```python
analyzer = Standard1DAnalyzer(line_config_name="U_BCaveICT")
```

Key methods:
- `load_image(file_path)` — uses `read_1d_data()` not image reader; returns Nx2 array
- `preprocess_data(data)` — applies scale factors + line processing pipeline
- `analyze_image(image, ...) -> ImageAnalyzerResult` — data_type="1d"
- `render_image(result, ...)` — line plot with unit-aware axis labels

### `BeamAnalyzer(StandardAnalyzer)`

Adds beam-specific metrics (centroid, size, moments). Uses `analysis:` section of
`CameraConfig` validated into a typed `BeamAnalysisConfig`. Most commonly used
2D analyzer.

### `LineAnalyzer(Standard1DAnalyzer)`

Adds statistics: CoM, FWHM, RMS, peak analysis. Supports `metric_suffix` for
distinguishing variants (e.g., "before_foil" vs "after_foil").

### `ICT1DAnalyzer(Standard1DAnalyzer)`

Specialized for Integrated Current Transformer (charge measurement). Applies
Butterworth filter, calibration factor, time-step integration from `analysis:`
section of config.

### `DensityFromPhaseAnalyzer(ImageAnalyzer)`

Direct subclass of `ImageAnalyzer` (not Standard). Plasma density from wavefront
phase data — Abel inversion, background removal, rotation alignment, Gaussian
masking.

## Adding a New Offline Analyzer

### 2D analyzer

```python
from image_analysis.offline_analyzers.standard_analyzer import StandardAnalyzer
from image_analysis.types import ImageAnalyzerResult
from pydantic import BaseModel

class MyAnalysisConfig(BaseModel):
    threshold: float = 0.1
    # ... typed config fields from analysis: section of CameraConfig

class MyAnalyzer(StandardAnalyzer):
    def __init__(self, camera_config_name: str, **kwargs):
        super().__init__(camera_config_name, **kwargs)
        self._my_cfg = MyAnalysisConfig(**(self.camera_config.analysis or {}))

    def analyze_image(self, image, auxiliary_data=None) -> ImageAnalyzerResult:
        result = super().analyze_image(image, auxiliary_data)  # preprocessing + base result
        # Add custom scalars
        result.scalars["my_metric"] = compute_something(result.processed_image, self._my_cfg)
        return result
```

### 1D analyzer

Same pattern but inherit from `Standard1DAnalyzer` and use `self.line_config.analysis`.

## Key Design Decisions

- **Processing happens in `preprocess_image()` / `preprocess_data()`** — `analyze_image()`
  receives already-processed data. Keep analysis logic separate from preprocessing.
- **`analysis:` dict in config** — typed per-analyzer via a local Pydantic model
  defined in the analyzer class. Validated at `__init__` time, not at config-load time.
- **Scale factors applied first** — `x_scale_factor` / `y_scale_factor` run before
  ROI, so ROI boundaries and thresholds should be specified in scaled units.
- **`metric_suffix`** — Use when the same physical device has multiple analysis
  variants in one scan (e.g., two ROI regions). Keeps scalar keys unique.
- **Nx2 convention for 1D data** — Column 0 is always x (independent), column 1
  is always y (dependent). `read_1d_data()` enforces this.

## Filesystem invariants for offline_analyzers that write inside `scans/ScanNNN/`

Some analyzers (`LineStitcher`, `MagSpecManualCalibAnalyzer`,
`HASOHimgHasProcessor`, `GrenouilleAnalyzer`) save derived per-shot outputs
into a subfolder of the source scan dir — e.g. `<scan_dir>/<device>-interp/`.
This is intentional and mirrors notebook workflows. **But analysis code never
creates the scan folder itself.** See
[Cross-package invariants](../CLAUDE.md#cross-package-invariants) in the root
for the full background and the production incident that motivated this rule.

When you write a new analyzer that emits files inside the scan dir:

1. Compute `scan_dir` (typically `file_path.parent.parent`).
2. **Guard before any `mkdir`:**
   ```python
   if not scan_dir.is_dir():
       raise FileNotFoundError(
           f"Scan folder {scan_dir} is not visible; refusing to create "
           f"output subfolder. ..."
       )
   ```
3. Create the output subfolder with `mkdir(exist_ok=True)` only — **never**
   `parents=True` on a path that traverses through `scans/`. If `scan_dir` is
   real but the subfolder is missing, that's the one and only level you may
   create.

If your analyzer's save logic instead lives in a utility like
`save_background_to_file`, the utility must require its parent dir to exist
(raise `FileNotFoundError` otherwise) — the caller is responsible for the
guard above. `image_analysis.processing.array1d.background.save_background_to_file`
is the canonical example.

Invariant is pinned by tests:
- `tests/analyzers/test_line_stitcher.py::TestLineStitcherScanFolderInvariant`
- `tests/analyzers/test_magspec_calib.py::TestScanFolderInvariant`
- `tests/processing/test_array1d_background.py`
