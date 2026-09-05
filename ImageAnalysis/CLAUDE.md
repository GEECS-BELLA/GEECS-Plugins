# ImageAnalysis — Developer Context for Claude

Per-image analysis framework. Provides a pipeline-based processing system (ROI,
background subtraction, filtering, transforms) and a hierarchy of typed analyzer
classes. The output of every analyzer is a standardized `ImageAnalyzerResult`.

## Scope Note

The `analyzers/` directory is the modern, actively maintained home for all
image and 1D analyzer classes. Before the PR-E loader-API refactor
(ImageAnalysis 1.5.0) this directory held LabVIEW-era code and the modern
classes lived under `offline_analyzers/`; the rename inverted that. The
`offline_qualifier` was a holdover from a never-built online counterpart.
A nearly-empty `offline_analyzers/` directory may still exist with a couple
of subpackage remnants (`Thomson/`, `Undulator/`); it is deprecated and
should not be imported from.

## Package Layout

```
image_analysis/
  base.py                          # ImageAnalyzer abstract base; ImageAnalyzerResult in types.py
  types.py                         # ImageAnalyzerResult, Array1D, Array2D type aliases
  config/                          # Public configuration API — single entry point
    __init__.py                    # Exports: load_camera_config, load_line_config,
                                   #          load_diagnostic, create_image_analyzer,
                                   #          CameraConfig, Line1DConfig + all sub-models
    loader.py                      # YAML → typed model loaders
    factory.py                     # create_image_analyzer(AnalysisDiagnostic)
    registry.py                    # analyzer kind → implementing class
    diagnostic.py                  # re-export: AnalysisDiagnostic (+ DiagnosticAnalysisConfig alias)
    array2d_processing.py          # re-export: CameraConfig + 2D sub-models (geecs_schemas.analysis)
    array1d_processing.py          # re-export: Line1DConfig + 1D sub-models, to_data1d_config
  processing/
    array2d/
      background.py                # apply_background(image, config, *, cache=None)
      filtering.py                 # Gaussian / median filters
      masking.py                   # Crosshair and circular masking
      transforms.py                # Rotation, flip, distortion
      pipeline.py                  # apply_camera_processing_pipeline()
    array1d/
      background.py                # 1D background subtraction
      roi.py                       # 1D ROI (x-value range, not pixel)
      thresholding.py              # 1D thresholding
  analyzers/                       # Modern analyzer classes (was offline_analyzers/ pre-PR-E)
    __init__.py                    # Exports: StandardAnalyzer, Standard1DAnalyzer, BeamAnalyzer, LineAnalyzer
    standard_analyzer.py           # StandardAnalyzer — 2D foundation
    standard_1d_analyzer.py        # Standard1DAnalyzer — 1D foundation
    beam_analyzer.py               # BeamAnalyzer(StandardAnalyzer)
    line_analyzer.py               # LineAnalyzer(Standard1DAnalyzer)
    ict_1d_analyzer.py             # ICT1DAnalyzer(Standard1DAnalyzer)
    density_from_phase_analysis.py # DensityFromPhaseAnalyzer(ImageAnalyzer)
    line_stitcher.py               # LineStitcher
    magspec_manual_calib_analyzer.py
    grenouille_analyzer.py         # FROG / Grenouille
    HASO_himg_has_processor.py     # HASOHimgHasProcessor
    ...
  third_party_sdks/                # Vendor SDKs (WaveKit) — gitignored, out-of-tree install; see its README.md
```

## Core Abstractions

### `ImageAnalyzer` (base.py)

Abstract base class all analyzers implement.

```python
class ImageAnalyzer:
    run_analyze_image_asynchronously: bool = False

    def load_image(self, file_path: Path) -> Array1D | Array2D:
        # Default: read_imaq_image() — override for custom formats (TDMS, HDF5, CSV).
        # A geecs_data_utils ShotRef (capture frame stack + shot index)
        # resolves to that single frame before the default reader runs.

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

**The models live in GEECS-Schemas** (`geecs_schemas.analysis`, since
ImageAnalysis 2.0): the diagnostic document `AnalysisDiagnostic` (format
v2), the processing sections `CameraConfig` / `Line1DConfig`, the analyzer
spec union `AnalyzerSpec` (one typed spec per analyzer, discriminated on
`kind`), and the scan-runtime section `ScanRuntime`. They are pydantic-only,
so the portal, MCP, CI and the config editor validate a diagnostic without
this package. `image_analysis.config` re-exports them under the names the
processing code and analyzers use (`array2d_processing`,
`array1d_processing`, `diagnostic`) and owns the three things that need the
analysis stack:

- **`loader`** — `load_diagnostic(stem_or_path, config_dir=, overrides=)`
  → `AnalysisDiagnostic` (v1 files lift automatically);
  `load_camera_config` / `load_line_config` → the `image:` section of a
  diagnostic, or a bare section; `list_diagnostics`.
- **`factory`** — `create_image_analyzer(diag)`: resolves the class from
  `diag.analyzer.kind` and passes `spec=` / `camera_config=` /
  `line_config=` / `output_name=` where the constructor declares them.
- **`registry`** — `ANALYZER_CLASS_PATHS` (kind → class path, imported on
  demand so vendor SDKs stay unimported) and `analyzer_class(kind)`.
  `tests/test_config_registry.py` pins it against the schema's union.

A v2 diagnostic:

```yaml
schema_version: 2
name: UC_TopView                  # device folder under scans/ScanNNN/
output_name: UC_TopView_left      # optional output label (defaults to name)
analyzer: {kind: beam, compute_slopes: false}   # the analyzer + ITS parameters
image:                            # camera | line | omitted (haso, phase_downramp)
  type: camera
  bit_depth: 16
  roi: {x_min: 0, x_max: 650, y_min: 350, y_max: 650}
  background: {method: constant, constant_level: 5.0}
  pipeline: [background, roi]     # the bare list; only listed steps run
scan: {priority: 10, mode: per_shot, save: true, renderer: {cmap: plasma}}
```

`CameraConfig` / `Line1DConfig` are `extra="forbid"` down to every nested
section; a step runs only when listed in `pipeline` AND its section is
present. Scan-context backgrounds (`scan.background_source`) are resolved
by ScanAnalysis, which rewrites `image.background` to a static
`from_file` before per-shot processing. `Line1DConfig.data_loading` is the
schema `Data1DLoading`; `image_analysis.data_1d_utils.read_1d_data` (and
`config.array1d_processing.to_data1d_config`) hand it to GEECS-Data-Utils'
reader as its own `Data1DConfig`.

**Adding an analyzer** = one spec model in
`geecs_schemas.analysis.analyzers` (joined into `AnalyzerSpec`, with
`image_kind` = `"camera"` / `"line"` / `None`) + one line in
`ANALYZER_CLASS_PATHS` + a constructor that takes the spec:
`def __init__(self, camera_config, *, spec: MySpec | None = None,
output_name=None)`. Specs whose fields all have defaults may be optional
(notebook construction without one); specs with required fields are not.

## Analyzers

Analyzer constructors take **typed config models** (`CameraConfig` /
`Line1DConfig`) rather than string names. The string-by-name convenience
moved to the loader layer in PR-E. Two patterns:

```python
# Mode 1: direct construction for exploration
from image_analysis.config import load_camera_config
from image_analysis.analyzers.standard_analyzer import StandardAnalyzer

cfg = load_camera_config("UC_GaiaMode")
analyzer = StandardAnalyzer(camera_config=cfg)
```

```python
# Mode 2: config-driven factory (production scan path)
from image_analysis.config import load_diagnostic, create_image_analyzer

diag = load_diagnostic("UC_GaiaMode")          # → AnalysisDiagnostic (v1 files lift)
analyzer = create_image_analyzer(diag)         # → ImageAnalyzer instance
```

### `StandardAnalyzer` (2D foundation)

```python
analyzer = StandardAnalyzer(
    camera_config=cfg,       # typed CameraConfig (load via load_camera_config)
    output_name=None,        # output identifier; defaults to None in Mode-1
                             # (scalar keys are bare). Mode-2 factory passes
                             # diag.effective_output_name automatically.
)
```

Post-#412 the analyzer emits **bare scalar keys** (`"x_CoM"`, `"image_total"`,
…) regardless of `output_name`. ScanAnalysis is the sole layer that
applies the `output_name` prefix and `metric_suffix` to scalars when it
stores per-shot results. `output_name` is stored on the analyzer purely
so downstream consumers (output-dir labelling in `SingleDeviceScanAnalyzer`;
per-file paths in MagSpec) can read a stable identifier off the instance.

Key methods:
- `preprocess_image(image) -> np.ndarray` — applies full processing pipeline
- `analyze_image(image, auxiliary_data) -> ImageAnalyzerResult` — data_type="2d"
- `analyze_image_file(path, auxiliary_data)` — canonical scan-pipeline entry
- `render_image(result, vmin, vmax, cmap, ...) -> (Figure, Axes)` — static method
- `output_name` property — returns the configured output identifier (or `None`)

### `Standard1DAnalyzer` (1D foundation)

```python
analyzer = Standard1DAnalyzer(line_config=cfg)   # typed Line1DConfig
```

Key methods:
- `load_image(file_path)` — uses `read_1d_data()` not image reader; returns Nx2 array
- `preprocess_data(data)` — applies scale factors + line processing pipeline
- `analyze_image(image, ...) -> ImageAnalyzerResult` — data_type="1d"
- `render_image(result, ...)` — line plot with unit-aware axis labels

### `BeamAnalyzer(StandardAnalyzer)`

Adds beam-specific metrics (centroid, size, moments). Its parameters are
the `beam` spec (`BeamAnalyzerSpec`, alias `BeamAnalysisConfig`) passed as
`spec=`. Most commonly used 2D analyzer.

### `LineAnalyzer(Standard1DAnalyzer)`

Adds statistics: CoM, FWHM, RMS, peak analysis. Forwards `output_name` to
the Standard1D parent like every other analyzer in the family. For
"variant" use cases (e.g. before_foil vs after_foil) the diagnostic config
declares `metric_suffix` at the diagnostic layer; ScanAnalysis applies it
to all scalar keys.

### `ICT1DAnalyzer(Standard1DAnalyzer)`

Specialized for Integrated Current Transformer (charge measurement). Applies
Butterworth filter, calibration factor, time-step integration from the
`ict` spec (`IctAnalyzerSpec`, alias `ICTAnalysisConfig`).

### `DensityFromPhaseAnalyzer(ImageAnalyzer)`

Direct subclass of `ImageAnalyzer` (not Standard). Plasma density from wavefront
phase data — Abel inversion, background removal, rotation alignment, Gaussian
masking.

## Adding a New Analyzer

```python
# 1. geecs_schemas/analysis/analyzers.py — the spec (and add it to AnalyzerSpec)
class MyAnalyzerSpec(AnalyzerSpecBase):
    """One-line operator description."""
    image_kind: ClassVar[ImageKind] = "camera"
    kind: Literal["my_analyzer"] = Field("my_analyzer", description="...")
    threshold: float = Field(0.1, description="...")

# 2. image_analysis/config/registry.py — one line
#    "my_analyzer": "image_analysis.analyzers.my_analyzer.MyAnalyzer",

# 3. the class
from geecs_schemas.analysis import MyAnalyzerSpec
from image_analysis.analyzers.standard_analyzer import StandardAnalyzer
from image_analysis.config import CameraConfig
from image_analysis.types import ImageAnalyzerResult

class MyAnalyzer(StandardAnalyzer):
    def __init__(self, camera_config: CameraConfig, *, spec: MyAnalyzerSpec | None = None, output_name=None):
        super().__init__(camera_config=camera_config, output_name=output_name)
        self.spec = spec or MyAnalyzerSpec()

    def analyze_image(self, image, auxiliary_data=None) -> ImageAnalyzerResult:
        result = super().analyze_image(image, auxiliary_data)  # preprocessing + base result
        result.scalars["my_metric"] = compute_something(result.processed_image, self.spec)
        return result
```

1D analyzers inherit from `Standard1DAnalyzer`, take `line_config`, and
declare `image_kind = "line"`. Every field needs a `description=` (the
schema docgen test fails CI otherwise).

## Key Design Decisions

- **Processing happens in `preprocess_image()` / `preprocess_data()`** — `analyze_image()`
  receives already-processed data. Keep analysis logic separate from preprocessing.
- **Analyzer parameters are typed at the document layer** — the `analyzer:`
  spec union in GEECS-Schemas. A typo in a parameter name is a load error,
  not a silently ignored key (the pre-2.0 `analysis:` dict was validated
  leniently at `__init__`).
- **Scale factors applied first** — `x_scale_factor` / `y_scale_factor` run before
  ROI, so ROI boundaries and thresholds should be specified in scaled units.
- **Output naming lives at the diagnostic layer (#412)** — analyzers emit
  **bare** scalar keys; `AnalysisDiagnostic.output_name` and
  `metric_suffix` (read by ScanAnalysis) namespace them on the way to disk
  and in-memory consumers. See `ScanAnalysis/CLAUDE.md` for the full
  contract and the override use cases (output_name=UC_TopView_left vs
  output_name=UC_TopView_right for two variants of the same camera).
- **Nx2 convention for 1D data** — Column 0 is always x (independent), column 1
  is always y (dependent). `read_1d_data()` enforces this.

## Filesystem invariants for analyzers that write inside `scans/ScanNNN/`

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

## Ephemeral runs (the write-free contract)

`image_analysis.ephemeral.run_diagnostic_ephemeral(name_or_path, frames,
*, config_dir=..., overrides=..., auxiliary_data=...)` runs a configured
diagnostic over already-loaded frames with a hard no-writes guarantee —
the seam read-only viewers (the data portal's processing selector,
exploratory notebooks) use to get the production pipeline without the
production side effects. Its render form,
`render_diagnostic_ephemeral(..., window=, cmap=, figsize=, dpi=)`
(1.14.0), additionally draws each result with the analyzer's own
`render_image` into an **object-API `Figure`** — never pyplot, so it is
safe on a web server's threadpool — and adds the colorbar the base
renderer skips when handed an axes. The object-API helpers live in
`image_analysis.tools.rendering` (`new_figure`, `window_limits`,
`render_result_figure`, `render_frame_figure`, `RenderError`) next to
`base_render_image`; the ephemeral module only composes them with the
write gate. **`render_image` must keep honouring an `ax=` argument** —
pinned by `tests/test_ephemeral.py` for `StandardAnalyzer`,
`BeamAnalyzer`, `HiResMagCamAnalyzer`, `MagSpecManualCalibAnalyzer`
(2D `@staticmethod`s taking `vmin`/`vmax`/`cmap`) and
`Standard1DAnalyzer` (instance method taking plot kwargs). A renderer
that ignored `ax` would return an empty seam figure AND leak a
pyplot-registered figure per request on a server thread. Known
exception: `Undulator/BCaveMagSpecStitcher.py` keeps a legacy
`render_image(image, analysis_results_dict, …)` signature (and a
legacy dict `analyze_image` return) — through the seam either shape
ends as a `RenderError` (the dict reaches its renderer and fails
there); it predates the `ImageAnalyzerResult` contract and is not a
template. `render_frame_figure(image, …)` is the
base-renderer-only companion for images that are not one result (bin
averages).

The write gate is structural, and it depends on two conventions that
**must survive analyzer changes**:

1. **Path-gated writers stay path-gated.** Analyzers that persist
   derived per-shot files do so only when `auxiliary_data["file_path"]`
   (or equivalent constructor state) is present. The ephemeral runner
   takes in-memory frames only and refuses `file_path` in
   `auxiliary_data` (`ValueError`, never a silent strip) — so a
   path-gated writer is automatically dormant. If you add an analyzer
   with side effects on some *other* trigger — persisted files,
   transient temp files, subprocess spawns — gate them on `file_path`
   too, or add it to the denylist below.
2. **Analyzers with un-gated side effects go on `EPHEMERAL_DENYLIST`**
   (analyzer kinds, checked *before* the class is imported — which
   also keeps vendor SDK / DLL imports off hosts that lack them). Two
   current entries: **HASO** writes five sidecars per shot from
   `load_image` (instance state set there is what `analyze_image`
   packages, so the analyze-only ephemeral call would return a
   meaningless pass-through anyway, and the module hard-imports
   wavekit); **Grenouille**'s `analyze_image` unconditionally writes
   transient temp files and spawns a ~seconds 32-bit DLL subprocess per
   frame (cleaned up afterwards, but a per-request viewer must trigger
   neither). Remove an entry only when the analyzer gains an explicit
   ephemeral mode.

`list_diagnostics(config_dir=...)` (in `image_analysis.config`)
enumerates the loadable diagnostic IDs for pickers over the same tree.
