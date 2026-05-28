# Image Analysis

The Image Analysis package provides per-shot processing and analysis for
camera images and 1D signal traces. A diagnostic is described entirely by a
typed YAML config — what to subtract, where to crop, what to threshold, what
analyzer to run — and the same config drives both interactive notebook use
and the automated `LiveWatch` runner that processes scans as they complete.

The framework is generic enough to wrap any 2D array (standard camera
frames, HASO wavefront-sensor `.himg` files, …) and any 1D signal (ICT
traces, scope captures, FROG spectra).

The fastest way to see it in action is the
[Analysis tutorial](../tutorials/analysis.md), which walks the canonical
ConfigFileGUI → group → LiveWatch loop end to end.

---

## How a diagnostic is described

Each diagnostic is one YAML file (one camera, one 1D signal). At the top
level it carries three sections:

```yaml
name: UC_TopView
image_analyzer: image_analysis.analyzers.beam_analyzer.BeamAnalyzer
image:
  type: camera          # or "line" for a 1D Line1DConfig
  bit_depth: 16
  roi: {x_min: 0, x_max: 650, y_min: 350, y_max: 650}
  background: {method: constant, constant_level: 5.0}
  thresholding: {method: constant, value: 0.0, mode: to_zero}
  pipeline:
    steps: [background, roi, thresholding]
scan:
  priority: 50
  mode: per_shot
```

- **`name`** is the diagnostic identifier (matches the device or signal
  name in the GEECS database).
- **`image_analyzer`** is the dotted path of the Python class that runs
  against each processed shot — `BeamAnalyzer`, `StandardAnalyzer`,
  `Standard1DAnalyzer`, custom subclasses, etc.
- **`image:`** is the typed Pydantic config for per-shot processing. The
  `type: camera | line` discriminator routes the dict to either a
  [`CameraConfig`](api/core_modules.md) or a `Line1DConfig`.
- **`scan:`** is the orchestration block consumed by the Scan Analysis
  side (priority, mode, output slot, render kwargs). Image Analysis
  itself ignores it — it's there because the same YAML file is the unit
  of analysis for both packages.

## Pipeline is the source of truth

The `image.pipeline.steps` list is the **only** thing that decides which
processing steps run, and in what order. The step config blocks
(`background:`, `roi:`, `thresholding:`, …) describe *how* each step
behaves; whether the step actually runs is purely a function of whether
it appears in `pipeline.steps`.

```yaml
image:
  background: {method: constant, constant_level: 5.0}    # config exists
  thresholding: {method: constant, value: 0.0}           # config exists
  filtering: {kernel_size: 3}                            # config exists
  pipeline:
    steps: [background, thresholding]                    # …but filtering doesn't run
```

The full set of step types (the values of `ProcessingStepType`):

| Step              | What it does |
|---|---|
| `background`      | Subtract a static, dynamic, or hybrid background |
| `roi`             | Crop to a rectangle (`x_min/x_max/y_min/y_max`) |
| `crosshair_masking` | Null out the crosshair pixels camera frames sometimes carry |
| `circular_mask`   | Zero pixels outside a circular aperture |
| `vignette`        | Apply a vignette correction frame |
| `thresholding`    | Constant / Otsu / percentile thresholding, `to_zero` or `binary` |
| `filtering`       | Median filter / Gaussian smoothing |
| `transforms`      | Geometric transforms (rotate, flip, scale) |
| `normalization`   | Normalise pixel values to a target range |

A step can appear multiple times in `steps` if you genuinely want it to
run twice (rare but supported); the order in `steps` is the order of
execution.

## Package layout

```
image_analysis/
├── base.py                      # ImageAnalyzer abstract base
├── analyzers/                   # Concrete analyzers (one per diagnostic family)
│   ├── beam_analyzer.py         #   BeamAnalyzer  — generic 2D beam profile
│   ├── standard_analyzer.py     #   StandardAnalyzer — pass-through with stats
│   ├── line_analyzer.py         #   Line1DAnalyzer / Standard1DAnalyzer
│   ├── grenouille_analyzer.py   #   FROG pulse characterisation
│   ├── HASO_himg_has_processor.py  # HASO wavefront sensor
│   └── …
├── config/                      # Pydantic models + loaders
│   ├── array2d_processing.py    #   CameraConfig + per-step configs
│   ├── array1d_processing.py    #   Line1DConfig + per-step configs
│   ├── diagnostic.py            #   DiagnosticAnalysisConfig (the YAML schema)
│   ├── factory.py               #   create_image_analyzer(config)
│   └── loader.py                #   load_diagnostic(path)
├── processing/                  # Pipeline runtime
│   ├── array2d/
│   │   ├── pipeline.py          #   Runs pipeline.steps in order
│   │   ├── background.py        #   Each step gets its own module …
│   │   ├── filtering.py
│   │   ├── masking.py
│   │   ├── thresholding.py
│   │   └── …
│   └── array1d/                 # Same shape for 1D signals
├── algorithms/                  # Scalar/array reductions of processed images
├── data/, data_1d_utils.py      # File-format loaders (.tsv, .png, .himg, .has, …)
└── tools/                       # Synthetic image generators (for tests/examples)
```

The architectural rule of thumb: **processing** turns image-in into
image-out, **algorithms** turn processed-image-in into scalar-or-1D-out,
**analyzers** wrap those into the framework's `ImageAnalyzer` interface and
return an `ImageAnalyzerResult` per shot.

## Running an analyzer

Three equivalent entry points, depending on how much typing you want to do:

```python
# 1. From a YAML diagnostic file (most common)
from image_analysis.config import load_diagnostic, create_image_analyzer

diag = load_diagnostic("scan_analysis_configs/analyzers/HTU/UC_TopView.yaml")
analyzer = create_image_analyzer(diag)
result = analyzer.analyze_image(my_image_array)

# 2. From a programmatically-built CameraConfig
from image_analysis.config import CameraConfig, BackgroundConfig
from image_analysis.analyzers.beam_analyzer import BeamAnalyzer

config = CameraConfig(name="my_cam", bit_depth=16)
analyzer = BeamAnalyzer(config)
result = analyzer.analyze_image(my_image_array)

# 3. From a fully synthetic image (for tests and notebooks)
from image_analysis.tools.synthetic_generators import gaussian_beam_2d
img = gaussian_beam_2d(shape=(128, 128), center=(64.0, 64.0), seed=0)
result = analyzer.analyze_image(img)
```

`result` is an `ImageAnalyzerResult` — a typed dataclass carrying the
processed image, per-shot scalars (centroid, RMS, peak, …), and renderable
metadata.

## Examples

| Notebook | What it covers |
|---|---|
| [Basic Offline Analysis](examples/basic_usage_image_analyzer.ipynb) | Load → process → analyze a single camera image end to end |
| [Basic Usage — 1D Analyzer](examples/basic_usage_1D_analyzer.ipynb) | The same flow for a 1D signal trace |
| [Grenouille Analysis](examples/grenouille_analysis.ipynb) | FROG pulse characterisation as a worked example |
| [HasoLift Analysis](examples/HasoLift_analysis.ipynb) | Reading `.himg` wavefront data |

## See also

- The [Analysis tutorial](../tutorials/analysis.md) — how ConfigFileGUI
  edits these configs and LiveWatch dispatches them at scan time.
- [Scan Analysis overview](../scan_analysis/overview.md) — how a
  diagnostic config is wrapped into a per-scan workflow (binning,
  summary figures, s-file appending).
- [Analyzer Index](analyzer_index.md) — pick the right analyzer for your
  diagnostic (beam profile, FROG, magspec, ICT, HASO, …).
- [API Reference](api/core_modules.md) — the `ImageAnalyzer` base class
  and `StandardAnalyzer` reference docs.

---

## Notes

!!! warning "LabVIEW PNG images"
    PNGs written by LabVIEW use a non-standard bit-shift that off-the-shelf
    image loaders mishandle. Use `read_imaq_png_image()` instead — the
    framework's loaders go through it automatically.

!!! tip "Keep configs in version control"
    Diagnostic YAML files belong in the `GEECS-Plugins-Configs` repo
    alongside this one. The whole point of the typed-config design is
    that experiments can pin a specific analysis revision and reproduce
    a scan's analysis months later.
