# Image Analysis

The Image Analysis package provides a toolkit for processing and analyzing camera data from BELLA experiments. Analysis is driven by YAML configuration files, making workflows reproducible and easy to share. The framework is generic enough to work with any 2D array data — standard camera images, HasoLift wavefront sensor data (`.himg`), and more.

The best way to get started is the [examples](#examples) below.

---

## Key Features

- **Configuration-Driven Analysis**: YAML-based camera configurations define the full processing pipeline — no hardcoded parameters in code.
- **Modular Processing Pipeline**: Composable steps including background subtraction, masking, filtering, geometric transforms, and thresholding.
- **Extensible Architecture**: Straightforward to add new analyzers or processing steps.
- **Multiple File Formats**: `.tsv`, `.hdf5`, `.png`, `.himg`, `.has`.
- **Integration with Scan Analysis**: `Array2DAnalysis` (in Scan Analysis) uses these analyzers to automate full workflows across scans, including binning, rendering, and s-file appending.
- **LabVIEW Integration**: Online analysis support via GEECS Point Grey Camera devices (see note below).

---

## Package Architecture

### `analyzers` vs `offline_analyzers`

`analyzers` is reserved for LabVIEW-compatible analyzers (Python 3.6, no virtual environments). `offline_analyzers` are for post-analysis of recorded data and have no compatibility restrictions. In practice, the vast majority of new development belongs in `offline_analyzers`.

### `processing` vs `algorithms` vs `utils`

These three modules have a clear division of responsibility:

- **`processing/`** — Takes an image in, returns an image out (background subtraction, filtering, masking, transforms).
- **`algorithms/`** — Takes a processed image in, returns scalar or 1D array results (beam profile fitting, centroid, FWHM, etc.).
- **`utils/`** — Data type translation utilities (e.g. path → image array).

### Core Components

- **`config_loader.py`** — Load and validate YAML camera configurations.
- **`processing/`** — Modular image processing operations.
- **`offline_analyzers/`** — High-level analysis classes for common workflows.
- **`base.py`** — Foundation classes for building custom analyzers.

### Ready-to-Use Processing Steps

| Module | What it does |
|---|---|
| `processing/background.py` | Static, dynamic, and hybrid background subtraction |
| `processing/masking.py` | Crosshair masking, ROI cropping, circular masks |
| `processing/filtering.py` | Noise reduction and image enhancement |
| `processing/transforms.py` | Geometric transformations and corrections |
| `processing/thresholding.py` | Intensity-based segmentation |

---

## Examples

| Notebook | What it covers |
|---|---|
| [Basic Offline Analysis](examples/basic_offline_analysis.ipynb) | End-to-end offline analysis with a standard analyzer |
| [Basic Usage — 1D Analyzer](examples/basic_usage_1D_analyzer.ipynb) | Using 1D array analyzers for profile data |
| [Grenouille Analysis](examples/grenouille_analysis.ipynb) | Pulse characterization with Grenouille data |
| [HasoLift Analysis](examples/HasoLift_analysis.ipynb) | Wavefront analysis with HasoLift `.himg` files |

---

## Notes

!!! warning "LabVIEW PNG Images"
    When working with PNG images saved by LabVIEW, always use `read_imaq_png_image()` instead of standard image loading libraries. LabVIEW saves PNG images with a unique bit-shift that causes unexpected scaling if not handled correctly.

!!! tip "Configuration Management"
    Store camera configuration YAML files in version control alongside your analysis scripts to ensure reproducible results and easy collaboration.

!!! note "LabVIEW Online Analysis"
    The earliest use of this package was for online analysis inside the Point Grey Camera LabVIEW driver. This is functional but limited — Python 3.6 only, no virtual environments, and the implementation can be unreliable over long runs. New development should target offline analysis unless online integration is specifically required.
