# Analyzer Index

This page is a discovery aid: pick what you're trying to measure, find the analyzer, follow it to the worked example. The full set of analyzers lives in `image_analysis/offline_analyzers/`.

If you don't see an analyzer for what you need, the right starting point for a new one is `StandardAnalyzer` (for 2D images) or `Standard1DAnalyzer` (for 1D traces and lineouts) — both are designed to be subclassed.

## Analyzers by purpose

### Beam profile, position, size

**`BeamAnalyzer`** (`offline_analyzers/beam_analyzer.py`) — beam centroid, width, height, FWHM, optional slope/straightness metrics. Renders an annotated beam image with overlays. The standard tool for camera images of a beam profile.

Inherits from `StandardAnalyzer` and adds beam-specific outputs while delegating image processing (background subtraction, masking, filtering) to the base class.

### Pulse characterization (FROG / Grenouille)

**`GrenouilleAnalyzer`** (`offline_analyzers/grenouille_analyzer.py`) — FROG pulse retrieval via the FrogDll backend. Outputs temporal and spectral FWHM, retrieved trace, and lineout exports. The right tool for a Grenouille trace where you want a pulse duration estimate.

Worked example: [Grenouille Analysis notebook](examples/grenouille_analysis.ipynb).

### Magnetic spectrometer (energy spectra)

**`MagSpecManualCalibAnalyzer`** (`offline_analyzers/magspec_manual_calib_analyzer.py`) — magnetic spectrometer images with pixel-to-energy conversion using device-specific calibrations. Configured via YAML with `analysis.energy_range` and per-device calibration parameters.

### 1D line profiles, ICT charge traces

**`LineAnalyzer`** (`offline_analyzers/line_analyzer.py`) — center of mass, FWHM, RMS width, peak analysis, integrated signal for 1D profile data. Unit-aware reporting.

**`ICT1DAnalyzer`** (`offline_analyzers/ict_1d_analyzer.py`) — ICT charge measurement on oscilloscope voltage traces. Uses the `ict_algorithms` module. Configured by adding `ict_analysis_params` to the device's YAML config.

**`LineStitcher`** (`offline_analyzers/line_stitcher.py`) — for the case where multiple devices each cover a portion of a shared physical axis (e.g. magspec1 + magspec2 + magspec3 covering different energy ranges). Concatenates and sorts the per-device files into one analysis.

### Wavefront / phase

**`HASOHimgHasProcessor`** (`offline_analyzers/HASO_himg_has_processor.py`) — loads HASO `.himg` / `.has` files, applies masking and background subtraction, computes phase via zonal reconstruction. Saves slopes, phases, and intensity alongside the source. Requires WaveKit 4.3 (Windows-only at runtime).

Worked example: [HasoLift Analysis notebook](examples/HasoLift_analysis.ipynb).

**`DownrampPhaseAnalyzer`** (`offline_analyzers/downramp_phase_analyzer.py`) — plasma downramp shock analysis from phase data. Shock angle estimation, gradient and position detection, plateau and peak-to-plateau delta calculation. Output is a combined diagnostic figure as vector PDF.

**`PhaseDownrampProcessor`** (in `density_from_phase_analysis.py`) — class-based phase-map → plasma-density pipeline using PyAbel. Includes utilities for background removal, cropping, rotation alignment, Gaussian masking, and thresholding.

### Generic / starting points

**`StandardAnalyzer`** (`offline_analyzers/standard_analyzer.py`) — general-purpose 2D image analyzer with YAML config, Pydantic-validated parameters, and a modular processing pipeline (background subtraction, masking, filtering, transforms, thresholding). The parent class for most specialized 2D analyzers.

**`Standard1DAnalyzer`** (`offline_analyzers/standard_1d_analyzer.py`) — equivalent for 1D traces, spectra, and lineouts. Parent class for `LineAnalyzer` and `ICT1DAnalyzer`.

If you're writing a new analyzer, start by inheriting from one of these. The `StandardAnalyzer` docstring documents the pipeline hooks; the [Image Analysis Overview](overview.md) shows the broader architecture.

## What each analyzer needs

Every analyzer takes a YAML config (one per device) describing its processing pipeline and analysis parameters. Configs are typically stored alongside your experiment configuration so they're version-controlled with your scan setup. The general structure:

```yaml
device: U_DeviceName
processing:
  background:
    method: dynamic
    # ...
  masking:
    # ...
analysis:
  # analyzer-specific parameters
```

The exact shape of the `analysis` block depends on the analyzer. The simplest way to learn the schema is to look at an existing config or run the analyzer once and let the Pydantic validation tell you what it expects.

## When an analyzer is part of a scan

When `Array2DScanAnalyzer` or `Array1DScanAnalyzer` (in `scan_analysis`) wraps an image analyzer, the analyzer runs once per shot in every bin, the per-shot results get aggregated to per-bin scalars, and a summary figure is rendered for the whole scan. See the [Scan Analysis](../scan_analysis/overview.md) page for that wrapping pattern.

This is also how analyzer outputs end up appended to the scan's s-file — the scan analyzer takes the per-bin results from the image analyzer and writes them as new columns that appear next to the device variables that were originally recorded.

## Discoverability tip

Run `python -c "import image_analysis.offline_analyzers as oa; help(oa)"` to see the full list of analyzers in your installed version. The package's `__init__.py` re-exports them; the help output is a fast way to confirm what's available without browsing source.
