# Analyzer Index

This page is a discovery aid: pick what you're trying to measure, find the analyzer, follow it to the worked example. The full set of analyzers lives in `image_analysis/analyzers/`.

If you don't see an analyzer for what you need, the right starting point for a new one is `StandardAnalyzer` (for 2D images) or `Standard1DAnalyzer` (for 1D traces and lineouts) — both are designed to be subclassed.

## Analyzers by purpose

### Beam profile, position, size

**`BeamAnalyzer`** (`analyzers/beam_analyzer.py`) — beam centroid, width, height, FWHM, optional slope/straightness metrics. Renders an annotated beam image with overlays. The standard tool for camera images of a beam profile.

Inherits from `StandardAnalyzer` and adds beam-specific outputs while delegating image processing (background subtraction, masking, filtering) to the base class.

### Pulse characterization (FROG / Grenouille)

**`GrenouilleAnalyzer`** (`analyzers/grenouille_analyzer.py`) — FROG pulse retrieval via the FrogDll backend. Outputs temporal and spectral FWHM, retrieved trace, and lineout exports. The right tool for a Grenouille trace where you want a pulse duration estimate.

Worked example: [Grenouille Analysis notebook](examples/grenouille_analysis.ipynb).

**`FrogSpectralPhaseAnalyzer`** (`analyzers/frog_spectral_phase_analyzer.py`) — consumes the retrieved lineout TSV files written by `GrenouilleAnalyzer`, fits the spectral phase as a polynomial in angular-frequency detuning, and reports dispersion terms (GD, GDD, TOD).

### Magnetic spectrometer (energy spectra)

**`MagSpecManualCalibAnalyzer`** (`analyzers/magspec_manual_calib_analyzer.py`) — magnetic spectrometer images with pixel-to-energy conversion using device-specific calibrations. Configured via YAML with `analysis.energy_range` and per-device calibration parameters.

### 1D line profiles, ICT charge traces

**`LineAnalyzer`** (`analyzers/line_analyzer.py`) — center of mass, FWHM, RMS width, peak analysis, integrated signal for 1D profile data. Unit-aware reporting.

**`ICT1DAnalyzer`** (`analyzers/ict_1d_analyzer.py`) — ICT charge measurement on oscilloscope voltage traces. Uses the `ict_algorithms` module. Configured by adding `ict_analysis_params` to the device's YAML config.

**`LineStitcher`** (`analyzers/line_stitcher.py`) — for the case where multiple devices each cover a portion of a shared physical axis (e.g. magspec1 + magspec2 + magspec3 covering different energy ranges). Concatenates and sorts the per-device files into one analysis.

### Wavefront / phase

**`HASOHimgHasProcessor`** (`analyzers/HASO_himg_has_processor.py`) — loads HASO `.himg` / `.has` files, applies masking and background subtraction, computes phase via zonal reconstruction. Saves slopes, phases, and intensity alongside the source. Requires WaveKit 4.3 (Windows-only at runtime).

Worked example: [HasoLift Analysis notebook](examples/HasoLift_analysis.ipynb).

**`DownrampPhaseAnalyzer`** (`analyzers/downramp_phase_analyzer.py`) — plasma downramp shock analysis from phase data. Shock angle estimation, gradient and position detection, plateau and peak-to-plateau delta calculation. Output is a combined diagnostic figure as vector PDF.

**`PhaseDownrampProcessor`** (in `analyzers/density_from_phase_analysis.py`) — class-based phase-map → plasma-density pipeline using PyAbel. Includes utilities for background removal, cropping, rotation alignment, Gaussian masking, and thresholding.

### Generic / starting points

**`StandardAnalyzer`** (`analyzers/standard_analyzer.py`) — general-purpose 2D image analyzer with YAML config, Pydantic-validated parameters, and a modular processing pipeline (background subtraction, masking, filtering, transforms, thresholding). The parent class for most specialized 2D analyzers.

**`Standard1DAnalyzer`** (`analyzers/standard_1d_analyzer.py`) — equivalent for 1D traces, spectra, and lineouts. Parent class for `LineAnalyzer` and `ICT1DAnalyzer`.

If you're writing a new analyzer, start by inheriting from one of these. The `StandardAnalyzer` docstring documents the pipeline hooks; the [Image Analysis Overview](overview.md) shows the broader architecture.

## What each analyzer needs

Every analyzer takes a YAML config (one per device) describing its processing pipeline and analysis parameters. Configs are typically stored alongside your experiment configuration so they're version-controlled with your scan setup. A diagnostic config typically has an `image:` section (the per-shot processing pipeline, `type: camera` or `type: line`) and a `scan:` section (how the analyzer runs at the scan orchestration level) — HASO-style analyzers omit `image:` and pass constructor kwargs through the verbose `image_analyzer:` form instead. The skeleton:

```yaml
name: U_DeviceName
image_analyzer: image_analysis.analyzers.beam_analyzer.BeamAnalyzer
image:
  type: camera   # or "line" for a 1D signal
  # processing step blocks (roi, background, ...) + pipeline.steps
  analysis:
    # analyzer-specific parameters
scan:
  priority: 50
  mode: per_shot
```

The annotated version of this file — every section explained, with a full worked example — is in the [Image Analysis Overview](overview.md#how-a-diagnostic-is-described); the [Analysis tutorial](../tutorials/analysis.md) walks through editing one field by field. The exact shape of the `analysis` block depends on the analyzer — look at an existing config or run the analyzer once and let the Pydantic validation tell you what it expects.

## When an analyzer is part of a scan

When `Array2DScanAnalyzer` or `Array1DScanAnalyzer` (in `scan_analysis`) wraps an image analyzer, the analyzer runs once per shot in every bin, the per-shot results get aggregated to per-bin scalars, and a summary figure is rendered for the whole scan. See the [Scan Analysis](../scan_analysis/overview.md) page for that wrapping pattern.

This is also how analyzer outputs end up appended to the scan's s-file — the scan analyzer takes the per-bin results from the image analyzer and writes them as new columns that appear next to the device variables that were originally recorded.

## Discoverability tip

Run `python -c "import image_analysis.analyzers as a; help(a)"` to see the analyzers your installed version re-exports. The package's `__init__.py` re-exports the most commonly used classes; the more specialized analyzers (ICT, MagSpec, Grenouille, HASO, phase) live in their own modules under `image_analysis/analyzers/` and are imported from there directly.
