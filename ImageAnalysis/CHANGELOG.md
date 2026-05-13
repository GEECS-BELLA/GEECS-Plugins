# Changelog — image-analysis

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.4] — 2026-05-12

### Fixed
- `BackgroundConfig` now accepts `method="from_file"` with `background_scan_number`
  set and no explicit `file_path`.  Previously the `@field_validator("file_path")`
  ran before `background_scan_number` was validated (Pydantic processes fields
  top-to-bottom), so it always raised an error when only a scan number was given.
  Replaced with a `@model_validator(mode="after")` that checks both fields after
  full model construction.

## [1.1.3] — 2026-05-06

### Fixed
- `GrenouilleAnalyzer.analyze_image` no longer raises `AttributeError` when
  `auxiliary_data` is `None` (i.e. when called via `analyze_image_file` without
  passing auxiliary data). The TSV lineout export is now correctly skipped in
  that case.

### Added
- `tests/conftest.py` — session-scoped autouse fixture that initialises the
  image-analysis config base directory; no-op on CI.
- `tests/analyzers/test_grenouille_analyzer.py` — integration tests for
  `GrenouilleAnalyzer` covering scalar keys, physical plausibility of FWHM and
  FROG error, result structure, and sidecar TSV creation. Config is frozen in
  code to avoid YAML drift.
- `tests/analyzers/test_haso_analyzer.py` — integration tests for
  `HASOHimgHasProcessor` covering result structure and all five sidecar file
  outputs. Entire module skips when WaveKit SDK is unavailable.

## [1.1.2] — 2026-05-06

### Fixed
- `BeamAnalyzer` now reports `x_CoM`, `y_CoM`, and `peak_location` in the
  full-sensor coordinate system rather than in the ROI-local frame.
  `beam_profile_stats` gains an `roi_offset` parameter; `BeamAnalyzer`
  passes `(roi.x_min, roi.y_min)` automatically when an ROI is configured.
  Width stats (`rms`, `fwhm`) are unaffected by the offset.
- `apply_roi_cropping` no longer raises `ValueError` when the configured ROI
  extends beyond the actual image dimensions. Boundaries are now clamped to
  the image size with a `WARNING` log message. If clamping leaves a zero-area
  ROI the full image is returned.

## [1.1.1] — 2026-05-06

### Removed
- `lcls-tools` dependency (closes #231). The package was not used in any active
  code paths; the one internal helper (`gaussian_fit_beam_size`) has been
  rewritten using `scipy.optimize.curve_fit`, which is already a dependency.
  `image_analysis/algorithms/lcls_tools_gauss_fit.py` (a thin wrapper that was
  never imported) has been deleted.

## [1.1.0] — current
