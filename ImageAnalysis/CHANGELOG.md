# Changelog — image-analysis

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.1] — 2026-05-21

### Fixed
- Offline analyzers that write output subfolders inside `scans/ScanNNN/` no
  longer auto-create the scan folder itself. Affected sites:
  `LineStitcher._save_stitched_output` (creates `<scan_dir>/<self.name>/`)
  and `MagSpecManualCalibAnalyzer._save_calibrated_outputs` (creates
  `<scan_dir>/<camera>-interp/` and `<scan_dir>/<camera>-interpSpec/`). All
  three previously used `mkdir(parents=True, exist_ok=True)`, which silently
  re-created the scan folder if it was briefly invisible on the share. On a
  new NetApp/domain this masked transient SMB visibility blips as permanent
  data loss: the analyzer planted an empty `ScanNNN/` directory entry over
  the real scan contents. Each site now verifies the scan folder is visible
  before creating its output subfolder; if it isn't, it raises
  `FileNotFoundError`. The task queue logs the failure and records
  `state="failed"` when the scan folder/status directory is still writable.
  If the entire scan folder is absent, status writes are skipped rather than
  recreating the folder; the user can rerun/relaunch once the share recovers.
  This pairs with the
  `scan_analysis` 1.3.6 fix for the same anti-pattern in `task_queue`.
- `processing.array1d.background.save_background_to_file` no longer creates
  parent directories. Same anti-pattern: `mkdir(parents=True, exist_ok=True)`
  could silently materialise a scan folder if a caller passed a destination
  inside `scans/ScanNNN/`. Callers must ensure the parent directory exists
  before calling; otherwise the function now raises `FileNotFoundError`.
  (The function isn't currently called from anywhere in the codebase; this
  is a defensive consistency fix to prevent future regressions.)

## [1.2.0] — 2026-05-20

### Added
- `LineAnalyzer` now accepts a `metric_prefix` constructor arg that overrides
  the `line_config.name`-derived prefix on scalar metric keys. Lets one
  `Line1DConfig` be reused across multiple analyzer instances that report
  under different names — e.g. a stitcher that loads from a per-device
  config but emits metrics under a composite name.
- 1D background subtraction `FROM_FILE` now supports every format the line
  loader understands (npy, csv, tsv, tek_scope_hdf5, tdms_scope). The
  background file is read via `read_1d_data` using the same `data_loading`
  config the line itself uses. Previously only `.npy` / `.npz` were
  accepted.

### Fixed
- `StandardAnalyzer.analyze_image_batch` no longer raises `AttributeError`
  when the camera config has no `background:` section (in which case
  `background_manager` is `None`).

### Changed (breaking — internal/config-level)
- `LineStitcher` constructor: `output_folder` and `output_label` kwargs are
  replaced by a single `name` kwarg that drives the metric prefix, the
  output subdirectory, and the output filename label. Existing YAML configs
  must update from `output_folder`/`output_label` to `name`.
- `compute_background` (in `processing/array1d/background.py`) now accepts
  a `data_loading: Optional[Data1DConfig]` argument, required for the
  `FROM_FILE` method. Pipeline callers already pass it through; direct
  callers of this helper must update.

## [1.1.5] — 2026-05-19

### Added
- `FrogRetrievalResult.tw_per_joule` property — peak power per unit energy
  (TW/J), computed as `1000 / (sum(temporal_intensity) * dt)` with `dt` in
  femtoseconds. Matches the LabVIEW Grenouille analysis scalar.
- `GrenouilleAnalyzer` now emits `{camera_name}_tw_per_joule` in its scalar
  results.

## [1.1.4] — 2026-05-12

### Changed
- `data_1d_utils._read_csv` (and `_read_tsv` via delegation) now use
  `numpy.loadtxt` instead of `numpy.genfromtxt`. NumPy 1.23+ ships a C-coded
  parser for `loadtxt` that is ~10–20× faster on clean numeric tables.
  Profiling against `Standard1DAnalyzer` / `LineAnalyzer` showed ~90% of
  per-shot wallclock was inside `genfromtxt`; this swap removes that
  bottleneck for multi-scan processing of 1D text-format data (notably
  interpolated-spectrum files).

  **Behavior note:** `loadtxt` is stricter on malformed rows — it raises
  `ValueError` rather than substituting NaN. For clean processed spectra
  this is the desired behavior; loud failures are preferred over silent
  NaNs. Files with intentionally missing/non-numeric values that previously
  parsed via `genfromtxt`'s coercion will now fail to load.
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
