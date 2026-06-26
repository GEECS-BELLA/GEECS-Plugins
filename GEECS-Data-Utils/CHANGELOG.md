# Changelog — geecs-data-utils

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.9.1] — 2026-06-26

### Changed
- Dropped the Python 3.10 support claim; minimum is now `python >=3.11,<3.12`,
  matching the integrated monorepo environment (the root project and the
  GUI/PythonAPI/Bluesky packages all require >=3.11).

## [0.9.0] — 2026-06-18

### Added
- New `geecs_data_utils.io` subpackage owning generic `path -> numpy.ndarray`
  file readers, relocated from `image_analysis.utils`:
  - `geecs_data_utils.io.images.read_imaq_image` (format dispatcher),
    `read_imaq_png_image` (NI IMAQ 16-bit PNG), `read_tsv_file`,
    `load_image_from_h5`.
  - Gives ImageAnalysis, post-run analysis tools, and Bluesky external-asset
    handlers a shared reader foundation without depending on the
    `image_analysis` package.
- New hard dependencies for the readers: `pypng`, `imageio`, `h5py` (kept
  light; heavier image libs such as opencv / scikit-image deliberately stay
  out of this package).

### Notes
- These readers are consumer-only file loaders; they read existing files and
  never create scan folders (cross-package scan-folder invariant unaffected).

## [0.8.0] — 2026-06-15

### Added
- New `geecs_data_utils.tiled_export` module: reads a Bluesky scan back from a
  Tiled catalog and writes the legacy GEECS scalar files
  (`scans/ScanNNN/ScanDataScanNNN.txt` and the mutable `analysis/sNNN.txt`).
  - `write_scalar_files_from_tiled(uid, ...)` — fetch a run by uid and write
    both files; resolves Tiled connection from `~/.config/geecs_python_api/
    config.ini [tiled]` when not passed explicitly.
  - `build_legacy_scalar_dataframe(start_doc, primary_df)` — the pure transform
    (renames Bluesky `<ophyd>-<safe_var>` columns to legacy `Device Variable`
    via the run's `geecs_scalar_headers`, drops companion columns, emits
    `Bin #` / `scan` / `Shotnumber`), unit-testable without a live server.
  - Consumer-only: writes into an existing `scans/ScanNNN/` folder, never
    creates one (cross-package scan-folder invariant).
- New optional `tiled` extra (`pip install 'geecs-data-utils[tiled]'`); the
  module lazy-imports the Tiled client so the dependency is only needed for
  export.

## [0.7.0] — 2026-05-20

### Added
- New `geecs_data_utils.data` subpackage: shared tabular utilities used by
  both analysis and modeling layers.
  - `data.columns`: `find_cols`, `resolve_col`, `resolve_col_detailed`,
    `flatten_columns`, `ColumnMatchMode`, `ResolveColResult`.
  - `data.cleaning`: `RowFilterSpec`, `apply_row_filters`, `OutlierConfig`,
    `apply_outlier_config`, `sigma_clip_frame`, `sigma_nan_frame`.
  - `data.dataset`: `DatasetBuilder`, `DatasetFrame`, `LoadScansReport`
    for multi-scan scalar dataset assembly with filters / outliers /
    `dropna` and a visibility report for skipped scans.
- New `geecs_data_utils.analysis` subpackage:
  - `analysis.correlation`: `CorrelationReport` for target-vs-numeric
    correlation ranking (Pearson / Spearman / Kendall) with row filters,
    substring exclusions, and `top_n`.
- New optional `geecs_data_utils.modeling.ml` subpackage (install with the
  `ml` extra):
  - `MLDatasetBuilder` / `DatasetResult`: select target + features from a
    DataFrame for modeling, with optional `exclude_terms` for substring-
    based feature pruning (matching `CorrelationReport.exclude_terms`).
  - `RegressionTrainer` / `ModelArtifact`: linear / ridge / elastic-net
    fits with standard preprocessing, metrics, and optional CV scores.
  - `save_model_artifact` / `load_model_artifact`: joblib + JSON
    sidecars (`FeatureSchema`, `ModelMetadata`, `TrainingMetrics`).
    Metadata captures `sklearn_version`, `joblib_version`, `numpy_version`,
    `python_version`, and an `artifact_version` so loaders can warn on
    runtime mismatches.
  - `predict_from_scan`: inference helper that expects scan columns to
    match the training feature schema exactly.

### Changed
- `ScanData.find_cols` / `resolve_col` now delegate to
  `geecs_data_utils.data.columns` so single-scan and multi-scan code paths
  share semantics. Behavior is preserved.

### Removed
- Unused `ScanPaths.data_dict`, `ScanPaths.data_frame`, and
  `ScanPaths.get_device_data()`. No external callers within the monorepo.
## [0.6.4] — 2026-05-21

### Changed
- `ScanPaths` `read_mode` docstring tightened to document that
  `read_mode=False` (silent folder creation) is for scanner-side callers
  only — the GEECS scanner and BlueskyScanner, which legitimately bring new
  scan folders into existence. Analysis-side callers (ScanAnalysis,
  ImageAnalysis) must use the default `read_mode=True`. Behaviour is
  unchanged; the contract is now pinned by
  `tests/test_scan_paths_create_invariant.py`. Context: a sibling fix in
  `scan_analysis` 1.3.6 removed an analysis-side silent-create that
  converted transient SMB visibility blips into data loss.

## [0.6.3] — 2026-05-19

### Removed
- `EXPERIMENT_TO_SERVER_DICT` and the associated `_get_default_server_address` /
  `_is_default_server_address` helpers removed from `GeecsPathsConfig`. The dict
  was an implicit, hard-coded mapping of experiment names to server paths that
  silently overrode explicit config, caused confusion when paths differed between
  sites, and is now fully superseded by `GEECS_DATA_LOCAL_BASE_PATH` in
  `config.ini`. Any machine previously relying on the implicit `Z:/data` default
  should add `GEECS_DATA_LOCAL_BASE_PATH = Z:/data` to its config.

### Changed
- `ScanData.from_date` and `ScanData.latest`: `experiment` parameter is now
  `Optional[str]` (was `str`). Callers that pass `None` propagate to
  `ScanPaths.get_scan_tag`, which already handles `None` by falling back to
  `paths_config.experiment`; flat-layout sites can omit the experiment entirely.

## [0.6.2] — 2026-05-19

### Fixed
- `ScanPaths.get_daily_scan_folder`: skips the experiment path segment when
  `tag.experiment` is `None`, producing `{base}/Y{YYYY}/...` instead of
  crashing.

## [0.6.1] — 2026-05-19

### Changed
- `GeecsPathsConfig`: `GEECS_DATA_LOCAL_BASE_PATH` from `config.ini` is now
  tried **before** the experiment-to-server-address dict (`EXPERIMENT_TO_SERVER_DICT`),
  which becomes a fallback. This means analysis-only machines that define a
  local data root are no longer overridden by the `Z:/data` server default.
- `GeecsPathsConfig`: `experiment` is now optional — a `ConfigurationError` is
  only raised when `base_path` cannot be determined. Callers that need the
  experiment name (e.g. LiveWatch, GDoc integration) supply it at runtime via
  `ScanTag`; it no longer needs to be defined in `config.ini`.
- `_get_default_server_address` signature updated to accept `Optional[str]`.

## [0.6.0] — 2026-05-12

### Added
- `ScanPaths.build_device_file_map`, `ScanPaths.build_asset_filename`, and
  `ScanPaths.build_asset_path` accept an optional `device_file_stem` kwarg
  (default `None` → falls back to `device`). Use this when a device's data
  folder name differs from the token used inside per-shot filenames — e.g.,
  folder `U_BCaveMagSpec-interpSpec` containing files named
  `Scan042_U_BCaveMagSpec_001.csv`.
- `ScanData._append_expected_asset_columns`, `ScanData.set_data_frame`,
  `ScanData.load_scalars`, and `ScanData.from_date` accept an optional
  `stem_override: dict[str, str]` kwarg, mirroring the existing
  `ext_override` pattern. Maps device folder names to their in-filename
  stems so the `<device>_expected_path` DataFrame columns resolve to real
  files for affected devices. Without the override, those columns
  previously contained nonexistent paths for any device where the folder
  name and filename stem differ.

## [0.5.0] — 2026-05-08

### Added
- `timestamp_from_string(string)` and `timestamp_from_filename(file)` migrated
  from `geecs_python_api.tools.files.timestamping`. Both are exported from the
  package root. Eliminates the scanner's dependency on the now-deleted
  `GEECS-PythonAPI` timestamping module.
- `tests/test_utils.py` — first test suite for `geecs_data_utils`.

## [0.4.1] — 2026-05-07

### Changed
- `ScanConfig` migrated from `@dataclass` to `pydantic.BaseModel`.
  Construction syntax is unchanged (all fields use keyword arguments); the
  migration adds runtime validation and makes `ScanConfig` composable with other
  Pydantic models throughout the scanner engine.

## [0.4.0] — 2026-05-07

### Added
- `scan_log_loader` module providing `LogEntry`, `Severity`, `parse_lines`,
  `parse_scan_log`, and `load_scan_log`. Reads the per-scan log format
  written by `geecs_scanner.logging_setup.attach_scan_log` (multi-line
  tracebacks aggregated into the preceding record). Returned models are
  shared with the new `geecs-log-triage` subpackage and intended for any
  consumer needing to read scan logs (notebooks, plotting helpers,
  diagnostics tooling).

## [0.3.0] — 2026-05-06

### Added
- `GeecsPathsConfig` now reads an optional `wavekit_config_path` key from the
  `[Paths]` section of `config.ini` and exposes it as an attribute (consistent
  with the existing `frog_dll_path` / `frog_python32_path` pattern). Returns
  `None` if the key is absent or the path does not exist.

## [0.2.1] — current
<!-- Add entries here when changes are made -->
