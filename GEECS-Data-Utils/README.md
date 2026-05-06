# GEECS Data Utils

Utilities for **GEECS experimental data**: scan folder layout, loading TDMS/scalar tables into pandas, path/config discovery, and shared helpers used across GEECS plugins.

Common entry points are re-exported from `geecs_data_utils` (see `__all__` in `geecs_data_utils/__init__.py`). Subpackages below cover newer shared **data** / **analysis** layers and optional **modeling**.

## Core scan handling (`scan_paths`, `scan_data`)

- **`ScanPaths`** (`geecs_data_utils.scan_paths`) — Locate a scan directory from a filesystem path or from a **`ScanTag`** (experiment + date + scan number) plus optional base directory; read scan metadata (info files, naming conventions). Owns path navigation; does **not** hold loaded tabular data.
- **`ScanData`** (`geecs_data_utils.scan_data`) — Extends **`ScanPaths`** with **data loading**: TDMS (`nptdms`), scalar/`sfile` ingestion into **`data_frame`**, column helpers (`find_cols`, **`resolve_col`** delegating to **`geecs_data_utils.data.columns`**), binning/aggregation helpers, and related scan-table utilities.

Typical pattern: construct **`ScanData`** (or **`ScanPaths`**) for one scan, then read scalars or TDMS as needed. Multi-scan tables are often built with **`DatasetBuilder`** (see below).

## Paths & configuration (`geecs_paths_config`, `config_base`, `config_roots`, `utils`)

- **`GeecsPathsConfig`** — Experiment-aware defaults for where GEECS data and plugin config roots live (INI-backed / deployment-specific behavior; see module docstring).
- **`ConfigDirManager`** (`config_base`) — Reusable “config directory + cached lookups” helper (env var bootstrap, optional fallback resolver).
- **`image_analysis_config`** / **`scan_analysis_config`** (`config_roots`) — Pre-wired **`ConfigDirManager`** instances for ImageAnalysis and ScanAnalysis config trees (documented env vars and fallbacks in that module).
- **`SysPath`**, **`ConfigurationError`**, **`month_to_int`** (`utils`) — Shared typing for path-like values, configuration failures, and month parsing used by scan path logic.

## Scan identifiers & modes (`type_defs`)

- **`ScanTag`** — Structured experiment/date/scan-number identifier (pydantic model).
- **`ScanMode`**, **`ScanConfig`** — Enumerations / configs used with GEECS scanner GUIs and scan descriptions (see module for fields).

## Scans metadata database (`scans_database`)

- **`ScanDatabase`** — Query a **Hive-partitioned Parquet** scan-metadata dataset (year/month pruning, filters, optional YAML-named presets). Depends on **pyarrow**/pandas; aimed at interactive or batch discovery over large archives rather than single-scan folder access.

## Plotting helpers (`plotting_utils`)

- **`plot_binned`**, **`plot_binned_multi`** — Minimal matplotlib helpers for **binned** scalar frames (MultiIndex columns with `center` / optional asymmetric errors). Companion to binned outputs produced in **`scan_data`**.

---

## Shared data utilities (`geecs_data_utils.data`)

Column matching (`resolve_col`, `find_cols`, …), row filters / outlier helpers, and generic scan/DataFrame assembly live here. Import from `geecs_data_utils.data` or use the submodule paths below.

## Scan table assembly (`geecs_data_utils.data.dataset`)

Non-ML helpers sit alongside `data.cleaning` and `data.columns` (see above).

- **`DatasetBuilder.from_date_scan_numbers`** — one call: load many scan numbers for a date, concatenate scalar frames, optionally apply row filters / outlier config / `dropna`. Sets **`DatasetFrame.load_report`** with which numbers loaded vs skipped.
- **`DatasetBuilder.load_scans_from_date_report`** — same loading loop with explicit **`LoadScansReport`** (`scans`, `numbers_loaded`, `skipped` reasons). Use when you need visibility without building a table yet.
- **`DatasetBuilder.load_scans_from_date`** — returns only the list of loaded **`ScanData`** instances (default `on_missing="skip"`).

When concatenating scans whose scalar column sets differ, pandas may introduce **NaN columns** on some rows; see the module docstring in `geecs_data_utils/data/dataset.py` for details.

## Tabular analysis (`geecs_data_utils.analysis`)

- **`CorrelationReport`** (`analysis.correlation`) — rank numeric columns vs a target (Pearson / Spearman / Kendall), optional row filters via `data.cleaning.apply_row_filters`, substring exclusions, and `top_n` truncation.

## Modeling (`geecs_data_utils.modeling`)

The **`modeling`** package is reserved for higher-level “fit / persist / predict” style workflows. **`modeling.ml`** holds sklearn-oriented pieces (dataset shaping for regression, trainers, artifact save/load, inference helpers). That code paths through **`geecs_data_utils.data`** for assembly and column logic; it is **optional** (install with the Poetry extra **`ml`** / scikit-learn where applicable).

**Status:** this ML-oriented surface is **currently unused and unvetted** in production GEECS workflows—treat it as experimental if you import from `geecs_data_utils.modeling` or `geecs_data_utils.modeling.ml`.
