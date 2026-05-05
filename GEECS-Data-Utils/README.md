# Data Utils Module

## Scan table assembly (`geecs_data_utils.data.dataset`)

Non-ML helpers live next to `data.cleaning` and `data.columns`.

- **`DatasetBuilder.from_date_scan_numbers`** — one call: load many scan numbers for a date, concatenate scalar frames, optionally apply row filters / outlier config / `dropna`. Sets **`DatasetFrame.load_report`** with which numbers loaded vs skipped.
- **`DatasetBuilder.load_scans_from_date_report`** — same loading loop with explicit **`LoadScansReport`** (`scans`, `numbers_loaded`, `skipped` reasons). Use when you need visibility without building a table yet.
- **`DatasetBuilder.load_scans_from_date`** — returns only the list of loaded **`ScanData`** instances (default `on_missing="skip"`).

When concatenating scans whose scalar column sets differ, pandas may introduce **NaN columns** on some rows; see the module docstring in `geecs_data_utils/data/dataset.py` for details.

## Tabular analysis (`geecs_data_utils.analysis`)

- **`CorrelationReport`** (`analysis.correlation`) — rank numeric columns vs a target (Pearson / Spearman / Kendall), optional row filters via `data.cleaning.apply_row_filters`, substring exclusions, and `top_n` truncation.
