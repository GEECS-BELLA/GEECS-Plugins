# `geecs_data_utils.ml` — Scan-to-Beam Regression Framework

A lightweight Python package for building, training, saving, and deploying regression models on GEECS experimental scan data.

## Installation

The ML subpackage requires the optional `ml` extras:

```bash
pip install geecs-data-utils[ml]
```

This pulls in `scikit-learn` (and optionally `optuna` for future tuning support).

## Shared data utilities (non-ML)

Multi-scan assembly and cleaning live in `geecs_data_utils.data.dataset` (see the package `README.md`). Typical flow: `DatasetBuilder.from_date_scan_numbers(...)` for a one-shot table plus `DatasetFrame.load_report` for skipped scan numbers, or `DatasetBuilder.load_scans_from_date_report` if you only need load diagnostics before concatenation.

Correlation ranking (`CorrelationReport`) is implemented in `geecs_data_utils.analysis.correlation` and re-exported from `geecs_data_utils.ml` for convenience; prefer `from geecs_data_utils.analysis import CorrelationReport` in new code.

## Quick Start

```python
from geecs_data_utils import ScanData
from geecs_data_utils.ml import (
    BeamPredictionDatasetBuilder,
    CorrelationReport,
    RegressionTrainer,
    save_model_artifact,
    load_model_artifact,
)

# 1. Load scan data
scan = ScanData.from_date(year=2026, month=2, day=18, number=1, experiment="Undulator")

# 2. Build an ML-ready dataset
ds = BeamPredictionDatasetBuilder.from_scan(
    scan,
    feature_specs=["laser", "pressure", "timing"],
    target_specs=["charge"],
)

# 3. Rank features by correlation
report = CorrelationReport.from_dataframe(ds.frame, target="charge", method="spearman")
top_features = report.top_features(n=10)

# 4. Train a regression model
trainer = RegressionTrainer(model="ridge", model_params={"alpha": 1.0})
artifact = trainer.fit(
    ds.frame[top_features],
    ds.frame["charge"],
    target_name="charge",
    cv=5,
)
print(f"R² = {artifact.metrics.r2:.3f}")
print(f"CV R² = {artifact.metrics.cv_r2_mean:.3f} +/- {artifact.metrics.cv_r2_std:.3f}")

# 5. Save the trained model
save_model_artifact(artifact, "models/charge_ridge_v1")

# 6. Load and predict later
loaded = load_model_artifact("models/charge_ridge_v1")
predictions = loaded.predict(new_data[top_features])
```

## Package Structure

```
geecs_data_utils/analysis/
    __init__.py
    correlation.py         # CorrelationReport (tabular correlation ranking)

geecs_data_utils/ml/
    __init__.py            # Public API exports
    dataset.py             # BeamPredictionDatasetBuilder, DatasetResult, OutlierConfig
    feature_selection.py   # re-exports CorrelationReport from analysis
    preprocessing.py       # build_preprocessing_pipeline (internal)
    models.py              # RegressionTrainer, ModelArtifact
    persistence.py         # save_model_artifact, load_model_artifact
    inference.py           # predict_from_scan helper
    schemas.py             # FeatureSchema, ModelMetadata, TrainingMetrics
```

## API Reference

### Dataset Construction

#### `BeamPredictionDatasetBuilder`

Assembles ML-ready DataFrames from GEECS scan data.

| Method | Description |
|--------|-------------|
| `.from_scan(scan, ...)` | Build from a single `ScanData` object |
| `.from_scans(scans, ...)` | Build from multiple scans (concatenated) |
| `.from_dataframe(df, ...)` | Build from a raw pandas DataFrame |

Common parameters:
- `feature_specs` — one resolved column per term via `geecs_data_utils.data.columns.resolve_col` (same rules as `ScanData.resolve_col` without local aliases)
- `target_specs` / `target_column` — the column to predict
- `filters` — row filters as `(column, operator, value)` tuples, e.g. `[("charge", ">", 2)]`
- `outlier_config` — optional `OutlierConfig(method="nan", sigma=5.0)` for outlier handling
- `dropna` — drop rows with NaN (default `True`)

Returns a `DatasetResult` with `.frame`, `.feature_columns`, `.target_column`, `.rows_raw`, `.rows_final`.

#### `OutlierConfig`

```python
OutlierConfig(method="nan", sigma=5.0, columns=None)
```

- `method="nan"` — replace outlier values with NaN
- `method="clip"` — remove entire rows containing outliers
- `sigma` — number of standard deviations for the threshold
- `columns` — specific columns to apply to (default: all numeric)

### Correlation ranking (also `geecs_data_utils.analysis`)

#### `CorrelationReport`

```python
from geecs_data_utils.analysis import CorrelationReport

report = CorrelationReport.from_dataframe(
    df,
    target="charge",
    method="spearman",           # "pearson", "spearman", or "kendall"
    exclude_terms=["timestamp", "shotnumber"],  # exclude columns by substring
    filters=[("charge", ">", 0)],              # row filters
    top_n=20,                                   # limit results
)

report.ranked_features      # list of feature names, strongest correlation first
report.top_features(n=10)   # top N feature names
report.correlations         # pd.Series of correlation values
report.filtered_frame       # the DataFrame used for computation
```

#### Outlier Functions

```python
# Remove rows with values beyond +/- sigma
clean_df = sigma_clip_frame(df, sigma=6.0, columns=["col_a", "col_b"])

# Replace outlier values with NaN (keeps all rows)
clean_df = sigma_nan_frame(df, sigma=5.0)
```

### Training

#### `RegressionTrainer`

```python
trainer = RegressionTrainer(
    model="ridge",               # "linear", "ridge", or "elasticnet"
    model_params={"alpha": 1.0}, # passed to the sklearn estimator
    impute=True,                 # include median imputation in pipeline
    scale=True,                  # include standard scaling in pipeline
)

artifact = trainer.fit(
    X,                           # DataFrame or numpy array
    y,                           # Series or numpy array
    feature_names=["a", "b"],    # optional if X is a DataFrame
    target_name="charge",
    cv=5,                        # optional cross-validation folds
    scan_info={"scan": 1},       # optional metadata
)
```

Returns a `ModelArtifact` containing the fitted pipeline, feature schema, metadata, and metrics.

#### `ModelArtifact`

The central object — returned by `RegressionTrainer.fit()` and by `load_model_artifact()`.

```python
artifact.predict(df)              # run predictions (validates schema first)
artifact.pipeline                 # the fitted sklearn Pipeline
artifact.feature_schema           # FeatureSchema with .feature_names, .target_name
artifact.metadata                 # ModelMetadata (model_type, created_at, training_rows, ...)
artifact.metrics                  # TrainingMetrics (r2, mae, rmse, cv_r2_mean, cv_r2_std)
```

### Persistence

#### Saving

```python
save_model_artifact(artifact, "models/charge_ridge_v1")
```

Creates a directory with:
```
models/charge_ridge_v1/
    model.joblib          # fitted sklearn Pipeline (binary)
    metadata.json         # training metadata (human-readable)
    feature_schema.json   # feature contract (human-readable)
    metrics.json          # evaluation metrics (human-readable)
```

#### Loading

```python
loaded = load_model_artifact("models/charge_ridge_v1")
preds = loaded.predict(new_data)
```

The loaded artifact knows what features it expects. Passing data with wrong columns raises a clear `ValueError`.

### Inference

#### Direct prediction

```python
artifact.predict(df)  # df must have exactly the expected feature columns
```

#### From a ScanData object

```python
from geecs_data_utils.ml import predict_from_scan

preds = predict_from_scan(artifact, scan)
```

Optional `feature_specs` uses `geecs_data_utils.data.columns.find_cols` on the scan frame (same matching rules as `ScanData.find_cols`) to discover columns that contain the trained feature names.

### Schema Validation

```python
# This raises ValueError with a clear message:
loaded.predict(df_with_wrong_columns)
# ValueError: Missing features: ['feature_a']; Unexpected features: ['wrong_col']
```

## Supported Models

| Name | sklearn Class | Notes |
|------|--------------|-------|
| `"linear"` | `LinearRegression` | No hyperparameters |
| `"ridge"` | `Ridge` | `alpha` controls regularization |
| `"elasticnet"` | `ElasticNet` | `alpha` + `l1_ratio` |

## Tests

Run from the repository root:

```bash
poetry run python -m pytest GEECS-Data-Utils/tests/ -v
```

Test coverage:
- Feature selection: correlation ranking, outlier handling, exclusion terms, all three methods
- Dataset builder: column selection, outlier configs, row filters, missing column errors
- Models: all three estimators, cross-validation, schema validation, numpy input
- Persistence: save/load roundtrip, metadata/schema/metrics preservation, JSON readability
- Integration: full end-to-end workflow (build, rank, train, save, load, predict)

## Dependencies

Required (via `geecs-data-utils[ml]`):
- `scikit-learn >= 1.3`
- `pandas`, `numpy` (core geecs-data-utils deps)

Optional (not required for any current functionality):
- `optuna >= 3.0` (reserved for future tuning support)
