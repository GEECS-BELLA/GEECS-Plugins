"""Tests for MLDatasetBuilder."""

import numpy as np
import pytest

from geecs_data_utils.modeling.ml.dataset import MLDatasetBuilder
from geecs_data_utils.data import OutlierConfig


class TestFromDataframe:
    """Tests for MLDatasetBuilder.from_dataframe."""

    def test_basic_build(self, sample_df):
        """Builds a dataset with explicit columns."""
        ds = MLDatasetBuilder.from_dataframe(
            sample_df,
            feature_columns=["feature_a", "feature_b"],
            target_column="charge",
        )
        assert ds.target_column == "charge"
        assert ds.feature_columns == ["feature_a", "feature_b"]
        assert "charge" in ds.frame.columns
        assert ds.rows_final <= ds.rows_raw

    def test_auto_features(self, sample_df):
        """Without feature_columns, all numeric cols except target are used."""
        ds = MLDatasetBuilder.from_dataframe(sample_df, target_column="charge")
        assert "charge" not in ds.feature_columns
        assert len(ds.feature_columns) > 0

    def test_outlier_config_nan(self, sample_df):
        """Outlier config with nan method works."""
        df = sample_df.copy()
        df.loc[0, "feature_a"] = 1000.0
        ds = MLDatasetBuilder.from_dataframe(
            df,
            feature_columns=["feature_a", "feature_b"],
            target_column="charge",
            outlier_config=OutlierConfig(method="nan", sigma=3.0),
        )
        # Row 0 should be dropped due to NaN after outlier handling + dropna
        assert ds.rows_final < ds.rows_raw

    def test_outlier_config_clip(self, sample_df):
        """Outlier config with clip method works."""
        df = sample_df.copy()
        df.loc[0, "feature_a"] = 1000.0
        ds = MLDatasetBuilder.from_dataframe(
            df,
            feature_columns=["feature_a", "feature_b"],
            target_column="charge",
            outlier_config=OutlierConfig(method="clip", sigma=3.0),
        )
        assert ds.rows_final < ds.rows_raw

    def test_row_filters(self, sample_df):
        """Row filters reduce the dataset."""
        ds = MLDatasetBuilder.from_dataframe(
            sample_df,
            feature_columns=["feature_a"],
            target_column="charge",
            filters=[("charge", ">", 0)],
        )
        assert ds.rows_final < ds.rows_raw

    def test_missing_column_raises(self, sample_df):
        """Referencing a missing column raises ValueError."""
        with pytest.raises(ValueError, match="Columns not found"):
            MLDatasetBuilder.from_dataframe(
                sample_df,
                feature_columns=["nonexistent"],
                target_column="charge",
            )

    def test_exclude_specs_drops_matching_features(self, sample_df):
        """exclude_specs removes columns whose names contain the substring."""
        # sample_df has feature_a / feature_b / feature_c; auto-select picks
        # all three, exclude_specs="_c" should drop feature_c.
        ds = MLDatasetBuilder.from_dataframe(
            sample_df,
            target_column="charge",
            exclude_specs=["_c"],
        )
        assert "feature_c" not in ds.feature_columns
        assert "feature_a" in ds.feature_columns
        assert "feature_b" in ds.feature_columns

    def test_exclude_specs_applies_to_explicit_features(self, sample_df):
        """exclude_specs works even with an explicit feature_columns list."""
        ds = MLDatasetBuilder.from_dataframe(
            sample_df,
            feature_columns=["feature_a", "feature_b", "feature_c"],
            target_column="charge",
            exclude_specs=["_b"],
        )
        assert ds.feature_columns == ["feature_a", "feature_c"]

    def test_exclude_specs_never_drops_target(self, sample_df):
        """A spec that would match the target leaves the target intact."""
        ds = MLDatasetBuilder.from_dataframe(
            sample_df,
            target_column="charge",
            exclude_specs=["charge"],
        )
        assert ds.target_column == "charge"
        assert "charge" not in ds.feature_columns  # target is never a feature

    def test_exclude_specs_removing_everything_raises(self, sample_df):
        """exclude_specs that wipes the candidate set raises a clear error."""
        with pytest.raises(ValueError, match="removed every candidate feature"):
            MLDatasetBuilder.from_dataframe(
                sample_df,
                feature_columns=["feature_a", "feature_b"],
                target_column="charge",
                exclude_specs=["feature"],
            )

    def test_no_dropna(self, sample_df):
        """With dropna=False, NaN rows are kept."""
        df = sample_df.copy()
        df.loc[0, "feature_a"] = np.nan
        ds = MLDatasetBuilder.from_dataframe(
            df,
            feature_columns=["feature_a", "feature_b"],
            target_column="charge",
            dropna=False,
        )
        assert ds.rows_final == ds.rows_raw
