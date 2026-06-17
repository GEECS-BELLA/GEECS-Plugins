"""Tests for sigma_clip_frame and sigma_nan_frame outlier helpers."""

import pandas as pd

from geecs_data_utils.data import sigma_clip_frame, sigma_nan_frame


class TestSigmaClipFrame:
    """Tests for sigma_clip_frame."""

    def test_removes_outlier_rows(self, sample_df):
        """Rows with extreme values are removed."""
        df = sample_df.copy()
        df.loc[0, "feature_a"] = 1000.0  # extreme outlier
        result = sigma_clip_frame(df, sigma=3.0)
        assert len(result) < len(df)
        assert 0 not in result.index

    def test_preserves_normal_data(self, sample_df):
        """Normal data should mostly survive clipping."""
        result = sigma_clip_frame(sample_df, sigma=6.0)
        assert len(result) >= len(sample_df) * 0.9

    def test_specific_columns(self, sample_df):
        """Only clip on specified columns."""
        df = sample_df.copy()
        df.loc[0, "feature_a"] = 1000.0
        df.loc[1, "feature_b"] = 1000.0
        result = sigma_clip_frame(df, sigma=3.0, columns=["feature_a"])
        # Row 0 removed (feature_a outlier), row 1 kept (feature_b not clipped)
        assert 0 not in result.index
        assert 1 in result.index


class TestSigmaNanFrame:
    """Tests for sigma_nan_frame."""

    def test_replaces_outliers_with_nan(self, sample_df):
        """Outlier values become NaN."""
        df = sample_df.copy()
        df.loc[0, "feature_a"] = 1000.0
        result = sigma_nan_frame(df, sigma=3.0)
        assert pd.isna(result.loc[0, "feature_a"])
        # Row is still present
        assert len(result) == len(df)

    def test_non_outliers_unchanged(self, sample_df):
        """Non-outlier values remain the same."""
        result = sigma_nan_frame(sample_df, sigma=6.0)
        # Most values should be unchanged
        assert result["feature_a"].notna().sum() >= len(sample_df) * 0.9
