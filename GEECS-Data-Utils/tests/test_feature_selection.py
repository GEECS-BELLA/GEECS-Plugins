"""Tests for correlation ranking and outlier handling."""

import pandas as pd
import pytest

from geecs_data_utils.data import sigma_clip_frame, sigma_nan_frame
from geecs_data_utils.ml.feature_selection import CorrelationReport


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


class TestCorrelationReport:
    """Tests for CorrelationReport."""

    def test_basic_ranking(self, sample_df):
        """Features are ranked by absolute correlation."""
        report = CorrelationReport.from_dataframe(sample_df, target="charge")
        ranked = report.ranked_features
        assert len(ranked) > 0
        # feature_a (coeff=3.0) should be most correlated
        assert ranked[0] == "feature_a"

    def test_top_features(self, sample_df):
        """top_features returns the requested count."""
        report = CorrelationReport.from_dataframe(sample_df, target="charge")
        top2 = report.top_features(n=2)
        assert len(top2) == 2

    def test_exclude_terms(self, sample_df):
        """Excluded terms are filtered from the ranking."""
        report = CorrelationReport.from_dataframe(
            sample_df, target="charge", exclude_terms=["timestamp", "shot"]
        )
        for feat in report.ranked_features:
            assert "timestamp" not in feat.lower()
            assert "shot" not in feat.lower()

    def test_row_filters(self, sample_df):
        """Row filters reduce the dataset."""
        report = CorrelationReport.from_dataframe(
            sample_df,
            target="charge",
            filters=[("charge", ">", 0)],
        )
        assert report.rows_after_filter < report.rows_before_filter

    def test_methods(self, sample_df):
        """All three correlation methods work."""
        for method in ("pearson", "spearman", "kendall"):
            report = CorrelationReport.from_dataframe(
                sample_df, target="charge", method=method
            )
            assert len(report.correlations) > 0

    def test_top_n_limit(self, sample_df):
        """top_n limits the result."""
        report = CorrelationReport.from_dataframe(sample_df, target="charge", top_n=2)
        assert len(report.correlations) == 2

    def test_invalid_target_raises(self, sample_df):
        """Non-existent target raises ValueError."""
        with pytest.raises(ValueError, match="not numeric or not present"):
            CorrelationReport.from_dataframe(sample_df, target="nonexistent")
