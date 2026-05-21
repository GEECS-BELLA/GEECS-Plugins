"""Tests for CorrelationReport ranking and filtering."""

import pytest

from geecs_data_utils.analysis import CorrelationReport


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
