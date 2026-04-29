"""Correlation-based feature ranking and outlier handling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd

from geecs_data_utils.data.cleaning import apply_row_filters

# ---------------------------------------------------------------------------
# Correlation report
# ---------------------------------------------------------------------------

CorrelationMethod = Literal["pearson", "spearman", "kendall"]


@dataclass
class CorrelationReport:
    """Result of a correlation ranking analysis.

    Attributes
    ----------
    correlations : Series
        Correlation of each feature with the target, sorted by absolute value
        (strongest first).
    target : str
        Name of the target column.
    method : str
        Correlation method used.
    filtered_frame : DataFrame
        The dataframe that was used for computing correlations (after any row
        filtering and column exclusion).
    rows_before_filter : int
        Row count before filtering.
    rows_after_filter : int
        Row count after filtering.
    """

    correlations: pd.Series
    target: str
    method: str
    filtered_frame: pd.DataFrame
    rows_before_filter: int = 0
    rows_after_filter: int = 0

    # -- convenience properties ------------------------------------------------

    @property
    def ranked_features(self) -> List[str]:
        """Feature names ordered by absolute correlation strength."""
        return self.correlations.index.tolist()

    def top_features(self, n: int = 10) -> List[str]:
        """Return the top *n* most correlated feature names.

        Parameters
        ----------
        n : int
            Number of features to return.

        Returns
        -------
        list of str
        """
        return self.correlations.head(n).index.tolist()

    # -- factory ---------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target: str,
        *,
        method: CorrelationMethod = "pearson",
        exclude_terms: Optional[Sequence[str]] = None,
        filters: Optional[List[Tuple[str, str, Union[int, float]]]] = None,
        top_n: Optional[int] = None,
    ) -> CorrelationReport:
        """Compute a correlation report for *target* against all numeric columns.

        Parameters
        ----------
        df : DataFrame
            Input data (typically the output of a dataset builder).
        target : str
            Column name to compute correlations against.
        method : ``"pearson"`` | ``"spearman"`` | ``"kendall"``
            Correlation method.
        exclude_terms : sequence of str, optional
            Substrings — any column whose name contains one of these terms
            (case-insensitive) is excluded from the ranking.
        filters : list of (column, operator, value) tuples, optional
            Row filters applied before computing correlations.
            Supported operators: ``">"``, ``"<"``, ``">="``, ``"<="``,
            ``"=="``, ``"!="``.
        top_n : int, optional
            If given, limit the result to the top *n* correlations.

        Returns
        -------
        CorrelationReport
        """
        rows_before = len(df)
        filtered = df.copy()

        if target not in filtered.columns:
            raise ValueError(
                f"Target column '{target}' is not numeric or not present in the frame."
            )

        # Row filters
        if filters:
            filtered = apply_row_filters(filtered, filters)

        # Drop NaN in target
        filtered = filtered.dropna(subset=[target])

        rows_after = len(filtered)

        # Numeric correlation matrix
        numeric = filtered.select_dtypes(include="number")
        if target not in numeric.columns:
            raise ValueError(
                f"Target column '{target}' is not numeric or not present in the frame."
            )
        corr_series: pd.Series = numeric.corr(method=method)[target].drop(
            labels=target, errors="ignore"
        )

        # Exclude columns by terms
        if exclude_terms:
            pattern = "|".join(exclude_terms)
            mask = ~corr_series.index.str.contains(pattern, case=False)
            corr_series = corr_series[mask]

        # Sort by absolute value
        corr_series = corr_series.reindex(
            corr_series.abs().sort_values(ascending=False).index
        )

        # Drop NaN correlations
        corr_series = corr_series.dropna()

        if top_n is not None:
            corr_series = corr_series.head(top_n)

        return cls(
            correlations=corr_series,
            target=target,
            method=method,
            filtered_frame=filtered,
            rows_before_filter=rows_before,
            rows_after_filter=rows_after,
        )
