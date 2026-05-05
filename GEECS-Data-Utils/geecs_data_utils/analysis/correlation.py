"""Correlation-based ranking of numeric columns against a target (non-ML).

:class:`CorrelationReport` mirrors typical exploratory workflows: optional row
filters (:func:`~geecs_data_utils.data.cleaning.apply_row_filters`), drop rows
with NaN in the target, then correlate every numeric column with that target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import pandas as pd

from geecs_data_utils.data.cleaning import RowFilterSpec, apply_row_filters

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
        Table after row filters (if any) and dropping rows where *target* is NaN.
        ``exclude_terms`` only trims the reported correlation series; it does not
        drop columns from this frame.
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

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        target: str,
        *,
        method: CorrelationMethod = "pearson",
        exclude_terms: Optional[Sequence[str]] = None,
        filters: Optional[List[RowFilterSpec]] = None,
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

        if filters:
            filtered = apply_row_filters(filtered, filters)

        filtered = filtered.dropna(subset=[target])

        rows_after = len(filtered)

        numeric = filtered.select_dtypes(include="number")
        if target not in numeric.columns:
            raise ValueError(
                f"Target column '{target}' is not numeric or not present in the frame."
            )
        corr_series: pd.Series = numeric.corr(method=method)[target].drop(
            labels=target, errors="ignore"
        )

        if exclude_terms:
            pattern = "|".join(exclude_terms)
            mask = ~corr_series.index.str.contains(pattern, case=False)
            corr_series = corr_series[mask]

        corr_series = corr_series.reindex(
            corr_series.abs().sort_values(ascending=False).index
        )

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
