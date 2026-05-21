"""ML-oriented dataset assembly on top of :mod:`geecs_data_utils.data.dataset`.

Notes
-----
``MLDatasetBuilder`` operates on a DataFrame that the caller has already
assembled (typically via :class:`~geecs_data_utils.data.dataset.DatasetBuilder`
for multi-scan loads, or directly for a single scan). Keeping scan-loading
out of this layer avoids duplicating selection logic across two entry points
and keeps the modeling surface focused on "I have a frame, give me X/y."

Processing is conceptually three stages:

1. Generic cleaning via
   :class:`~geecs_data_utils.data.dataset.DatasetBuilder.from_dataframe` with
   ``dropna=False`` so final NaN handling stays with ML column selection.
2. Target and feature columns chosen by explicit names.
3. Optional ``dropna`` on the selected columns only.

For tabular assembly without sklearn column semantics, use
:mod:`geecs_data_utils.data.dataset` directly.

See Also
--------
geecs_data_utils.data.dataset.DatasetBuilder : Generic concat/clean assembly.
geecs_data_utils.data.columns.resolve_col : Column matching helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from geecs_data_utils.data.cleaning import OutlierConfig, RowFilterSpec
from geecs_data_utils.data.dataset import DatasetBuilder


@dataclass
class DatasetResult:
    """Container returned by :class:`MLDatasetBuilder`.

    ``frame`` contains exactly the predictor columns plus the target, ordered for
    ``X`` / ``y`` extraction.

    Attributes
    ----------
    frame : pandas.DataFrame
        Subset ready for modeling (typically numeric columns).
    feature_columns : list of str
        Predictor column names present in ``frame``.
    target_column : str
        Response column name in ``frame``.
    scan_info : dict
        Optional provenance carried through from upstream assembly.
    rows_raw : int
        Row count before this builder's final ``dropna``.
    rows_final : int
        Row count after final ``dropna`` when enabled.
    """

    frame: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    scan_info: Dict[str, Any] = field(default_factory=dict)
    rows_raw: int = 0
    rows_final: int = 0


class MLDatasetBuilder:
    """Build :class:`DatasetResult` from an existing DataFrame.

    For multi-scan loads, build the source frame with
    :class:`~geecs_data_utils.data.dataset.DatasetBuilder` first and then pass
    its ``frame`` to :meth:`from_dataframe`. This separation keeps the modeling
    layer free of scan-loading concerns.

    See Also
    --------
    geecs_data_utils.data.dataset.DatasetBuilder
        Generic scan and DataFrame assembly.
    """

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        *,
        feature_columns: Optional[Sequence[str]] = None,
        target_column: str,
        filters: Optional[List[RowFilterSpec]] = None,
        outlier_config: Optional[OutlierConfig] = None,
        dropna: bool = True,
    ) -> DatasetResult:
        """Build from a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input table.
        feature_columns : sequence of str, optional
            Exact predictor names; if omitted, all numeric columns except
            ``target_column`` are used.
        target_column : str
            Exact response column name.
        filters : list, optional
            Row filters for ``DatasetBuilder``.
        outlier_config : OutlierConfig, optional
            Outlier handling for ``DatasetBuilder``.
        dropna : bool, default True
            Drop NaNs on the selected columns.

        Returns
        -------
        DatasetResult

        Raises
        ------
        ValueError
            If any selected column is missing from ``df``.
        """
        assembled = DatasetBuilder.from_dataframe(
            df,
            filters=filters,
            outlier_config=outlier_config,
            dropna=False,
        )
        out = assembled.frame
        rows_raw = assembled.rows_raw

        if feature_columns is not None:
            features = list(feature_columns)
        else:
            features = [
                c
                for c in out.select_dtypes(include="number").columns
                if c != target_column
            ]

        selected = features + [target_column]
        missing = [c for c in selected if c not in out.columns]
        if missing:
            raise ValueError(f"Columns not found in dataframe: {missing}")

        out = out[selected]
        if dropna:
            out = out.dropna()

        return DatasetResult(
            frame=out,
            feature_columns=features,
            target_column=target_column,
            rows_raw=rows_raw,
            rows_final=len(out),
        )
