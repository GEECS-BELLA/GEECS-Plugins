"""ML dataset assembly: column resolution on top of :mod:`geecs_data_utils.data.dataset`.

Concatenation, outliers, and row filtering reuse :class:`~geecs_data_utils.data.dataset.DatasetBuilder`
so behavior stays aligned with non-ML scan table workflows (Task D split).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from geecs_data_utils.data.cleaning import OutlierConfig, RowFilterSpec
from geecs_data_utils.data.columns import resolve_col
from geecs_data_utils.data.dataset import DatasetBuilder


@dataclass
class DatasetResult:
    """The output of :class:`BeamPredictionDatasetBuilder`.

    Attributes
    ----------
    frame : DataFrame
        The assembled ML-ready dataframe.
    feature_columns : list of str
        Names of the feature columns in *frame*.
    target_column : str
        Name of the target column in *frame*.
    scan_info : dict
        Metadata about the source scan(s).
    rows_raw : int
        Row count of the merged / single-scan frame before outlier and row-filter steps.
    rows_final : int
        Row count in ``frame`` after the full ML pipeline (including optional ``dropna``).
    """

    frame: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    scan_info: Dict[str, Any] = field(default_factory=dict)
    rows_raw: int = 0
    rows_final: int = 0


class BeamPredictionDatasetBuilder:
    """Assembles ML-ready DataFrames from GEECS scan data.

    Scan concatenation and shared cleaning (outliers, row filters) delegate to
    :class:`~geecs_data_utils.data.dataset.DatasetBuilder` with ``dropna=False``
    so ML column selection runs first; final ``dropna`` applies only to the
    selected feature and target columns.

    Examples
    --------
    >>> from geecs_data_utils import ScanData
    >>> from geecs_data_utils.ml import BeamPredictionDatasetBuilder
    >>> scan = ScanData.from_date(year=2026, month=2, day=18, number=1,
    ...                           experiment="Undulator")
    >>> ds = BeamPredictionDatasetBuilder.from_scan(
    ...     scan,
    ...     feature_specs=["laser", "pressure"],
    ...     target_specs=["charge"],
    ... )
    >>> ds.frame.head()
    """

    @staticmethod
    def from_scan(
        scan: Any,
        *,
        feature_specs: Optional[Sequence[str]] = None,
        target_specs: Optional[Sequence[str]] = None,
        target_column: Optional[str] = None,
        filters: Optional[List[RowFilterSpec]] = None,
        outlier_config: Optional[OutlierConfig] = None,
        dropna: bool = True,
    ) -> DatasetResult:
        """Build a dataset from a single :class:`~geecs_data_utils.ScanData`.

        Parameters
        ----------
        scan : ScanData
            A loaded scan data object (must have a populated ``data_frame``).
        feature_specs : sequence of str, optional
            Search terms resolved with :func:`~geecs_data_utils.data.columns.resolve_col`
            (same rules as :meth:`~geecs_data_utils.ScanData.resolve_col`, without
            local aliases). One column per spec. If ``None``, all numeric columns
            (except the target) are used.
        target_specs : sequence of str, optional
            Terms tried in order until one resolves via ``resolve_col``.
            Ignored if *target_column* is provided directly.
        target_column : str, optional
            Explicit target column name.  Takes precedence over
            *target_specs*.
        filters : list of (column, operator, value) tuples, optional
            Row filters applied before assembling the dataset.
        outlier_config : OutlierConfig, optional
            Outlier handling configuration.
        dropna : bool
            If ``True``, drop rows with any ``NaN`` in the selected columns.

        Returns
        -------
        DatasetResult
        """
        if scan.data_frame is None:
            raise ValueError("ScanData has no loaded data_frame.")

        assembled = DatasetBuilder.from_scan(
            scan,
            filters=filters,
            outlier_config=outlier_config,
            dropna=False,
        )
        return BeamPredictionDatasetBuilder._build_ml_columns(
            assembled.frame,
            rows_raw=assembled.rows_raw,
            feature_specs=feature_specs,
            target_specs=target_specs,
            target_column=target_column,
            dropna=dropna,
            scan_info=assembled.scan_info,
        )

    @staticmethod
    def from_scans(
        scans: Sequence[Any],
        *,
        feature_specs: Optional[Sequence[str]] = None,
        target_specs: Optional[Sequence[str]] = None,
        target_column: Optional[str] = None,
        filters: Optional[List[RowFilterSpec]] = None,
        outlier_config: Optional[OutlierConfig] = None,
        dropna: bool = True,
    ) -> DatasetResult:
        """Build a dataset from multiple scans concatenated together.

        Parameters
        ----------
        scans : sequence of ScanData
            Loaded scan data objects.
        feature_specs, target_specs, target_column, filters, outlier_config, dropna
            Same as :meth:`from_scan`.

        Returns
        -------
        DatasetResult
        """
        assembled = DatasetBuilder.from_scans(
            scans,
            filters=filters,
            outlier_config=outlier_config,
            dropna=False,
        )
        return BeamPredictionDatasetBuilder._build_ml_columns(
            assembled.frame,
            rows_raw=assembled.rows_raw,
            feature_specs=feature_specs,
            target_specs=target_specs,
            target_column=target_column,
            dropna=dropna,
            scan_info=assembled.scan_info,
        )

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
        """Build a dataset from a raw DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input data.
        feature_columns : sequence of str, optional
            Explicit feature column names.  If ``None``, all numeric columns
            except the target are used.
        target_column : str
            Target column name.
        filters : list of (column, operator, value) tuples, optional
            Row filters.
        outlier_config : OutlierConfig, optional
            Outlier handling configuration.
        dropna : bool
            Drop rows with ``NaN`` in selected columns.

        Returns
        -------
        DatasetResult
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

    # ------------------------------------------------------------------
    # Internal: ML column selection (after shared DatasetBuilder cleaning)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ml_columns(
        out: pd.DataFrame,
        *,
        rows_raw: int,
        feature_specs: Optional[Sequence[str]],
        target_specs: Optional[Sequence[str]],
        target_column: Optional[str],
        dropna: bool,
        scan_info: Dict[str, Any],
    ) -> DatasetResult:
        """Resolve target/features with :func:`~geecs_data_utils.data.columns.resolve_col`; subset columns.

        Expects *out* to already reflect :meth:`~geecs_data_utils.data.dataset.DatasetBuilder.prepare_frame`
        (with ``dropna=False`` when called from the public builders).
        """
        if target_column is not None:
            tgt = target_column
        elif target_specs is not None:
            tgt = _resolve_column(out, target_specs)
        else:
            raise ValueError("Either target_column or target_specs must be provided.")

        if feature_specs is not None:
            features = _resolve_feature_columns(out, feature_specs)
            features = [f for f in features if f != tgt]
        else:
            features = [
                c for c in out.select_dtypes(include="number").columns if c != tgt
            ]

        selected = features + [tgt]
        out = out[[c for c in selected if c in out.columns]]

        if dropna:
            out = out.dropna()

        return DatasetResult(
            frame=out,
            feature_columns=[c for c in features if c in out.columns],
            target_column=tgt,
            scan_info=scan_info,
            rows_raw=rows_raw,
            rows_final=len(out),
        )


def _resolve_column(df: pd.DataFrame, specs: Sequence[str]) -> str:
    """Return the first *spec* that :func:`~geecs_data_utils.data.columns.resolve_col` resolves."""
    for spec in specs:
        try:
            return resolve_col(df, spec)
        except ValueError:
            continue
    raise ValueError(f"Could not resolve any column from specs: {list(specs)}")


def _resolve_feature_columns(df: pd.DataFrame, specs: Sequence[str]) -> List[str]:
    """Resolve each *spec* to at most one column; skip specs that match nothing."""
    features: List[str] = []
    seen: set[str] = set()
    for spec in specs:
        try:
            col = resolve_col(df, spec)
        except ValueError:
            continue
        if col not in seen:
            features.append(col)
            seen.add(col)
    return features
