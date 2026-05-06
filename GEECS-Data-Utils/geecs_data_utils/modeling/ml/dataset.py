"""ML-oriented dataset assembly on top of :mod:`geecs_data_utils.data.dataset`.

Notes
-----
Processing is conceptually three stages:

1. Generic assembly and cleaning via
   :class:`~geecs_data_utils.data.dataset.DatasetBuilder` with ``dropna=False``,
   so final NaN handling stays with ML column selection.
2. Target and feature columns chosen by explicit names or by *specs* resolved
   with :func:`~geecs_data_utils.data.columns.resolve_col`.
3. Optional ``dropna`` on the selected columns only.

For tabular assembly without sklearn column semantics, use
:mod:`geecs_data_utils.data.dataset` directly.

See Also
--------
geecs_data_utils.data.dataset.DatasetBuilder : Generic concat/clean assembly.
geecs_data_utils.data.columns.resolve_col : Column matching for specs.
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
    """Container returned by :class:`BeamPredictionDatasetBuilder`.

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
        Optional provenance from ``DatasetBuilder`` (e.g. merged scan metadata).
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


class BeamPredictionDatasetBuilder:
    """Build :class:`DatasetResult` from scans or from an existing DataFrame.

    ``from_scan`` and ``from_scans`` delegate to ``DatasetBuilder``, then apply
    ML-specific column resolution. ``from_dataframe`` skips scan loading but
    uses the same cleaning hooks.

    Notes
    -----
    Target selection: pass ``target_column`` for an exact name, or
    ``target_specs`` to try each string with ``resolve_col`` until one matches.

    Feature selection: pass ``feature_specs`` to resolve distinct columns in
    order; if omitted, all numeric columns except the target are used.

    See Also
    --------
    geecs_data_utils.data.dataset.DatasetBuilder
        Generic scan and DataFrame assembly.
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
        """Assemble from one scan with ``ScanData``-like ``data_frame``.

        Parameters
        ----------
        scan : object
            Must provide ``data_frame`` (e.g. :class:`~geecs_data_utils.scan_data.ScanData`).
        feature_specs : sequence of str, optional
            Passed to :func:`~geecs_data_utils.data.columns.resolve_col` in order.
        target_specs : sequence of str, optional
            First matching spec becomes the target; ignored if ``target_column`` is set.
        target_column : str, optional
            Exact target column name; mutually exclusive with needing ``target_specs``.
        filters : list, optional
            Row filters forwarded to ``DatasetBuilder``.
        outlier_config : OutlierConfig, optional
            Outlier handling forwarded to ``DatasetBuilder``.
        dropna : bool, default True
            Drop rows with NaN in selected feature + target columns.

        Returns
        -------
        DatasetResult

        Raises
        ------
        ValueError
            If ``scan.data_frame`` is None, or neither ``target_column`` nor
            ``target_specs`` yields a target.
        """
        if scan.data_frame is None:
            raise ValueError("ScanData has no loaded data_frame.")

        assembled = DatasetBuilder.from_scan(
            scan,
            filters=filters,
            outlier_config=outlier_config,
            dropna=False,
        )
        return BeamPredictionDatasetBuilder._build_model_columns(
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
        """Like :meth:`from_scan`, but concatenate multiple scans first.

        Parameters
        ----------
        scans : sequence
            Scan objects accepted by :meth:`DatasetBuilder.from_scans`.
        feature_specs, target_specs, target_column, filters, outlier_config, dropna
            Same semantics as :meth:`from_scan`.

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
        return BeamPredictionDatasetBuilder._build_model_columns(
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
        """Build from a DataFrame without loading scans.

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

    @staticmethod
    def _build_model_columns(
        out: pd.DataFrame,
        *,
        rows_raw: int,
        feature_specs: Optional[Sequence[str]],
        target_specs: Optional[Sequence[str]],
        target_column: Optional[str],
        dropna: bool,
        scan_info: Dict[str, Any],
    ) -> DatasetResult:
        """Resolve specs, subset columns, optionally drop NaNs (internal).

        Parameters
        ----------
        out : pandas.DataFrame
            Assembled frame from ``DatasetBuilder``.
        rows_raw : int
            Row count before this step's ``dropna``.
        feature_specs : sequence of str, optional
        target_specs : sequence of str, optional
        target_column : str, optional
        dropna : bool
        scan_info : dict
            Forwarded into :class:`DatasetResult`.

        Returns
        -------
        DatasetResult
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
    """First column matched by any spec via :func:`~geecs_data_utils.data.columns.resolve_col`.

    Parameters
    ----------
    df : pandas.DataFrame
    specs : sequence of str
        Tried in order until one resolves.

    Returns
    -------
    str
        Resolved column name.

    Raises
    ------
    ValueError
        If no spec matches.
    """
    for spec in specs:
        try:
            return resolve_col(df, spec)
        except ValueError:
            continue
    raise ValueError(f"Could not resolve any column from specs: {list(specs)}")


def _resolve_feature_columns(df: pd.DataFrame, specs: Sequence[str]) -> List[str]:
    """Resolve each spec to one column; preserve order; omit duplicates.

    Parameters
    ----------
    df : pandas.DataFrame
    specs : sequence of str

    Returns
    -------
    list of str
    """
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
