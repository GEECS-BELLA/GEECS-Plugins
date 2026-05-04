"""Dataset construction from GEECS scan data for ML workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from geecs_data_utils.data.cleaning import (
    apply_row_filters,
    OutlierConfig,
    apply_outlier_config,
)
from geecs_data_utils.data.columns import resolve_col


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
        Row count before any filtering or outlier removal.
    rows_final : int
        Row count after filtering.
    """

    frame: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    scan_info: Dict[str, Any] = field(default_factory=dict)
    rows_raw: int = 0
    rows_final: int = 0


class BeamPredictionDatasetBuilder:
    """Assembles ML-ready DataFrames from GEECS scan data.

    This class converts one or more :class:`~geecs_data_utils.ScanData`
    objects (or raw DataFrames) into a rectangular dataset suitable for
    regression training.

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
        filters: Optional[list] = None,
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
        df = scan.data_frame
        if df is None:
            raise ValueError("ScanData has no loaded data_frame.")

        scan_info = _extract_scan_info(scan)
        return BeamPredictionDatasetBuilder._build(
            df,
            feature_specs=feature_specs,
            target_specs=target_specs,
            target_column=target_column,
            filters=filters,
            outlier_config=outlier_config,
            dropna=dropna,
            scan_info=scan_info,
        )

    @staticmethod
    def from_scans(
        scans: Sequence[Any],
        *,
        feature_specs: Optional[Sequence[str]] = None,
        target_specs: Optional[Sequence[str]] = None,
        target_column: Optional[str] = None,
        filters: Optional[list] = None,
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
        frames = []
        scan_infos = []
        for s in scans:
            if s.data_frame is None:
                continue
            frames.append(s.data_frame)
            scan_infos.append(_extract_scan_info(s))

        if not frames:
            raise ValueError("No scans with loaded data_frame provided.")

        df = pd.concat(frames, ignore_index=True)
        scan_info = {"scans": scan_infos, "total_scans": len(scan_infos)}

        return BeamPredictionDatasetBuilder._build(
            df,
            feature_specs=feature_specs,
            target_specs=target_specs,
            target_column=target_column,
            filters=filters,
            outlier_config=outlier_config,
            dropna=dropna,
            scan_info=scan_info,
        )

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        *,
        feature_columns: Optional[Sequence[str]] = None,
        target_column: str,
        filters: Optional[list] = None,
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
        rows_raw = len(df)
        out = df.copy()

        # Outlier handling
        if outlier_config is not None:
            out = apply_outlier_config(out, outlier_config)

        # Row filters
        if filters:
            out = apply_row_filters(out, filters)

        # Resolve features
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
    # Internal build logic
    # ------------------------------------------------------------------

    @staticmethod
    def _build(
        df: pd.DataFrame,
        *,
        feature_specs: Optional[Sequence[str]],
        target_specs: Optional[Sequence[str]],
        target_column: Optional[str],
        filters: Optional[list],
        outlier_config: Optional[OutlierConfig],
        dropna: bool,
        scan_info: Dict[str, Any],
    ) -> DatasetResult:
        """Shared build logic for scan-based factories."""
        rows_raw = len(df)
        out = df.copy()

        # Outlier handling
        if outlier_config is not None:
            out = apply_outlier_config(out, outlier_config)

        # Row filters
        if filters:
            out = apply_row_filters(out, filters)

        # Resolve target
        tgt: str
        if target_column is not None:
            tgt = target_column
        elif target_specs is not None:
            tgt = _resolve_column(out, target_specs)
        else:
            raise ValueError("Either target_column or target_specs must be provided.")

        # Resolve features
        if feature_specs is not None:
            features = _resolve_feature_columns(out, feature_specs)
            # Exclude target from features
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_scan_info(scan: Any) -> Dict[str, Any]:
    """Extract metadata from a ScanData object."""
    info: Dict[str, Any] = {}
    if hasattr(scan, "paths") and hasattr(scan.paths, "tag"):
        tag = scan.paths.tag
        if hasattr(tag, "_asdict"):
            info.update(tag._asdict())
        elif hasattr(tag, "__dict__"):
            info.update(tag.__dict__)
    return info


def _resolve_column(df: pd.DataFrame, specs: Sequence[str]) -> str:
    """Resolve a single column; try each spec until :func:`resolve_col` succeeds."""
    for spec in specs:
        try:
            return resolve_col(df, spec)
        except ValueError:
            continue
    raise ValueError(f"Could not resolve any column from specs: {list(specs)}")


def _resolve_feature_columns(df: pd.DataFrame, specs: Sequence[str]) -> List[str]:
    """One resolved column per spec (deduplicated), using :func:`resolve_col`."""
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
