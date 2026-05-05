"""Generic dataframe assembly and cleaning utilities for scan datasets.

Notes
-----
**Concat semantics.** :meth:`DatasetBuilder.from_scans` uses ``pandas.concat``
with default outer alignment on columns. If per-scan scalar tables use
different column sets (or MultiIndex levels differ), the merged frame can
contain **extra columns filled with NaN** on rows from scans that did not have
that column, or **dtype coercions** where pandas unifies types. This is normal
pandas behavior, not a schema merge step. For strict column parity, normalize
or validate frames before concatenation (outside this module or in a future
helper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Sequence

import pandas as pd

from geecs_data_utils.data.cleaning import (
    OutlierConfig,
    RowFilterSpec,
    apply_outlier_config,
    apply_row_filters,
)


@dataclass
class LoadScansReport:
    """Outcome of loading many scan numbers for one calendar date.

    Attributes
    ----------
    scans
        Successfully loaded :class:`~geecs_data_utils.ScanData` instances
        (each with a non-``None`` ``data_frame``).
    numbers_loaded
        Scan numbers corresponding to ``scans`` (same order).
    skipped
        ``(number, reason)`` for each number that did not yield a usable frame:
        load errors (``FileNotFoundError``, ``ValueError``) or an empty
        ``data_frame`` after load.
    """

    scans: list[Any] = field(default_factory=list)
    numbers_loaded: list[int] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class DatasetFrame:
    """Result of assembling and optionally cleaning a table.

    Attributes
    ----------
    frame
        The working DataFrame (after any outlier / filter / dropna steps).
    scan_info
        For ``from_scan``: tag fields from :meth:`DatasetBuilder.extract_scan_info`
        (e.g. scan number, experiment), when the input had ``paths.tag``.
        For ``from_scans``: ``{"scans": [...], "total_scans": N}`` with one dict
        per contributing scan. Empty when built from ``from_dataframe`` only.
    rows_raw
        Row count of the concatenated or single-scan frame **before** cleaning.
    rows_final
        Row count of ``frame`` after cleaning. Compare to ``rows_raw`` to see
        how many rows were removed by outliers, filters, and optional ``dropna``.
    load_report
        Set when the frame was built via :meth:`DatasetBuilder.from_date_scan_numbers`:
        which scan numbers loaded vs were skipped while resolving paths.
    """

    frame: pd.DataFrame
    scan_info: Dict[str, Any] = field(default_factory=dict)
    rows_raw: int = 0
    rows_final: int = 0
    load_report: LoadScansReport | None = None


class DatasetBuilder:
    """Concatenate scan scalar tables and run shared cleaning.

    Inputs are duck-typed: anything with a ``data_frame`` attribute works;
    :class:`~geecs_data_utils.ScanData` is the usual type. Heavy lifting uses
    :meth:`prepare_frame` (outliers → row filters → optional ``dropna``).

    :class:`~geecs_data_utils.ml.dataset.BeamPredictionDatasetBuilder` calls these
    entry points with ``dropna=False``, then applies ML-specific column selection
    and drops NaNs only on the chosen columns.

    ``ScanData`` is imported inside load helpers so this module stays importable
    without pulling scan IO unless those paths are used.
    """

    @staticmethod
    def extract_scan_info(scan: Any) -> Dict[str, Any]:
        """Copy scan identity fields from ``scan.paths.tag`` into a plain dict.

        Used so assembled results carry provenance (date, scan number, experiment,
        etc.) without holding references to full :class:`~geecs_data_utils.ScanData`
        objects. Non-``ScanData`` inputs typically yield an empty dict.
        """
        info: Dict[str, Any] = {}
        if hasattr(scan, "paths") and hasattr(scan.paths, "tag"):
            tag = scan.paths.tag
            if hasattr(tag, "_asdict"):
                info.update(tag._asdict())
            elif hasattr(tag, "__dict__"):
                info.update(tag.__dict__)
        return info

    @staticmethod
    def load_scans_from_date_report(
        *,
        year: int,
        month: int,
        day: int,
        experiment: str,
        numbers: Iterable[int],
        base_directory: Path | None = None,
        load_scalars: bool = True,
        source: Literal["sfile", "tdms"] = "sfile",
        append_paths: bool = False,
        on_missing: Literal["skip", "raise"] = "skip",
    ) -> LoadScansReport:
        """Try each scan number; record successes and skips (see :class:`LoadScansReport`).

        Parameters
        ----------
        numbers
            Scan indices to try (e.g. ``range(1, 55)``).
        on_missing
            ``"skip"`` records failures in ``LoadScansReport.skipped`` and continues.
            ``"raise"`` re-raises the first ``FileNotFoundError`` or ``ValueError``
            from :meth:`~geecs_data_utils.ScanData.from_date`, or raises
            ``ValueError`` if a scan loads but ``data_frame`` is ``None``.
        append_paths
            Passed to :meth:`~geecs_data_utils.ScanData.from_date` (often ``False``
            when you only need scalar columns for concatenation).

        Notes
        -----
        Side-effect free regarding disk beyond calling ``ScanData.from_date`` per number.
        """
        from geecs_data_utils.scan_data import ScanData

        scans: list[Any] = []
        numbers_loaded: list[int] = []
        skipped: list[tuple[int, str]] = []

        for number in numbers:
            try:
                sd = ScanData.from_date(
                    year=year,
                    month=month,
                    day=day,
                    number=number,
                    experiment=experiment,
                    base_directory=base_directory,
                    load_scalars=load_scalars,
                    source=source,
                    append_paths=append_paths,
                )
            except (FileNotFoundError, ValueError) as exc:
                if on_missing == "raise":
                    raise
                skipped.append((number, f"{type(exc).__name__}: {exc}"))
                continue
            if sd.data_frame is None:
                if on_missing == "raise":
                    raise ValueError(
                        f"Scan {number} loaded but data_frame is None "
                        "(try load_scalars=True or check data availability)."
                    )
                skipped.append(
                    (number, "No data_frame after load (empty or load_scalars=False).")
                )
                continue
            scans.append(sd)
            numbers_loaded.append(number)

        return LoadScansReport(
            scans=scans,
            numbers_loaded=numbers_loaded,
            skipped=skipped,
        )

    @staticmethod
    def load_scans_from_date(
        *,
        year: int,
        month: int,
        day: int,
        experiment: str,
        numbers: Iterable[int],
        base_directory: Path | None = None,
        load_scalars: bool = True,
        source: Literal["sfile", "tdms"] = "sfile",
        append_paths: bool = False,
        on_missing: Literal["skip", "raise"] = "skip",
    ) -> list[Any]:
        """Load scans for a date; return successful :class:`~geecs_data_utils.ScanData` instances.

        Order matches iteration over ``numbers`` (missing scans omitted).

        For skipped numbers and reasons, use :meth:`load_scans_from_date_report`.
        """
        return DatasetBuilder.load_scans_from_date_report(
            year=year,
            month=month,
            day=day,
            experiment=experiment,
            numbers=numbers,
            base_directory=base_directory,
            load_scalars=load_scalars,
            source=source,
            append_paths=append_paths,
            on_missing=on_missing,
        ).scans

    @staticmethod
    def from_date_scan_numbers(
        *,
        year: int,
        month: int,
        day: int,
        experiment: str,
        numbers: Iterable[int],
        base_directory: Path | None = None,
        load_scalars: bool = True,
        source: Literal["sfile", "tdms"] = "sfile",
        append_paths: bool = False,
        on_missing: Literal["skip", "raise"] = "skip",
        filters: list[RowFilterSpec] | None = None,
        outlier_config: OutlierConfig | None = None,
        dropna: bool = False,
    ) -> DatasetFrame:
        """Load many scan numbers for one date, concatenate, then run :meth:`prepare_frame`.

        Equivalent to :meth:`load_scans_from_date_report` plus :meth:`from_scans`,
        with :attr:`DatasetFrame.load_report` set for troubleshooting skips.

        Raises
        ------
        ValueError
            If no scans could be loaded (all numbers missing or empty frames).
        """
        report = DatasetBuilder.load_scans_from_date_report(
            year=year,
            month=month,
            day=day,
            experiment=experiment,
            numbers=numbers,
            base_directory=base_directory,
            load_scalars=load_scalars,
            source=source,
            append_paths=append_paths,
            on_missing=on_missing,
        )
        if not report.scans:
            raise ValueError(
                "No scans could be loaded for the given date, experiment, and numbers. "
                f"Skipped ({len(report.skipped)}): {report.skipped[:10]}"
                + (" ..." if len(report.skipped) > 10 else "")
            )
        out = DatasetBuilder.from_scans(
            report.scans,
            filters=filters,
            outlier_config=outlier_config,
            dropna=dropna,
        )
        out.load_report = report
        return out

    @staticmethod
    def from_scan(
        scan: Any,
        *,
        filters: list[RowFilterSpec] | None = None,
        outlier_config: OutlierConfig | None = None,
        dropna: bool = False,
    ) -> DatasetFrame:
        """Copy ``scan.data_frame``, apply :meth:`prepare_frame`, attach ``scan_info``."""
        df = getattr(scan, "data_frame", None)
        if df is None:
            raise ValueError("ScanData has no loaded data_frame.")

        cleaned = DatasetBuilder.prepare_frame(
            df, filters=filters, outlier_config=outlier_config, dropna=dropna
        )
        return DatasetFrame(
            frame=cleaned,
            scan_info=DatasetBuilder.extract_scan_info(scan),
            rows_raw=len(df),
            rows_final=len(cleaned),
        )

    @staticmethod
    def from_scans(
        scans: Sequence[Any],
        *,
        filters: list[RowFilterSpec] | None = None,
        outlier_config: OutlierConfig | None = None,
        dropna: bool = False,
    ) -> DatasetFrame:
        """Assemble and optionally clean one frame from multiple scans.

        Frames are concatenated with ``pandas.concat(..., ignore_index=True)``.
        See the module docstring for column-alignment / NaN behavior when schemas
        differ across scans.
        """
        frames: list[pd.DataFrame] = []
        scan_infos: list[Dict[str, Any]] = []
        for s in scans:
            df = getattr(s, "data_frame", None)
            if df is None:
                continue
            frames.append(df)
            scan_infos.append(DatasetBuilder.extract_scan_info(s))

        if not frames:
            raise ValueError("No scans with loaded data_frame provided.")

        merged = pd.concat(frames, ignore_index=True)
        cleaned = DatasetBuilder.prepare_frame(
            merged, filters=filters, outlier_config=outlier_config, dropna=dropna
        )
        return DatasetFrame(
            frame=cleaned,
            scan_info={"scans": scan_infos, "total_scans": len(scan_infos)},
            rows_raw=len(merged),
            rows_final=len(cleaned),
        )

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        *,
        filters: list[RowFilterSpec] | None = None,
        outlier_config: OutlierConfig | None = None,
        dropna: bool = False,
    ) -> DatasetFrame:
        """Apply :meth:`prepare_frame` to an arbitrary table (no ``scan_info``)."""
        cleaned = DatasetBuilder.prepare_frame(
            df, filters=filters, outlier_config=outlier_config, dropna=dropna
        )
        return DatasetFrame(frame=cleaned, rows_raw=len(df), rows_final=len(cleaned))

    @staticmethod
    def prepare_frame(
        df: pd.DataFrame,
        *,
        filters: list[RowFilterSpec] | None = None,
        outlier_config: OutlierConfig | None = None,
        dropna: bool = False,
    ) -> pd.DataFrame:
        """Ordered pipeline: optional outliers → optional row filters → optional ``dropna``.

        Used by :class:`DatasetBuilder` and by ML builders that disable ``dropna``
        here so NaNs can be handled after column selection.

        Returns
        -------
        pandas.DataFrame
            A copy of ``df`` after the requested steps (possibly unchanged length).
        """
        out = df.copy()
        if outlier_config is not None:
            out = apply_outlier_config(out, outlier_config)
        if filters:
            out = apply_row_filters(out, filters)
        if dropna:
            out = out.dropna()
        return out
