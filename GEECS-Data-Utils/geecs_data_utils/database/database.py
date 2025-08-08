"""
ScanDatabase: Unified metadata interface for GEECS scan records.

This class provides lazy and in-memory filtering of structured scan metadata
stored as a partitioned Parquet dataset. It supports efficient on-disk filtering
(via PyArrow expressions) and automatic in-memory loading for advanced operations
on JSON-based columns like `ecs_dump` and `scan_metadata`.

Typical usage:
--------------
db = ScanDatabase(path)
db.filter_by_date_range(...)
db.filter_by_device(...)
db.load()
results = db.query_ecs_device("U_HexapodY")

See Also
--------
ScanEntry : Model representing individual scan records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Callable
from datetime import date

import pyarrow.dataset as ds
import pandas as pd

from geecs_data_utils.database.entries import ScanEntry


class ScanDatabase:
    """
    Unified Parquet-backed database for querying GEECS scan metadata.

    This class provides a high-performance interface for storing, loading, and
    querying scan metadata collected from GEECS experiments. Data is stored in
    Apache Parquet format for efficient on-disk filtering using PyArrow, while
    also supporting in-memory loading for more complex queries.

    The interface is designed to be consistent regardless of whether the
    underlying operations are performed in-memory or on-disk, allowing for
    seamless switching between fast disk-backed filtering and full in-memory
    analysis.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the Parquet file containing the scan database.
    lazy : bool, optional
        If True, the database is not loaded into memory until required.
        Disk-based filtering via PyArrow will be used when possible.
        Defaults to True.
    memory_threshold : int, optional
        Maximum number of rows to load into memory automatically when
        performing queries. If the filtered result exceeds this threshold,
        the result will remain in a PyArrow Table unless explicitly converted.
        Defaults to 100_000.
    verbose : bool, optional
        If True, print diagnostic and performance information during
        filtering and loading. Defaults to False.

    Attributes
    ----------
    parquet_path : Path
        Absolute path to the Parquet file containing the scan database.
    lazy : bool
        Whether the database defers loading into memory.
    memory_threshold : int
        Row count limit for automatic in-memory conversion.
    verbose : bool
        Verbosity flag for logging diagnostic information.
    _table : pyarrow.Table or None
        Internal reference to the loaded in-memory table, if available.

    Notes
    -----
    - Filtering is supported on all top-level columns (e.g. ``year``,
      ``month``, ``day``, ``number``, ``scan_parameter``).
    - JSON-like fields (e.g. ``scan_metadata``) are stored as strings
      for compatibility and full-text matching, but cannot be filtered
      efficiently on-disk.
    - All methods return consistent results regardless of storage mode;
      however, on-disk filtering is generally faster for large datasets.

    Examples
    --------
    Load a database and filter by scan parameter::

        db = ScanDatabase("/data/Undulator/scan_database.parquet")
        result = db.filter(scan_parameter="U_ESP_JetXYZ:Position.Axis 3")

    Load fully into memory for iterative analysis::

        db = ScanDatabase("/data/Undulator/scan_database.parquet", lazy=False)
        all_scans = db.to_pandas()

    Chain filters and retrieve as a Pandas DataFrame::

        result_df = (
            db.filter(year=2025, month=8)
              .filter(background=False)
              .to_pandas()
        )
    """

    def __init__(self, dataset_path: Path):
        """
        Initialize a ScanDatabase with the specified Parquet dataset path.

        Parameters
        ----------
        dataset_path : Path
            Path to the Parquet dataset directory.
        """
        self.dataset_path = Path(dataset_path)
        self._dataset = ds.dataset(
            self.dataset_path, format="parquet", partitioning="hive"
        )
        self._filters: List[ds.Expression] = []
        self._memory_scans: Optional[List[ScanEntry]] = None
        self._is_loaded: bool = False

    def __repr__(self):
        """
        Return string representation of the ScanDatabase instance.

        Returns
        -------
        str
            Representation showing load state and number of filters.
        """
        return (
            f"<ScanDatabase (loaded={self._is_loaded}, filters={len(self._filters)})>"
        )

    def filter_by_date_range(self, start: date, end: date) -> None:
        """
        Add a filter to select scans within a specific date range.

        Parameters
        ----------
        start : date
            Start date (inclusive).
        end : date
            End date (inclusive).

        Notes
        -----
        If ``start`` is after ``end``, the range is normalized automatically.
        """
        # Normalize dates so start <= end
        if start > end:
            start, end = end, start

        def _ymd_value(y: int, m: int, d: int) -> int:
            """Convert (year, month, day) to sortable yyyymmdd integer."""
            return y * 10000 + m * 100 + d

        expr = (
            ds.field("year") * 10000 + ds.field("month") * 100 + ds.field("day")
            >= _ymd_value(start.year, start.month, start.day)
        ) & (
            ds.field("year") * 10000 + ds.field("month") * 100 + ds.field("day")
            <= _ymd_value(end.year, end.month, end.day)
        )
        self._filters.append(expr)

    def filter_by_device(self, device_name: str) -> None:
        """
        Add a filter to select scans that include a given non-scalar device.

        Parameters
        ----------
        device_name : str
            Name of the device to filter by.
        """
        expr = ds.field("non_scalar_devices").list_contains(device_name)
        self._filters.append(expr)

    def filter_by_experiment(self, experiment: str) -> None:
        """
        Add a filter to select scans from a specific experiment.

        Parameters
        ----------
        experiment : str
            Experiment name to match.
        """
        expr = ds.field("experiment") == experiment
        self._filters.append(expr)

    def filter_by_scan_param(self, param: str) -> None:
        """
        Add a filter based on the scan parameter string in scan metadata.

        Parameters
        ----------
        param : str
            Scan parameter substring to search for.
        """
        expr = ds.field("scan_metadata_str").str_contains(param, ignore_case=True)
        self._filters.append(expr)

    def add_filter(self, expression: ds.Expression) -> None:
        """
        Add a custom PyArrow filter expression.

        Parameters
        ----------
        expression : pyarrow.dataset.Expression
            A valid PyArrow filter expression.
        """
        self._filters.append(expression)

    def reset_filters(self) -> None:
        """Reset all filters and unload any loaded memory scans."""
        self._filters.clear()
        self._memory_scans = None
        self._is_loaded = False

    def describe_filters(self) -> None:
        """Print all currently applied filter expressions."""
        for i, f in enumerate(self._filters):
            print(f"[{i}] {f}")

    def _combined_filter(self) -> Optional[ds.Expression]:
        """
        Combine all filter expressions into a single PyArrow filter.

        Returns
        -------
        pyarrow.dataset.Expression or None
            Combined filter expression, or None if no filters.
        """
        if not self._filters:
            return None
        expr = self._filters[0]
        for f in self._filters[1:]:
            expr = expr & f
        return expr

    def load(self) -> None:
        """Load filtered scan entries from disk into memory."""
        self._ensure_loaded()

    def _ensure_loaded(self):
        """Load and decode scan entries if not already loaded into memory."""
        if self._is_loaded:
            return
        table = self._dataset.to_table(filter=self._combined_filter())
        df = table.to_pandas()

        for col in ["scan_metadata", "ecs_dump"]:
            if col in df.columns:
                df[col] = df[col].apply(json.loads)

        self._memory_scans = [
            ScanEntry.model_validate(
                {
                    **row.to_dict(),
                    "scan_tag": {
                        "year": row["year"],
                        "month": row["month"],
                        "day": row["day"],
                        "number": row["number"],
                        "experiment": row["experiment"],
                    },
                }
            )
            for _, row in df.iterrows()
        ]
        self._is_loaded = True

    def query_ecs_device(self, device_name: str) -> List[ScanEntry]:
        """
        Filter loaded scans to include only those with a specific ECS device.

        Parameters
        ----------
        device_name : str
            Name of the ECS device to match.

        Returns
        -------
        List[ScanEntry]
            Matching scan entries.
        """
        self._ensure_loaded()
        return [s for s in self._memory_scans if device_name in s.ecs_dump.device_names]

    def query_by_scan_param(self, param: str) -> List[ScanEntry]:
        """
        Filter loaded scans by exact match on scan parameter.

        Parameters
        ----------
        param : str
            Scan parameter value to match.

        Returns
        -------
        List[ScanEntry]
            Matching scan entries.
        """
        self._ensure_loaded()
        return [
            s for s in self._memory_scans if s.scan_metadata.scan_parameter == param
        ]

    def search_notes(self, keyword: str) -> List[ScanEntry]:
        """
        Search notes field for a case-insensitive keyword.

        Parameters
        ----------
        keyword : str
            Keyword to search in the notes.

        Returns
        -------
        List[ScanEntry]
            Matching scan entries.
        """
        self._ensure_loaded()
        return [
            s
            for s in self._memory_scans
            if s.notes and keyword.lower() in s.notes.lower()
        ]

    def filter_memory(self, fn: Callable[[ScanEntry], bool]) -> None:
        """
        Filter loaded scan entries using a custom predicate function.

        Parameters
        ----------
        fn : Callable[[ScanEntry], bool]
            Function that returns True for entries to keep.
        """
        self._ensure_loaded()
        self._memory_scans = [s for s in self._memory_scans if fn(s)]

    @property
    def memory_scans(self) -> List[ScanEntry]:
        """
        Access the list of loaded ScanEntry records.

        Returns
        -------
        List[ScanEntry]
            The currently loaded scan entries.
        """
        self._ensure_loaded()
        return self._memory_scans

    def count(self) -> int:
        """
        Count the number of scan entries (filtered or loaded).

        Returns
        -------
        int
            Number of matching entries.
        """
        if self._is_loaded:
            return len(self._memory_scans)
        return self._dataset.count(filter=self._combined_filter())

    def preview(self, n: int = 5) -> pd.DataFrame:
        """
        Return a preview of the filtered scan records.

        Parameters
        ----------
        n : int, optional
            Number of rows to return (default is 5).

        Returns
        -------
        pandas.DataFrame
            Table preview.
        """
        if self._is_loaded:
            return pd.DataFrame([s.model_dump() for s in self._memory_scans[:n]])
        table = self._dataset.to_table(filter=self._combined_filter())
        return table.to_pandas().head(n)

    def to_json_file(self, path: str | Path) -> None:
        """
        Export the currently loaded scan entries to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        self._ensure_loaded()
        path = Path(path)
        path.write_text(
            json.dumps([s.model_dump() for s in self._memory_scans], indent=2)
        )
