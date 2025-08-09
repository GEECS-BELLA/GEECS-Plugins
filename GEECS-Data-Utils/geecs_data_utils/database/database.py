"""
Provides a searchable interface to a partitioned scans database stored in Parquet format.

This module defines the `ScanDatabase` class, which enables efficient querying of
experiment scan metadata stored as a Hive-style partitioned Parquet dataset (partitioned
by year and month). Users can apply date ranges, experiment names, scan parameters,
non-scalar device filters, and ECS variable conditions, either interactively or through
predefined named filters loaded from a YAML configuration.

The design optimizes I/O by loading only the relevant year/month partitions before applying
in-memory filters with pandas, avoiding the overhead of full dataset reads. Named filters
allow for common query patterns to be persisted and reused across sessions.

Classes
-------
ScanDatabase
    Interface for loading, filtering, and previewing scan metadata from a
    partitioned Parquet dataset.

See Also
--------
pandas.read_parquet : Underlying function used for loading Parquet partitions.
pyarrow.parquet : Backend engine for reading partitioned Parquet files.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, List, Tuple, Callable, Optional
import pandas as pd
import json


class ScanDatabase:
    """Partition-aware Parquet reader with fast date pruning and composable filters."""

    def __init__(self, parquet_root: Union[str, Path]):
        """Initialize with the root directory of the Hive-partitioned dataset."""
        self.root = Path(parquet_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Parquet root does not exist: {self.root}")
        self._date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._df_filters: List[Callable[[pd.DataFrame], pd.DataFrame]] = []

    # -----------------------
    # Filters
    # -----------------------
    def date_range(
        self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]
    ) -> "ScanDatabase":
        """
        Set an inclusive date range for partition pruning and day-level trim.

        Parameters
        ----------
        start : str or pandas.Timestamp
            Start date (inclusive).
        end : str or pandas.Timestamp
            End date (inclusive).

        Returns
        -------
        ScanDatabase
            Self for chaining.
        """
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s > e:
            s, e = e, s
        self._date_range = (s, e)
        return self

    def filter_scan_parameter_contains(
        self, substring: str, *, case: bool = False
    ) -> "ScanDatabase":
        """Filter rows where `scan_parameter` contains a substring."""
        sub = substring

        def _f(df: pd.DataFrame) -> pd.DataFrame:
            if "scan_parameter" not in df.columns:
                return df
            return df[
                df["scan_parameter"]
                .astype("string")
                .str.contains(sub, case=case, na=False)
            ]

        self._df_filters.append(_f)
        return self

    def filter_experiment_equals(self, name: str) -> "ScanDatabase":
        """Filter rows where `experiment` equals the provided name."""
        exp = name

        def _f(df: pd.DataFrame) -> pd.DataFrame:
            if "experiment" not in df.columns:
                return df
            return df[df["experiment"] == exp]

        self._df_filters.append(_f)
        return self

    def filter_device_contains(self, device_substring: str) -> "ScanDatabase":
        """Filter rows where any `non_scalar_devices` item contains the substring (case-insensitive)."""
        needle = device_substring.lower()

        def _f(df: pd.DataFrame) -> pd.DataFrame:
            col = "non_scalar_devices"
            if col not in df.columns:
                return df

            def _has(lst) -> bool:
                if isinstance(lst, list):
                    return any(isinstance(x, str) and needle in x.lower() for x in lst)
                if isinstance(lst, str):  # stray single string
                    return needle in lst.lower()
                return False

            return df[df[col].apply(_has)]

        self._df_filters.append(_f)
        return self

    @staticmethod
    def _ecs_json_load(s: object):
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return None
        if not isinstance(s, str) or not s.strip():
            return None
        try:
            return json.loads(s)
        except Exception:
            return None

    @staticmethod
    def _i_contains(hay: str, needle: str) -> bool:
        try:
            return needle.lower() in hay.lower()
        except Exception:
            return False

    @staticmethod
    def _ecs_values(row_ecs: object, device_like: str, variable_like: str):
        obj = (
            row_ecs
            if isinstance(row_ecs, dict)
            else ScanDatabase._ecs_json_load(row_ecs)
        )
        if not isinstance(obj, dict):
            return
        devices = obj.get("devices")
        if not isinstance(devices, list):
            return
        for rec in devices:
            if not isinstance(rec, dict):
                continue
            name = rec.get("name", "")
            if not ScanDatabase._i_contains(str(name), device_like):
                continue
            params = rec.get("parameters")
            if not isinstance(params, dict):
                continue
            for var_name, val in params.items():
                if ScanDatabase._i_contains(str(var_name), variable_like):
                    yield val

    def filter_ecs_value_within(
        self, device_like: str, variable_like: str, target: float, tol: float
    ) -> "ScanDatabase":
        """Keep rows where any matching ECS value is within target±tol (values are parsed as floats)."""
        tgt, tol = float(target), float(tol)

        def _ok_num(v) -> bool:
            try:
                return abs(float(v) - tgt) <= tol
            except Exception:
                return False

        def _f(df: pd.DataFrame) -> pd.DataFrame:
            if "ecs_dump" not in df.columns:
                return df
            mask = df["ecs_dump"].apply(
                lambda s: any(
                    _ok_num(v)
                    for v in ScanDatabase._ecs_values(s, device_like, variable_like)
                )
            )
            return df[mask]

        self._df_filters.append(_f)
        return self

    def filter_ecs_value_contains(
        self, device_like: str, variable_like: str, text: str, case: bool = False
    ) -> "ScanDatabase":
        """Keep rows where any matching ECS value contains the given text."""
        if case:

            def _ok_str(v) -> bool:
                return isinstance(v, str) and (text in v)
        else:
            t = text.lower()

            def _ok_str(v) -> bool:
                return isinstance(v, str) and (t in v.lower())

        def _f(df: pd.DataFrame) -> pd.DataFrame:
            if "ecs_dump" not in df.columns:
                return df
            mask = df["ecs_dump"].apply(
                lambda s: any(
                    _ok_str(v)
                    for v in ScanDatabase._ecs_values(s, device_like, variable_like)
                )
            )
            return df[mask]

        self._df_filters.append(_f)
        return self

    def where(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> "ScanDatabase":
        """Add a custom pandas filter callable(df) -> df."""
        self._df_filters.append(fn)
        return self

    def reset(self) -> "ScanDatabase":
        """Clear date range and all filters."""
        self._date_range = None
        self._df_filters.clear()
        return self

    # -----------------------
    # Internal helpers
    # -----------------------
    def _all_partitions(self) -> List[Tuple[int, int]]:
        """Return all (year, month) partitions present on disk."""
        parts: List[Tuple[int, int]] = []
        for ydir in self.root.glob("year=*"):
            try:
                y = int(ydir.name.split("=", 1)[1])
            except Exception:
                continue
            for mdir in ydir.glob("month=*"):
                try:
                    m = int(mdir.name.split("=", 1)[1])
                    parts.append((y, m))
                except Exception:
                    continue
        parts.sort()
        return parts

    def _partitions_to_read(self) -> List[Path]:
        """Compute partition directories to load based on the date range."""
        if self._date_range is None:
            # No pruning → all partitions
            return [
                self.root / f"year={y}" / f"month={m}"
                for (y, m) in self._all_partitions()
                if (self.root / f"year={y}" / f"month={m}").exists()
            ]

        s, e = self._date_range
        months = pd.period_range(s, e, freq="M")
        paths: List[Path] = []
        for p in months:
            folder = self.root / f"year={p.year}" / f"month={p.month}"
            if folder.exists():
                paths.append(folder)
        return paths

    def _load_partitions(self) -> pd.DataFrame:
        """Load selected partitions, trim by day if needed, then apply filters."""
        dfs: List[pd.DataFrame] = []
        for folder in self._partitions_to_read():
            try:
                dfs.append(pd.read_parquet(folder))
            except Exception as e:
                print(f"[WARN] Skipping partition {folder}: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Day-level trim inside selected months
        if self._date_range is not None and all(
            c in df.columns for c in ("year", "month", "day")
        ):
            s, e = self._date_range
            ts = pd.to_datetime(
                {"year": df["year"], "month": df["month"], "day": df["day"]},
                errors="coerce",
            )
            df = df[(ts >= s) & (ts <= e)].copy()

        # Apply queued pandas filters
        for f in self._df_filters:
            df = f(df)
            # print(f"[DEBUG] filter {getattr(f, '__name__', 'callable')} kept {after}/{before}")
        return df

    # -----------------------
    # Public API
    # -----------------------
    def preview(self, n: int = 5) -> pd.DataFrame:
        """Return the first n rows after applying filters."""
        return self._load_partitions().head(n)

    def to_df(self) -> pd.DataFrame:
        """Return the fully filtered DataFrame."""
        return self._load_partitions()
