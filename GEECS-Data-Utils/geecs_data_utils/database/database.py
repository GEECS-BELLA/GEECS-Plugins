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
from typing import Union, List, Tuple, Callable, Optional, Any, Mapping
import pandas as pd
import json
import yaml


class ScanDatabase:
    """Partition-aware Parquet reader with fast date pruning and composable filters."""

    def __init__(
        self, parquet_root: Union[str, Path], *, autoload_presets: bool = True
    ):
        """Initialize with the root directory of the Hive-partitioned dataset."""
        self.root = Path(parquet_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Parquet root does not exist: {self.root}")

        self._date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self._df_filters: List[Callable[[pd.DataFrame], pd.DataFrame]] = []
        self._named_specs: dict[
            str, dict[str, Any]
        ] = {}  # raw specs from YAML or registered

        if autoload_presets:
            self._autoload_presets()

    # -----------------------
    # Preset filter loading
    # -----------------------
    def _preset_default_path(self) -> Path:
        """Return default filters YAML path inside the repo/package."""
        return Path(__file__).resolve().parent / "filters" / "scan_filters.yml"

    def _autoload_presets(self) -> None:
        """Autoload named filter presets from filters/scan_filters.yml if present."""
        preset_path = self._preset_default_path()
        if preset_path.exists():
            try:
                self.load_named_filters(preset_path)
            except Exception as e:
                # Non-fatal: keep operating without presets
                print(f"[WARN] Failed to load presets from {preset_path}: {e}")

    def load_named_filters(self, path: Union[str, Path]) -> "ScanDatabase":
        """Load named filter specs (including composites) from a YAML file."""
        p = Path(path)
        data = yaml.safe_load(p.read_text())
        specs = (data or {}).get("filters") or {}
        if not isinstance(specs, Mapping):
            raise ValueError("YAML must contain a top-level 'filters' mapping")
        self._named_specs = dict(specs)
        return self

    def register_named_filter(
        self,
        name: str,
        *,
        kind: str | None = None,
        args: Mapping[str, Any] | None = None,
        subfilters: list[str] | None = None,
    ) -> "ScanDatabase":
        """Register a single named filter spec at runtime (supports composites)."""
        if not name:
            raise ValueError("Filter name must be non-empty")
        if kind == "composite":
            self._named_specs[name] = {
                "kind": "composite",
                "subfilters": list(subfilters or []),
            }
        elif kind:
            self._named_specs[name] = {"kind": kind, "args": dict(args or {})}
        else:
            raise ValueError(
                "Provide 'kind' (and 'args') or kind='composite' with 'subfilters'"
            )
        return self

    def list_named_filters(self) -> list[str]:
        """Return the list of available named filter names."""
        return sorted(self._named_specs.keys())

    def describe_named_filter(self, name: str) -> dict[str, Any]:
        """Return the raw spec dict for a named filter."""
        spec = self._named_specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown named filter: '{name}'")
        return spec

    def apply(self, *names: str) -> "ScanDatabase":
        """Apply one or more named filters (supports composite filters)."""
        if not names:
            return self
        seen: set[str] = set()
        for nm in names:
            self._apply_named(nm, seen)
        return self

    def _apply_named(self, name: str, seen: set[str]) -> None:
        if name in seen:
            raise ValueError(f"Cycle detected in composite filters at '{name}'")
        spec = self._named_specs.get(name)
        if not spec:
            raise KeyError(f"Unknown named filter: '{name}'")
        seen.add(name)
        kind = spec.get("kind")
        if kind == "composite":
            for sub in spec.get("subfilters", []) or []:
                self._apply_named(sub, seen)
            return
        self._instantiate_and_enqueue(kind, spec.get("args") or {})

    def _instantiate_and_enqueue(self, kind: str, args: Mapping[str, Any]) -> None:
        """Map 'kind' to the corresponding filter method and enqueue it."""
        if kind == "ecs_value_within":
            self.filter_ecs_value_within(**args)
        elif kind == "ecs_value_contains":
            self.filter_ecs_value_contains(**args)
        elif kind == "scan_parameter_contains":
            self.filter_scan_parameter_contains(**args)
        elif kind == "experiment_equals":
            self.filter_experiment_equals(**args)
        elif kind == "device_contains":
            self.filter_device_contains(**args)
        else:
            raise ValueError(f"Unknown filter kind: {kind}")

    # -----------------------
    # Filters (chainable)
    # -----------------------
    def date_range(
        self, start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp]
    ) -> "ScanDatabase":
        """Set an inclusive date range for partition pruning and day-level trim."""
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
                if isinstance(lst, str):
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
        """Keep rows where any matching ECS value is within target±tol (values parsed as floats)."""
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
    # Partition I/O helpers
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

        # day-level trim
        if self._date_range is not None and all(
            c in df.columns for c in ("year", "month", "day")
        ):
            s, e = self._date_range
            ts = pd.to_datetime(
                {"year": df["year"], "month": df["month"], "day": df["day"]},
                errors="coerce",
            )
            df = df[(ts >= s) & (ts <= e)].copy()

        # apply pandas filters
        for f in self._df_filters:
            df = f(df)
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
