"""
Provides a searchable interface to a partitioned scans scans_database stored in Parquet format.

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
import pyarrow.dataset as ds
import pandas as pd

import json
import yaml
from pydantic import ValidationError

from .filter_models import FilterSpec, parse_filter_spec_from_yaml


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
        self._named_specs: dict[str, FilterSpec] = {}  # Pydantic-validated filter specs

        # Extract experiment name from path
        self.experiment = self._infer_experiment()

        if autoload_presets:
            self._autoload_experiment_presets()

    # -----------------------
    # Preset filter loading
    # -----------------------
    def _infer_experiment(self) -> Optional[str]:
        """Extract experiment name from path like /data/Undulator/scan_database_parquet."""
        # Look for parent directory that's not 'scan_database_parquet'
        for part in reversed(self.root.parts):
            if part != "scan_database_parquet":
                return part
        return None

    def _preset_default_path(self) -> Path:
        """Return default filters YAML path inside the repo/package."""
        return Path(__file__).resolve().parent / "filters" / "scan_filters.yml"

    def _autoload_experiment_presets(self) -> None:
        """Load experiment-specific filter presets, falling back to generic ones."""
        if not self.experiment:
            # No experiment detected, try generic presets
            self._autoload_generic_presets()
            return

        # Try experiment-specific file first
        exp_preset = (
            self._preset_default_path().parent / f"{self.experiment.lower()}.yml"
        )
        if exp_preset.exists():
            try:
                self.load_named_filters(exp_preset)
                return
            except Exception as e:
                print(
                    f"[WARN] Failed to load {self.experiment} presets from {exp_preset}: {e}"
                )

        # Fallback to generic presets
        self._autoload_generic_presets()

    def _autoload_generic_presets(self) -> None:
        """Autoload named filter presets from filters/scan_filters.yml if present."""
        preset_path = self._preset_default_path()
        if preset_path.exists():
            try:
                self.load_named_filters(preset_path)
            except Exception as e:
                # Non-fatal: keep operating without presets
                print(f"[WARN] Failed to load generic presets from {preset_path}: {e}")

    def _apply_dated_filter_spanning(
        self, name: str, date_range: Tuple[pd.Timestamp, pd.Timestamp]
    ) -> None:
        """Apply filter using all applicable versions across the date range."""
        filter_spec = self._named_specs.get(name)
        if not filter_spec:
            raise KeyError(f"Unknown named filter: '{name}'")

        # Get applicable versions for the date range
        start_date, end_date = date_range[0].date(), date_range[1].date()
        applicable_versions = filter_spec.get_versions_for_range(start_date, end_date)

        if not applicable_versions:
            raise ValueError(
                f"No valid versions of '{name}' for the specified date range"
            )

        # Create the composite filter function
        composite_filter = self._create_composite_dated_filter_pydantic(
            applicable_versions, date_range
        )
        self._df_filters.append(composite_filter)

    def _resolve_dated_filter(
        self,
        name: str,
        query_date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> dict:
        """
        Resolve a filter name to the appropriate version based on date validity.

        Parameters
        ----------
        name : str
            Filter name to resolve
        query_date_range : tuple of pd.Timestamp, optional
            Date range of the current query. If None, uses current date.

        Returns
        -------
        dict
            The appropriate filter spec for the date range
        """
        spec_list = self._named_specs.get(name)
        if not spec_list:
            raise KeyError(f"Unknown filter: '{name}'")

        # Handle simple (non-dated) filters
        if isinstance(spec_list, dict):
            return spec_list

        # Handle dated filter versions
        if not isinstance(spec_list, list):
            raise ValueError(f"Filter '{name}' must be dict or list of dicts")

        # Determine effective date for resolution
        if query_date_range:
            # Use the start of the query range
            effective_date = query_date_range[0].date()
        else:
            # Use current date if no query range specified
            effective_date = pd.Timestamp.now().date()

        # Find the best matching version
        for version in spec_list:
            valid_from = pd.to_datetime(version.get("valid_from", "1900-01-01")).date()
            valid_to_str = version.get("valid_to")
            valid_to = (
                pd.to_datetime(valid_to_str).date()
                if valid_to_str
                else pd.Timestamp.now().date()
            )

            if valid_from <= effective_date <= valid_to:
                # Return the spec without the date metadata
                return {
                    k: v
                    for k, v in version.items()
                    if k not in ["valid_from", "valid_to"]
                }

        # No valid version found
        raise ValueError(
            f"No valid version of filter '{name}' for date {effective_date}"
        )

    def _apply_resolved_spec(self, name: str, spec: dict, seen: set[str]) -> None:
        """Apply a resolved filter spec, handling composite filters."""
        if name in seen:
            raise ValueError(f"Cycle detected in composite filters at '{name}'")
        seen.add(name)

        kind = spec.get("kind")
        if kind == "composite":
            for sub in spec.get("subfilters", []) or []:
                if self._date_range:
                    self._apply_dated_filter_spanning(sub, self._date_range)
                else:
                    resolved_sub = self._resolve_dated_filter(sub, None)
                    self._apply_resolved_spec(sub, resolved_sub, seen)
            return

        self._instantiate_and_enqueue(kind, spec.get("args") or {})

    def _create_composite_dated_filter_pydantic(
        self, applicable_versions: List, date_range: Tuple[pd.Timestamp, pd.Timestamp]
    ) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """Create a composite filter that applies the right Pydantic FilterVersion based on scan date."""

        def _composite_dated_filter(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df

            # Create scan dates from year/month/day columns
            if not all(c in df.columns for c in ["year", "month", "day"]):
                # If we don't have date columns, apply all versions (OR logic)
                result_mask = pd.Series([False] * len(df), index=df.index)
                for version in applicable_versions:
                    filtered_subset = self._apply_single_filter_version(df, version)
                    result_mask.loc[filtered_subset.index] = True
                return df[result_mask]

            scan_dates = pd.to_datetime(
                {"year": df["year"], "month": df["month"], "day": df["day"]}
            )

            result_mask = pd.Series([False] * len(df), index=df.index)

            # Apply each version to its applicable date range
            start_date, end_date = date_range[0].date(), date_range[1].date()
            for version in applicable_versions:
                # Calculate overlap period for this version
                valid_from = version.valid_from or pd.Timestamp("1900-01-01").date()
                valid_to = version.valid_to or pd.Timestamp.now().date()

                overlap_start = pd.Timestamp(max(valid_from, start_date))
                overlap_end = pd.Timestamp(min(valid_to, end_date))

                date_mask = (scan_dates >= overlap_start) & (scan_dates <= overlap_end)
                if date_mask.any():
                    # Apply this version's filter to the subset
                    subset_df = df[date_mask]
                    filtered_subset = self._apply_single_filter_version(
                        subset_df, version
                    )
                    # Mark these rows as passing the filter
                    result_mask.loc[filtered_subset.index] = True

            return df[result_mask]

        return _composite_dated_filter

    def _apply_single_filter_version(self, df: pd.DataFrame, version) -> pd.DataFrame:
        """Apply a single Pydantic FilterVersion to a DataFrame and return the filtered result."""
        kind = version.kind
        args = version.args

        if kind == "ecs_value_contains":
            device_like = args.device_like
            variable_like = args.variable_like
            text = args.text
            case = args.case

            if case:

                def _ok_str(v) -> bool:
                    return isinstance(v, str) and (text in v)
            else:
                t = text.lower()

                def _ok_str(v) -> bool:
                    return isinstance(v, str) and (t in v.lower())

            if "ecs_dump" not in df.columns:
                return df
            mask = df["ecs_dump"].apply(
                lambda s: any(
                    _ok_str(v)
                    for v in ScanDatabase._ecs_values(s, device_like, variable_like)
                )
            )
            return df[mask]

        elif kind == "ecs_value_within":
            device_like = args.device_like
            variable_like = args.variable_like
            target = float(args.target)
            tol = float(args.tol)

            def _ok_num(v) -> bool:
                try:
                    return abs(float(v) - target) <= tol
                except Exception:
                    return False

            if "ecs_dump" not in df.columns:
                return df
            mask = df["ecs_dump"].apply(
                lambda s: any(
                    _ok_num(v)
                    for v in ScanDatabase._ecs_values(s, device_like, variable_like)
                )
            )
            return df[mask]

        elif kind == "scan_parameter_contains":
            substring = args.substring
            case = args.case

            if "scan_parameter" not in df.columns:
                return df
            return df[
                df["scan_parameter"]
                .astype("string")
                .str.contains(substring, case=case, na=False)
            ]

        elif kind == "experiment_equals":
            name = args.name
            if "experiment" not in df.columns:
                return df
            return df[df["experiment"] == name]

        elif kind == "device_contains":
            device_substring = args.device_substring
            needle = device_substring.lower()

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

        elif kind == "composite":
            # Handle composite filters recursively
            result_mask = pd.Series([True] * len(df), index=df.index)
            for subfilter_name in args.subfilters:
                subfilter_spec = self._named_specs.get(subfilter_name)
                if subfilter_spec:
                    # For composite filters, we need to apply all subfilters
                    # This is a simplified approach - in practice you might want more sophisticated logic
                    for sub_version in subfilter_spec.versions:
                        sub_result = self._apply_single_filter_version(df, sub_version)
                        result_mask &= df.index.isin(sub_result.index)
            return df[result_mask]

        else:
            # Unknown filter type, return unchanged
            return df

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

    # -----------------------
    # Partition I/O helpers
    # -----------------------
    def _load_partitions(self) -> pd.DataFrame:
        """Load selected partitions via a single hive-partitioned dataset, then apply filters."""
        # Build a single dataset rooted at the hive tree
        dataset = ds.dataset(str(self.root), format="parquet", partitioning="hive")

        # Optional partition pruning using a filter expression
        filter_expr = None
        if self._date_range is not None:
            s, e = self._date_range
            months = pd.period_range(s, e, freq="M")

            # group months by year to build (year==Y & month IN [...]) OR ...
            by_year = {}
            for p in months:
                by_year.setdefault(p.year, []).append(p.month)

            exprs = []
            for y, mlist in by_year.items():
                exprs.append(
                    (ds.field("year") == int(y))
                    & ds.field("month").isin(list(map(int, mlist)))
                )
            # Combine with OR
            if exprs:
                filter_expr = exprs[0]
                for ex in exprs[1:]:
                    filter_expr = filter_expr | ex

        # Scan only the needed partitions; this injects year/month columns automatically
        table = dataset.to_table(filter=filter_expr)
        df = table.to_pandas()  # (avoid Arrow-backed dtypes here)

        # Day-level trim if day exists (partition filter above is month-level)
        if self._date_range is not None and all(
            c in df.columns for c in ("year", "month", "day")
        ):
            s, e = self._date_range
            ts = pd.to_datetime(
                {"year": df["year"], "month": df["month"], "day": df["day"]},
                errors="coerce",
            )
            df = df[(ts >= s) & (ts <= e)].copy()

        # Normalize ECS once so filters are fast and stable
        # if "ecs_dump" in df.columns:
        #     df["__ecs"] = df["ecs_dump"].astype("object").map(self._ecs_json_load)

        if "ecs_dump" in df.columns:
            df["__ecs"] = df["ecs_dump"].astype(str).map(self._ecs_json_load)

        # Apply any queued pandas filters
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
        import numpy as np
        import pandas as pd

        needle = device_substring.lower()

        def _f(df: pd.DataFrame) -> pd.DataFrame:
            col = "non_scalar_devices"
            if col not in df.columns:
                return df

            def _has(v) -> bool:
                # normalize to list[str] just-in-time
                if isinstance(v, np.ndarray):
                    v = v.ravel().tolist()
                elif isinstance(v, list):
                    pass
                elif v is None or (isinstance(v, float) and pd.isna(v)):
                    v = []
                elif isinstance(v, str):
                    v = [v]
                else:
                    v = []

                return any(isinstance(x, str) and needle in x.lower() for x in v)

            return df[df[col].apply(_has)]

        self._df_filters.append(_f)
        return self

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
            col = "__ecs" if "__ecs" in df.columns else "ecs_dump"
            mask = df[col].apply(
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
            col = "__ecs" if "__ecs" in df.columns else "ecs_dump"
            mask = df[col].apply(
                lambda s: any(
                    _ok_str(v)
                    for v in ScanDatabase._ecs_values(s, device_like, variable_like)
                )
            )
            return df[mask]

        self._df_filters.append(_f)
        return self

    def load_named_filters(self, path: Union[str, Path]) -> "ScanDatabase":
        """Load named filter specs (including composites) from a YAML file."""
        p = Path(path)
        data = yaml.safe_load(p.read_text())
        specs = (data or {}).get("filters") or {}
        if not isinstance(specs, Mapping):
            raise ValueError("YAML must contain a top-level 'filters' mapping")

        # Parse and validate using Pydantic models
        validated_specs = {}
        total_filters = len(specs)
        skipped_filters = []

        for name, raw_spec in specs.items():
            try:
                # Parse using Pydantic models with automatic validation
                filter_spec = parse_filter_spec_from_yaml(name, raw_spec)
                validated_specs[name] = filter_spec
            except (ValidationError, ValueError) as e:
                print(f"[ERROR] Filter '{name}' validation failed: {e} - REMOVED")
                skipped_filters.append(name)
            except Exception as e:
                print(f"[ERROR] Filter '{name}' unexpected error: {e} - REMOVED")
                skipped_filters.append(name)

        self._named_specs.update(validated_specs)

        # Report loading summary
        loaded_count = len(validated_specs)
        print(f"[INFO] Loaded {loaded_count}/{total_filters} filters from {p.name}")
        if skipped_filters:
            print(f"[INFO] Skipped filters: {', '.join(skipped_filters)}")

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

    def list_named_filters(self, show_dates: bool = False) -> list[str]:
        """Return available filter names, optionally showing validity periods."""
        if not show_dates:
            return sorted(self._named_specs.keys())

        result = []
        for name, filter_spec in self._named_specs.items():
            if len(filter_spec.versions) > 1:
                # Multiple dated versions
                for version in filter_spec.versions:
                    valid_from = version.valid_from or "always"
                    valid_to = version.valid_to or "current"
                    result.append(f"{name}[{valid_from} to {valid_to}]")
            else:
                # Single version
                version = filter_spec.versions[0]
                if version.valid_from or version.valid_to:
                    valid_from = version.valid_from or "always"
                    valid_to = version.valid_to or "current"
                    result.append(f"{name}[{valid_from} to {valid_to}]")
                else:
                    result.append(f"{name}[undated]")
        return sorted(result)

    def describe_named_filter(self, name: str) -> dict[str, Any]:
        """Return the raw spec dict for a named filter."""
        spec = self._named_specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown named filter: '{name}'")
        return spec

    def apply(self, *names: str) -> "ScanDatabase":
        """Apply one or more named filters, using stored date range for version resolution."""
        if not names:
            return self

        for name in names:
            if self._date_range:
                # Use the multi-version spanning logic
                self._apply_dated_filter_spanning(name, self._date_range)
            else:
                # No date range set, use current date or latest version
                resolved_spec = self._resolve_dated_filter(name, None)
                self._apply_resolved_spec(name, resolved_spec, set())
        return self
