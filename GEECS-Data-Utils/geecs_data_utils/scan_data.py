"""
GEECS scan data loading and management utilities.

This module provides functionality for loading and manipulating GEECS
experimental scan data, including TDMS file reading, scalar data loading,
and data format conversions.

Contains the ScanData class which extends ScanPaths with data loading
capabilities for GEECS experimental scans.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import shutil
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    TypeAlias,
    Hashable,
)
import logging
import re

import numpy as np
import pandas as pd
import nptdms as tdms

from geecs_data_utils.scan_paths import ScanPaths
from geecs_data_utils.type_defs import parse_ecs_dump, ECSDump


# ----------------------------- Types & Config ---------------------------------

AggT = Literal["mean", "median"]
ErrT = Literal["std", "stderr", "iqr", "percentile", "mad"]
DropT = Literal["any", "all"]

ColumnMatchMode: TypeAlias = Literal[
    "contains", "startswith", "endswith", "regex", "exact"
]


@dataclass(frozen=True)
class BinningConfig:
    """Configuration for per-bin aggregation.

    The binner computes, for each selected value column in each bin,
    three subcolumns:
        (value, "center"), (value, "err_low"), (value, "err_high")

    Parameters
    ----------
    bin_col
        Column name to use for bin identity (or source for numeric binning).
    value_cols
        Columns to aggregate; if None, use numeric columns (excluding 'Shotnumber' and bin_col).
    agg
        Center estimator per bin: "mean" or "median".
    err
        Error definition per bin:
          - "std"       : sample standard deviation (symmetric → err_low == err_high)
          - "stderr"    : std / sqrt(N) (symmetric)
          - "mad"       : median absolute deviation (scaled by 1.4826 if scale_to_sigma) (symmetric)
          - "iqr"       : inter-quantile range using `percentiles` (asymmetric:
                          err_low = center - q_low; err_high = q_high - center)
          - "percentile": same as "iqr" but with arbitrary (low, high) `percentiles` (asymmetric)
    ddof
        Degrees of freedom for standard deviation calculations (used by "std"/"stderr").
    percentiles
        (low, high) quantiles for "iqr"/"percentile" methods; defaults to (0.25, 0.75).
    scale_to_sigma
        If True, scale symmetric measures toward σ:
          - "mad": multiply by 1.4826
          - "std"/"stderr": no additional scaling (these are already σ-like);
        Ignored for asymmetric methods ("iqr", "percentile") since those are one-sided offsets.
    min_count
        Minimum samples required for a bin to be reported (after grouping).
    dropna
        Row-wise NA handling before grouping: "any" drops a row if any value_cols NA,
        "all" drops only if all value_cols NA.

    Numeric binning (optional)
    --------------------------
    bin_edges
        Explicit edges for pd.cut.
    bin_width
        Uniform width; builds edges from data range (optionally starting at `origin`).
    quantile_bins
        e.g., 10 → deciles via pd.qcut.
    right
        Include right edge if using cut.
    label
        Label emitted for numeric bins: "interval", "left", "center" (default), or "right".
    origin
        Starting point for width-bins; defaults to data min if None.
    """

    bin_col: str = "Bin #"
    value_cols: Optional[Iterable[str]] = None
    agg: AggT = "median"
    err: ErrT = "iqr"
    ddof: int = 1
    percentiles: Tuple[float, float] = (0.25, 0.75)
    scale_to_sigma: bool = False
    min_count: int = 1
    dropna: DropT = "any"

    # numeric binning options
    bin_edges: Optional[Sequence[float]] = None
    bin_width: Optional[float] = None
    quantile_bins: Optional[int] = None
    right: bool = True
    label: Literal["interval", "left", "center", "right"] = "center"
    origin: Optional[float] = None


def read_geecs_tdms(file_path: Path) -> Optional[dict[str, dict[str, np.ndarray]]]:
    """
    Read a GEECS TDMS file and return nested dictionary structure.

    Parameters
    ----------
    file_path : Path
        Path to the TDMS file to read

    Returns
    -------
    Optional[dict[str, dict[str, np.ndarray]]]
        Nested dictionary with structure device -> variable -> ndarray,
        None if file is not valid TDMS format

    Examples
    --------
    >>> data = read_geecs_tdms(Path("scan001.tdms"))
    >>> if data:
    ...     print(data.keys())  # Device names
    """
    if not file_path.is_file() or file_path.suffix.lower() != ".tdms":
        return None

    with tdms.TdmsFile.open(str(file_path)) as f_tdms:

        def convert(channel: tdms.TdmsChannel):
            arr = channel[:]
            try:
                return arr.astype("float64")
            except ValueError:
                return arr

        return {
            group.name: {
                var.name.split(group.name)[1].lstrip("_"): convert(var)
                for var in group.channels()
            }
            for group in f_tdms.groups()
        }


def geecs_tdms_dict_to_panda(
    data_dict: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """
    Convert nested TDMS dictionary into a multi-indexed pandas DataFrame.

    Parameters
    ----------
    data_dict : dict[str, dict[str, np.ndarray]]
        Nested dictionary from read_geecs_tdms with device -> variable -> data structure

    Returns
    -------
    pd.DataFrame
        Multi-indexed DataFrame with devices as top-level columns,
        indexed by shot number

    Examples
    --------
    >>> data = read_geecs_tdms(Path("scan001.tdms"))
    >>> df = geecs_tdms_dict_to_panda(data)
    >>> print(df.columns.levels[0])  # Device names
    """
    return pd.concat(
        map(pd.DataFrame, data_dict.values()), keys=data_dict.keys(), axis=1
    ).set_index("Shotnumber")


# ------------------------------- Core Class -----------------------------------


class ScanData:
    """Container for a single scan: paths + scalar DataFrame + lazy asset index.

    This class composes a :class:`ScanPaths` (path logic) and provides:
    - Optional scalar DataFrame loading (s-file or TDMS→DataFrame).
    - Lazy, normalized asset indexing (no bytes loaded).
    - Convenience helpers for grouping/averaging images by ``Bin #``.
    - Flexible column resolution (case-insensitive, substring/regex).
    - Per-bin scalar aggregation with configurable center and error.

    Parameters
    ----------
    paths
        A pre-constructed :class:`ScanPaths` instance pointing to the scan.

    Notes
    -----
    Use the factories :meth:`from_date` and :meth:`latest` for ergonomic creation.
    """

    # --------------------------- Construction ---------------------------------

    def __init__(self, *, paths: ScanPaths):
        self.paths: ScanPaths = paths
        self.data_frame: Optional[pd.DataFrame] = None

        # Binning state
        self._bin_cfg: BinningConfig = BinningConfig()
        self._binned_cache: Optional[pd.DataFrame] = None
        self._df_version: int = 0
        self._binned_key: Optional[Tuple] = None

        # Local (user) aliases for columns (independent of DAQ "Alias:" strings)
        self.column_aliases: Dict[str, str] = {}

    # Factory helpers -----------------------------------------------------------

    @classmethod
    def from_date(
        cls,
        *,
        year: int,
        month: int,
        day: int,
        number: int,
        experiment: str,
        base_directory: Optional[Path] = None,
        load_scalars: bool = True,
        source: Literal["sfile", "tdms"] = "sfile",
        append_paths: bool = True,
    ) -> "ScanData":
        """Construct a :class:`ScanData` from date/number.

        Parameters
        ----------
        year, month, day, number, experiment
            Identify the scan.
        base_directory
            Base data root if not configured globally.
        load_scalars
            If True, load scalar DataFrame immediately.
        source
            ``"sfile"`` (default) or ``"tdms"`` for scalar source.
        append_paths
            If true, ad device/shot paths to df.

        Returns
        -------
        ScanData
        """
        tag = ScanPaths.get_scan_tag(year, month, day, number, experiment=experiment)
        paths = ScanPaths(tag=tag, base_directory=base_directory)
        sd = cls(paths=paths)
        if load_scalars:
            sd.load_scalars(source=source, append_paths=append_paths)
        return sd

    @classmethod
    def latest(
        cls,
        experiment: str,
        *,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        base_directory: Optional[Path] = None,
        load_scalars: bool = True,
        source: Literal["sfile", "tdms"] = "sfile",
    ) -> "ScanData":
        """Construct a :class:`ScanData` for the latest scan on a date.

        Parameters
        ----------
        experiment
            Experiment name.
        year, month, day
            Optional date components; defaults to today if omitted.
        base_directory
            Base data root if not configured globally.
        load_scalars
            If True, load scalar DataFrame immediately.
        source
            ``"sfile"`` (default) or ``"tdms"``.

        Returns
        -------
        ScanData
        """
        tag = ScanPaths.get_latest_scan_tag(
            experiment=experiment,
            year=year,
            month=month,
            day=day,
            base_directory=base_directory,
        )
        if not tag:
            raise ValueError("No scans found for the specified date/experiment.")
        paths = ScanPaths(tag=tag, base_directory=base_directory)
        sd = cls(paths=paths)
        if load_scalars:
            sd.load_scalars(source=source)
        return sd

    # ------------------------------ Scalars I/O --------------------------------

    def load_scalars(
        self, *, source: Literal["sfile", "tdms"] = "sfile", append_paths: bool = True
    ) -> None:
        """Load the scalar DataFrame (s-file or TDMS converted).

        Parameters
        ----------
        source
            ``"sfile"`` to read ``s{scan}.txt`` from the analysis tree, or ``"tdms"`` to
            read ``ScanNNN.tdms`` and convert to a DataFrame if possible.
        append_paths
            If true, add device/shot paths to dataframe.

        Raises
        ------
        FileNotFoundError
            If the s-file is expected but missing.
        """
        if source == "sfile":
            tag = self.paths.get_tag()
            sfile = self.paths.get_analysis_folder().parent / f"s{tag.number}.txt"
            if not sfile.exists():
                raise FileNotFoundError(f"No sfile for scan {tag}")
            df = pd.read_csv(sfile, delimiter="\t")
            self.set_data_frame(df, append_paths=append_paths)

        elif source == "tdms":
            tag = self.paths.get_tag()
            tdms_path = self.paths.get_folder() / f"Scan{tag.number:03d}.tdms"
            if not tdms_path.exists():
                raise FileNotFoundError(f"TDMS file not found: {tdms_path}")
            dct = read_geecs_tdms(tdms_path) or {}
            if not dct:
                raise ValueError(f"TDMS file could not be parsed: {tdms_path}")
            df = geecs_tdms_dict_to_panda(dct)
            self.set_data_frame(df, append_paths=append_paths)

        else:
            raise ValueError(f"Unsupported source: {source!r}")

    def set_data_frame(self, df: pd.DataFrame, *, append_paths: bool = True) -> None:
        """Attach a scalar DataFrame and invalidate dependent caches.

        Parameters
        ----------
        df
            Scalar table for the scan (typically from s-file).
        append_paths
            If true, add device shot paths to dataframe.
        """
        if append_paths:
            df = self._append_expected_asset_columns(df)
        self.data_frame = df
        self._df_version += 1
        self._binned_cache = None
        self._binned_key = None

    # ------------------------- Flexible Column Resolution ----------------------

    def list_columns(self) -> List[str]:
        """List column names as strings (flattens MultiIndex columns if present).

        Returns
        -------
        list of str
        """
        return self._flatten_columns()

    def find_cols(
        self,
        query: Union[str, Sequence[str]],
        *,
        mode: ColumnMatchMode = "contains",
        case_sensitive: bool = False,
    ) -> List[str]:
        """Flexible column search.

        Parameters
        ----------
        query
            String or list of strings to search for.
        mode
            Search mode: ``"contains"`` (default), ``"startswith"``, ``"endswith"``,
            ``"regex"``, or ``"exact"``.
        case_sensitive
            If True, match with case sensitivity.

        Returns
        -------
        list of str
            Matching column names (flattened form). May be empty.
        """
        if self.data_frame is None:
            return []

        cols = self._flatten_columns()
        originals = cols
        hay = originals if case_sensitive else [c.lower() for c in originals]
        queries = [query] if isinstance(query, str) else list(query)

        matches: set[str] = set()
        for q in queries:
            needle = q if case_sensitive else str(q).lower()
            if mode == "regex":
                flags = 0 if case_sensitive else re.IGNORECASE
                pat = re.compile(str(q), flags=flags)
                for orig in originals:
                    if pat.search(orig):
                        matches.add(orig)
                continue

            for orig, h in zip(originals, hay):
                s = orig if case_sensitive else h
                if (
                    (mode == "contains" and needle in s)
                    or (mode == "startswith" and s.startswith(needle))
                    or (mode == "endswith" and s.endswith(needle))
                    or (mode == "exact" and s == needle)
                ):
                    matches.add(orig)

        return sorted(matches)

    def resolve_col(
        self,
        spec: str,
        *,
        mode: ColumnMatchMode = "contains",
        case_sensitive: bool = False,
        prefer_exact_ci: bool = True,
    ) -> str:
        """Resolve a loose column spec to a single best column name.

        Parameters
        ----------
        spec
            User-provided spec (may be an alias or partial/regex).
        mode
            Matching strategy used by :meth:`find_cols`: ``"contains"`` (default),
            ``"startswith"``, ``"endswith"``, ``"regex"``, or ``"exact"``.
        case_sensitive
            If True, enforce case-sensitive matching for the chosen mode.
        prefer_exact_ci
            Prefer exact (case-insensitive) matches over substring/regex matches.

        Returns
        -------
        str
            Selected column name.

        Raises
        ------
        ValueError
            If no match is found.
        """
        if self.data_frame is None:
            raise ValueError("No scalar dataframe loaded.")

        # 0) local alias wins immediately
        if spec in self.column_aliases:
            return self.column_aliases[spec]

        cols = self._flatten_columns()

        # 1) exact (case-insensitive) preferred
        if prefer_exact_ci:
            eq = [c for c in cols if c.lower() == spec.lower()]
            if len(eq) == 1:
                return eq[0]
            # fall through if 0 or >1

        # 2) search using requested mode
        hits = self.find_cols(spec, mode=mode, case_sensitive=case_sensitive)

        # 3) last-resort: also try 'contains' if the requested mode found nothing
        if not hits and mode != "contains":
            hits = self.find_cols(spec, mode="contains", case_sensitive=case_sensitive)

        if not hits:
            raise ValueError(
                f"No columns match spec {spec!r}. "
                f"Available (showing up to 6): {cols[:6]}..."
            )

        # 4) deterministic tie-break: prefer exact-ci, else shortest then lexicographic
        exact_ci = [h for h in hits if h.lower() == spec.lower()]
        if len(exact_ci) == 1:
            return exact_ci[0]
        if len(hits) > 1:
            winner = sorted(hits, key=lambda s: (len(s), s))[0]
            logging.warning(
                "Spec %r matched multiple columns (%d): %s; using %r",
                spec,
                len(hits),
                hits,
                winner,
            )
            return winner
        return hits[0]

    def add_local_alias(self, alias: str, actual_col: str) -> None:
        """Register a user-defined shorthand for a column name.

        Parameters
        ----------
        alias
            Local shorthand (e.g., ``"pressure"``).
        actual_col
            Full column name present in the DataFrame.
        """
        self.column_aliases[alias] = actual_col

    # ----------------------------- Binned Scalars ------------------------------

    def set_binning_config(self, **updates) -> None:
        """Update binning configuration and invalidate cache.

        Parameters
        ----------
        **updates
            Fields to replace on the current :class:`BinningConfig`.
        """
        if "value_cols" in updates and updates["value_cols"] is not None:
            updates["value_cols"] = tuple(map(str, updates["value_cols"]))
        self._bin_cfg = replace(self._bin_cfg, **updates)
        self._binned_cache = None
        self._binned_key = None

    @property
    def binned_scalars(self) -> pd.DataFrame:
        """
        Aggregate scalar data into bins with configurable center and error metrics.

        For each bin defined by ``bin_col`` in the current :class:`BinningConfig`,
        all selected numeric columns (``value_cols``) are aggregated. The result is a
        wide DataFrame with a two-level column index:
        ``(column_name, {"center", "err_low", "err_high"})``.

        Notes
        -----
        - If ``value_cols`` is None, all numeric columns in the scalar DataFrame
          are included (including the bin source column and Shotnumber).
        - The bin column is treated like any other numeric column: its per-bin
          center and errors are computed the same way as other variables.
        - Error definitions (``err``) control how ``err_low`` and ``err_high`` are
          computed:
            * ``"std"``     : sample standard deviation (symmetric).
            * ``"stderr"``  : standard error of the mean (symmetric).
            * ``"mad"``     : median absolute deviation (scaled if
                              ``scale_to_sigma=True``; symmetric).
            * ``"iqr"``     : interquartile range using ``percentiles``;
                              asymmetric offsets around the chosen center.
            * ``"percentile"``: arbitrary quantile range using ``percentiles``;
                                asymmetric offsets around the chosen center.
        - Counts per bin are included under the pseudo-column
          ``("count", "center")``.

        Returns
        -------
        pandas.DataFrame
            Binned scalar table with a MultiIndex on columns:

            * Level 0: original column names plus ``"count"``.
            * Level 1: one of ``{"center", "err_low", "err_high"}``.

            The row index corresponds to unique bin labels, which may be
            discrete values or numeric bin centers depending on the binning
            configuration.

        Raises
        ------
        ValueError
            If no scalar DataFrame is loaded.
        KeyError
            If the configured bin column is not found.
        """
        if self.data_frame is None:
            raise ValueError("No scalar dataframe loaded.")

        self._require_bin_col()
        cfg = self._bin_cfg
        key = (id(self.data_frame), self._df_version, self._bin_cfg_fingerprint())
        if self._binned_cache is not None and self._binned_key == key:
            return self._binned_cache.copy()

        df = self.data_frame.copy()

        # Drop NA rows according to policy (only for selected value cols)
        num_cols = df.select_dtypes(include=[np.number]).columns

        value_cols = (
            num_cols.tolist() if cfg.value_cols is None else list(cfg.value_cols)
        )

        if cfg.dropna == "any":
            df = df.dropna(subset=value_cols, how="any")
        else:
            df = df.dropna(subset=value_cols, how="all")

        # NEW: compute bin key (supports numeric binning) and group on it
        bin_key, bin_name = self._compute_bin_key(df)
        df = df.assign(**{bin_name: bin_key})
        g = df.groupby(bin_name, dropna=False, observed=True)

        # centers
        if cfg.agg == "median":
            center = g[value_cols].median()
        else:
            center = g[value_cols].mean()

        # --- errors: only err_low / err_high ---
        err_low = pd.DataFrame(index=center.index, columns=value_cols, dtype=float)
        err_high = pd.DataFrame(index=center.index, columns=value_cols, dtype=float)

        if cfg.err in ("std", "stderr"):
            std = g[value_cols].std(ddof=cfg.ddof)
            if cfg.err == "stderr":
                counts = g.size().reindex(std.index).astype(float)
                std = std.div(np.sqrt(counts), axis=0)

            # symmetric → fill both with same value
            err_low.loc[:, :] = std.values
            err_high.loc[:, :] = std.values

        elif cfg.err in ("iqr", "percentile"):
            p_lo, p_hi = cfg.percentiles
            qs = np.array([float(p_lo), float(p_hi)], dtype=float)
            qtbl = g[value_cols].quantile(q=qs)  # index: (bin, quantile)

            # wide form
            if isinstance(qtbl.index, pd.MultiIndex) and "quantile" in (
                qtbl.index.names or []
            ):
                qwide = qtbl.unstack("quantile")
            else:
                qwide = qtbl.unstack(level=-1)

            lo = qwide.xs(p_lo, axis=1, level=-1)
            hi = qwide.xs(p_hi, axis=1, level=-1)

            # asymmetric offsets relative to chosen center
            err_low = (center - lo).clip(lower=0)
            err_high = (hi - center).clip(lower=0)

        elif cfg.err == "mad":
            med = g[value_cols].median()
            mad = g[value_cols].apply(
                lambda sub: (sub - med.loc[sub.name]).abs().median()
            )
            mad = mad * (1.4826 if cfg.scale_to_sigma else 1.0)
            err_low.loc[:, :] = mad.values
            err_high.loc[:, :] = mad.values

        else:
            raise ValueError(f"Unknown err: {cfg.err}")

        # ---- build output with center, err_low, err_high ----
        pieces = {}
        for col in value_cols:
            pieces[col] = pd.concat(
                {
                    "center": center[col],
                    "err_low": err_low[col],
                    "err_high": err_high[col],
                },
                axis=1,
            )
        out = pd.concat(pieces, axis=1)

        # counts + min_count filter
        counts = g.size()
        out[("count", "center")] = counts
        if cfg.min_count > 1:
            keep = counts[counts >= cfg.min_count].index
            out = out.loc[keep]

        out = out.sort_index(axis=1)
        self._binned_cache = out
        self._binned_key = key
        return out.copy()

    def expected_paths_by_bin(
        self,
        device: str,
        *,
        variant: Optional[str] = None,
        bin_col: Optional[str] = None,
        dropna_paths: bool = True,
        exists_only: bool = False,
    ) -> Dict[Hashable, List[Path]]:
        """
        Group expected image paths by the current bin definition.

        Parameters
        ----------
        device
            Device name (subfolder).
        variant
            Optional variant suffix used when creating expected-path columns.
        bin_col
            Override the configured bin column for this call.
        dropna_paths
            If True, drop rows with missing path strings.
        exists_only
            If True, filter out paths that do not currently exist on disk.

        Returns
        -------
        dict[Hashable, list[pathlib.Path]]
            Mapping {bin_value -> [image paths]}.
        """
        if self.data_frame is None:
            raise ValueError("No scalar dataframe loaded.")

        # Optionally override the bin column for just this call
        if bin_col is not None:
            self._bin_cfg = replace(self._bin_cfg, bin_col=str(bin_col))

        # Ensure the bin source is present; compute the effective bin key
        self._require_bin_col()
        df = self.data_frame.copy()
        bin_key, bin_name = self._compute_bin_key(df)
        df = df.assign(**{bin_name: bin_key})

        col = self._expected_path_col(device, variant=variant)
        series = df[col]

        if dropna_paths:
            mask = series.notna()
            df = df.loc[mask]

        # Convert to Paths and optionally filter to existing files
        df = df.assign(
            _path_obj=df[col].map(lambda s: Path(s) if isinstance(s, str) else None)
        )
        if exists_only:
            df = df.loc[df["_path_obj"].map(lambda p: p is not None and p.exists())]

        out: Dict[Hashable, List[Path]] = {}
        for bval, group in df.groupby(bin_name, dropna=False, observed=True, sort=True):
            paths = [p for p in group["_path_obj"].tolist() if p is not None]
            if paths:
                out[bval] = paths
        return out

    # ------------------------------- Internals ---------------------------------

    def _flatten_columns(self) -> List[str]:
        """Flatten DataFrame columns to strings, joining MultiIndex with ':'.

        Returns
        -------
        list of str
        """
        if self.data_frame is None:
            return []
        cols = self.data_frame.columns
        if getattr(cols, "nlevels", 1) > 1:
            return [":".join(map(str, tup)) for tup in cols.to_list()]
        return list(map(str, cols))

    def _bin_cfg_fingerprint(self) -> Tuple:
        cfg = self._bin_cfg
        vc = None
        if cfg.value_cols is not None:
            vc = tuple(map(str, cfg.value_cols))
        # NEW parts for numeric binning
        edges = None if cfg.bin_edges is None else tuple(map(float, cfg.bin_edges))
        return (
            cfg.bin_col,
            vc,
            cfg.agg,
            cfg.err,
            cfg.ddof,
            float(cfg.percentiles[0]),
            float(cfg.percentiles[1]),
            cfg.scale_to_sigma,
            cfg.min_count,
            cfg.dropna,
            edges,  # new
            (None if cfg.bin_width is None else float(cfg.bin_width)),  # new
            (None if cfg.quantile_bins is None else int(cfg.quantile_bins)),  # new
            cfg.right,
            cfg.label,  # new
            (None if cfg.origin is None else float(cfg.origin)),  # new
        )

    def _require_bin_col(self) -> None:
        if self.data_frame is None:
            raise ValueError("No scalar dataframe loaded.")
        if self._bin_cfg.bin_col not in self.data_frame.columns:
            raise KeyError(
                f"Bin column {self._bin_cfg.bin_col!r} not found in DataFrame."
            )

    def _compute_bin_key(self, df: pd.DataFrame) -> Tuple[pd.Series, str]:
        """Return (bin_key_series, bin_name_used). Uses cfg.* to optionally build numeric bins."""
        cfg = self._bin_cfg
        src = cfg.bin_col
        if src not in df.columns:
            raise KeyError(f"Bin column {src!r} not found in DataFrame.")

        s = df[src]

        # only do numeric binning if the source is numeric
        if not pd.api.types.is_numeric_dtype(s):
            # non-numeric: fall back to identity bins
            return s, src

        # 1) explicit edges
        if cfg.bin_edges is not None:
            bins = pd.cut(s, bins=list(cfg.bin_edges), right=cfg.right)
        # 2) uniform width
        elif cfg.bin_width is not None:
            vmin = s.min() if cfg.origin is None else cfg.origin
            vmax = s.max()
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            # make sure we cover the max value
            n = int(np.ceil((vmax - vmin) / float(cfg.bin_width))) or 1
            edges = vmin + np.arange(n + 1, dtype=float) * float(cfg.bin_width)
            bins = pd.cut(s, bins=edges, right=cfg.right, include_lowest=True)
        # 3) quantile bins
        elif cfg.quantile_bins is not None:
            q = int(cfg.quantile_bins)
            q = max(1, q)
            bins = pd.qcut(s, q=q, duplicates="drop")
        else:
            # identity behavior (original)
            return s, src

        # labeling
        if cfg.label == "interval":
            labels = bins.astype(str)
        else:
            left = bins.cat.categories.left.values
            right = bins.cat.categories.right.values
            if cfg.label == "left":
                label_vals = left
            elif cfg.label == "right":
                label_vals = right
            else:  # "center"
                label_vals = (left + right) / 2.0
            mapper = dict(zip(bins.cat.categories, label_vals))
            labels = bins.map(mapper)

        bin_name = f"{src} (binned)"
        return labels, bin_name

    def _append_expected_asset_columns(
        self,
        df: pd.DataFrame,
        *,
        ext_override: Optional[dict[str, str]] = None,
        variants_override: Optional[dict[str, list[Optional[str]]]] = None,
    ) -> pd.DataFrame:
        """
        Add wide columns of expected paths for each device (and optional variant).

        Column names created:
          - ``<device>_expected_path``                  (no variant)
          - ``<device>_expected_<variant>_path``        (with variant)

        Each column contains the full expected file path (as a string) for every
        row's ``Shotnumber``. File extensions are inferred per device via
        :meth:`ScanPaths.infer_device_ext`, unless overridden with ``ext_override``.
        Variants default to ``[None]`` per device, and can be customized with
        ``variants_override``.

        Parameters
        ----------
        df
            Scalar DataFrame that must include ``"Shotnumber"``.
        ext_override
            Optional mapping ``{device: ext}`` to force a specific extension
            (e.g., ``{"UC_HiResMagCam": "png"}``).
        variants_override
            Optional mapping ``{device: [variant1, variant2, None, ...]}`` to
            control which variant-specific columns are created.

        Returns
        -------
        pandas.DataFrame
            A copy of ``df`` with one or more ``*_expected_*_path`` columns added.

        Notes
        -----
        - If ``"Shotnumber"`` is missing, the input ``df`` is returned unchanged.
        - Paths are generated with :meth:`ScanPaths.build_asset_path`.
        """
        if "Shotnumber" not in df.columns:
            return df

        shots = df["Shotnumber"].astype(int).tolist()
        devs = self.paths.list_device_folders()

        # Resolve per-device ext and variants
        ext_map: dict[str, str] = {}
        var_map: dict[str, list[Optional[str]]] = {}
        for dev in devs:
            ext_map[dev] = (ext_override or {}).get(dev) or self.paths.infer_device_ext(
                dev
            )
            var_map[dev] = (variants_override or {}).get(dev, [None])

        # Build and attach columns
        out = df.copy()
        for dev in devs:
            ext = ext_map[dev]
            for variant in var_map[dev]:
                col = (
                    f"{dev}_expected_path"
                    if not variant
                    else f"{dev}_expected_{variant}_path"
                )
                # Faster than row-wise apply: precompute for all shots
                paths = [
                    str(
                        self.paths.build_asset_path(
                            shot=s, device=dev, ext=ext, variant=variant
                        )
                    )
                    for s in shots
                ]
                out[col] = paths

        return out

    def _expected_path_col(self, device: str, variant: Optional[str] = None) -> str:
        """
        Return the expected-path column name for a device/variant.

        Looks for either:
          - "<device>_expected_path"                  (no variant)
          - "<device>_expected_<variant>_path"        (with variant)

        Raises
        ------
        KeyError
            If no matching expected-path column is found.
        """
        if self.data_frame is None:
            raise ValueError("No scalar dataframe loaded.")

        want = (
            f"{device}_expected_path"
            if not variant
            else f"{device}_expected_{variant}_path"
        )
        if want in self.data_frame.columns:
            return want

        # Fallback: search permissively (variant might have underscores, etc.)
        cols = [
            c
            for c in self.data_frame.columns
            if c.startswith(f"{device}_expected_") and c.endswith("_path")
        ]
        if not variant and f"{device}_expected_path" in cols:
            return f"{device}_expected_path"
        if variant:
            # try exact, then CI/underscore-insensitive
            for c in cols:
                if c == want:
                    return c

            def norm(s: str) -> str:
                return s.lower().replace("-", "_")

            for c in cols:
                if norm(c) == norm(want):
                    return c

        raise KeyError(
            f"No expected-path column found for device={device!r}, variant={variant!r}."
        )

    # ------------------------------extras---------------------
    def reload_sfile(self) -> None:
        """Re-read the analysis s-file into ``self.data_frame``.

        Notes
        -----
        This is a thin alias for ``load_scalars(source='sfile')`` to make intent explicit.
        """
        self.load_scalars(source="sfile")

    def copy_fresh_sfile_to_analysis(self) -> None:
        """Replace the analysis s-file with the fresh copy from the scan folder.

        Copies:
            ``<scan>/scans/ScanDataScanNNN.txt`` → ``<scan>/analysis/../sNNN.txt``

        Raises
        ------
        FileNotFoundError
            If the source s-file in ``scans/`` does not exist.
        """
        tag = self.paths.get_tag()
        scan_txt = self.paths.get_folder() / f"ScanDataScan{tag.number:03d}.txt"
        analysis_txt = self.paths.get_analysis_folder().parent / f"s{tag.number}.txt"

        if not scan_txt.exists():
            raise FileNotFoundError(f"Original s-file '{scan_txt}' not found.")
        if analysis_txt.exists():
            analysis_txt.unlink()

        shutil.copy2(src=scan_txt, dst=analysis_txt)

    def load_ecs_live_dump(self) -> ECSDump:
        """Load and parse the ECS Live Dump file for this scan via ``ScanPaths``.

        Returns
        -------
        ECSDump
            Parsed ECS dump structured by device name.

        Raises
        ------
        FileNotFoundError
            If no ECS dump file is available for this scan.
        """
        tag = self.paths.get_tag()
        ecs_path = self.paths.get_ecs_dump_file()
        if not ecs_path:
            raise FileNotFoundError(f"No ECS live dump file found for scan {tag}")
        return parse_ecs_dump(ecs_path)
