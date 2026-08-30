"""
GEECS scan data loading and management utilities.

This module provides functionality for loading and manipulating GEECS
experimental scan data, including TDMS file reading, scalar data loading,
and data format conversions.

Contains the ScanData class which extends ScanPaths with data loading
capabilities for GEECS experimental scans.
"""

from __future__ import annotations

from dataclasses import replace
import shutil
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    Hashable,
)
import logging

import numpy as np
import pandas as pd
import nptdms as tdms

from geecs_data_utils.scan_paths import ScanPaths
from geecs_data_utils.type_defs import parse_ecs_dump, ECSDump
from geecs_data_utils.data.columns import (
    ColumnMatchMode,
    find_cols,
    resolve_col_detailed,
)


# ----------------------------- Types & Config ---------------------------------

# BinningConfig and its aliases moved to the pure binning module
# (analysis-tabs W1c); re-exported here so existing imports
# (`from geecs_data_utils.scan_data import BinningConfig`) keep working.
from geecs_data_utils.data.binning import (  # noqa: E402
    BinningConfig,
    bin_frame,
    compute_bin_key,
)


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
    """
    Container for a single scan: paths + scalar DataFrame + lazy asset index.

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

        # Binning state (before data_frame — its setter touches these)
        self._bin_cfg: BinningConfig = BinningConfig()
        self._binned_cache: Optional[pd.DataFrame] = None
        self._df_version: int = 0
        self._binned_key: Optional[Tuple] = None

        self.data_frame = None

        # Local (user) aliases for columns (independent of DAQ "Alias:" strings)
        self.column_aliases: Dict[str, str] = {}

    @property
    def data_frame(self) -> Optional[pd.DataFrame]:
        """The scalar table (``None`` until loaded).

        Assigning to this attribute invalidates the binned-scalars cache —
        direct reassignment (``sd.data_frame = df``) is a supported pattern
        and must never serve stale binned results.
        """
        return self._data_frame

    @data_frame.setter
    def data_frame(self, df: Optional[pd.DataFrame]) -> None:
        self._data_frame = df
        # getattr: assignment on a bare instance (ScanData.__new__) is a
        # live pattern in the test suite.
        self._df_version = getattr(self, "_df_version", 0) + 1
        self._binned_cache = None
        self._binned_key = None

    # Factory helpers -----------------------------------------------------------

    @classmethod
    def from_date(
        cls,
        *,
        year: int,
        month: int,
        day: int,
        number: int,
        experiment: Optional[str] = None,
        base_directory: Optional[Path] = None,
        load_scalars: bool = True,
        source: Literal["sfile", "tdms"] = "sfile",
        append_paths: bool = True,
        stem_override: Optional[dict[str, str]] = None,
    ) -> "ScanData":
        """
        Construct a :class:`ScanData` from date/number.

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
        stem_override
            Optional ``{device: in_filename_stem}`` mapping forwarded to
            :meth:`load_scalars`. Use when a device's folder name differs
            from the in-filename token (e.g., folder
            ``U_BCaveMagSpec-interpSpec`` with files named
            ``Scan042_U_BCaveMagSpec_001.csv``).

        Returns
        -------
        ScanData
        """
        tag = ScanPaths.get_scan_tag(year, month, day, number, experiment=experiment)
        paths = ScanPaths(tag=tag, base_directory=base_directory)
        sd = cls(paths=paths)
        if load_scalars:
            sd.load_scalars(
                source=source,
                append_paths=append_paths,
                stem_override=stem_override,
            )
        return sd

    @classmethod
    def latest(
        cls,
        experiment: Optional[str] = None,
        *,
        year: Optional[int] = None,
        month: Optional[int] = None,
        day: Optional[int] = None,
        base_directory: Optional[Path] = None,
        load_scalars: bool = True,
        source: Literal["sfile", "tdms"] = "sfile",
    ) -> "ScanData":
        """
        Construct a :class:`ScanData` for the latest scan on a date.

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
        self,
        *,
        source: Literal["sfile", "tdms"] = "sfile",
        append_paths: bool = True,
        stem_override: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Load the scalar DataFrame (s-file or TDMS converted).

        Parameters
        ----------
        source
            ``"sfile"`` to read ``s{scan}.txt`` from the analysis tree, or ``"tdms"`` to
            read ``ScanNNN.tdms`` and convert to a DataFrame if possible.
        append_paths
            If true, add device/shot paths to dataframe.
        stem_override
            Optional ``{device: in_filename_stem}`` mapping forwarded to
            :meth:`set_data_frame`. Use when a device's folder name differs
            from the in-filename token (e.g., folder
            ``U_BCaveMagSpec-interpSpec`` with files named
            ``Scan042_U_BCaveMagSpec_001.csv``).

        Raises
        ------
        FileNotFoundError
            If the s-file is expected but missing.
        """
        if source == "sfile":
            from geecs_data_utils.data.sfile import read_sfile, sfile_path_for_scan

            tag = self.paths.get_tag()
            sfile = sfile_path_for_scan(self.paths.get_folder())
            if not sfile.exists():
                raise FileNotFoundError(f"No sfile for scan {tag}")
            df = read_sfile(sfile)
            self.set_data_frame(
                df, append_paths=append_paths, stem_override=stem_override
            )

        elif source == "tdms":
            tag = self.paths.get_tag()
            tdms_path = self.paths.get_folder() / f"Scan{tag.number:03d}.tdms"
            if not tdms_path.exists():
                raise FileNotFoundError(f"TDMS file not found: {tdms_path}")
            dct = read_geecs_tdms(tdms_path) or {}
            if not dct:
                raise ValueError(f"TDMS file could not be parsed: {tdms_path}")
            df = geecs_tdms_dict_to_panda(dct)
            self.set_data_frame(
                df, append_paths=append_paths, stem_override=stem_override
            )

        else:
            raise ValueError(f"Unsupported source: {source!r}")

    def set_data_frame(
        self,
        df: pd.DataFrame,
        *,
        append_paths: bool = True,
        stem_override: Optional[dict[str, str]] = None,
    ) -> None:
        """Attach a scalar DataFrame and invalidate dependent caches.

        Parameters
        ----------
        df
            Scalar table for the scan (typically from s-file).
        append_paths
            If true, add device shot paths to dataframe.
        stem_override
            Optional ``{device: in_filename_stem}`` mapping forwarded to
            :meth:`_append_expected_asset_columns`. Use when a device's folder
            name differs from the in-filename token (e.g., folder
            ``U_BCaveMagSpec-interpSpec`` with files named
            ``Scan042_U_BCaveMagSpec_001.csv``).
        """
        if append_paths:
            df = self._append_expected_asset_columns(df, stem_override=stem_override)
        self.data_frame = df  # the property setter invalidates the bin cache

    # ------------------------- Flexible Column Resolution ----------------------

    def list_columns(self) -> List[str]:
        """
        List column names as strings (flattens MultiIndex columns if present).

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
        """
        Flexible column search.

        Wrapper for find_cols in geecs_data_utils/data/columns.py.

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
        return find_cols(
            self.data_frame, query, mode=mode, case_sensitive=case_sensitive
        )

    def resolve_col(
        self,
        spec: str,
        *,
        mode: ColumnMatchMode = "contains",
        case_sensitive: bool = False,
        prefer_exact_ci: bool = True,
    ) -> str:
        """
        Resolve a loose column spec to a single best column name.

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

        if spec in self.column_aliases:
            return self.column_aliases[spec]

        result = resolve_col_detailed(
            self.data_frame,
            spec,
            mode=mode,
            case_sensitive=case_sensitive,
            prefer_exact_ci=prefer_exact_ci,
        )
        if result.ambiguous and result.candidates is not None:
            c = result.candidates
            logging.warning(
                "Spec %r matched multiple columns (%d): %s; using %r",
                spec,
                len(c),
                list(c),
                result.column,
            )
        return result.column

    def add_local_alias(self, alias: str, actual_col: str) -> None:
        """
        Register a user-defined shorthand for a column name.

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
        """
        Update binning configuration and invalidate cache.

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
        """Aggregate scalar data into bins (compatibility wrapper).

        Delegates to the pure :func:`geecs_data_utils.data.binning.bin_frame`
        with the instance's current :class:`BinningConfig`, then restores
        the legacy output shape: the per-bin counts are re-attached as the
        pseudo-column ``("count", "center")`` (new code should prefer
        :meth:`bin` / :func:`bin_frame`, whose :class:`BinnedFrame` carries
        counts as a separate series).

        Returns
        -------
        pandas.DataFrame
            Binned scalar table with MultiIndex columns
            ``(column, {"center", "err_low", "err_high"})`` plus
            ``("count", "center")``; one row per surviving bin.

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
        key = (self._df_version, self._bin_cfg_fingerprint())
        if self._binned_cache is not None and self._binned_key == key:
            return self._binned_cache.copy()
        result = bin_frame(self.data_frame, self._bin_cfg)
        out = result.frame.copy()
        out[("count", "center")] = result.counts
        out = out.sort_index(axis=1)
        self._binned_cache = out
        self._binned_key = key
        return out.copy()

    def bin(self, config: BinningConfig) -> pd.DataFrame:
        """Bin the scalar frame with *config* (the documented one-call API).

        Sets the instance's binning configuration and returns
        :attr:`binned_scalars` — the call the package docs always
        advertised (`sd.bin(config)`), now real.

        Parameters
        ----------
        config : BinningConfig
            The aggregation configuration to apply.

        Returns
        -------
        pandas.DataFrame
            See :attr:`binned_scalars`.
        """
        self._bin_cfg = config
        self._binned_cache = None
        return self.binned_scalars

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
        bin_key, bin_name = compute_bin_key(df, self._bin_cfg)
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
        """
        Flatten DataFrame columns to strings, joining MultiIndex with ':'.

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

    def _append_expected_asset_columns(
        self,
        df: pd.DataFrame,
        *,
        ext_override: Optional[dict[str, str]] = None,
        variants_override: Optional[dict[str, list[Optional[str]]]] = None,
        stem_override: Optional[dict[str, str]] = None,
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
        stem_override
            Optional mapping ``{device: in_filename_stem}`` for devices whose
            data folder name differs from the token used in the per-shot
            filename — e.g., ``{"U_BCaveMagSpec-interpSpec": "U_BCaveMagSpec"}``
            for files named ``Scan042_U_BCaveMagSpec_001.csv`` inside the
            ``U_BCaveMagSpec-interpSpec`` folder. Devices not in the mapping
            keep the default ``stem == device`` behavior.

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
        stem_map: dict[str, Optional[str]] = {}
        for dev in devs:
            ext_map[dev] = (ext_override or {}).get(dev) or self.paths.infer_device_ext(
                dev
            )
            var_map[dev] = (variants_override or {}).get(dev, [None])
            stem_map[dev] = (stem_override or {}).get(dev)

        # Build and attach columns
        out = df.copy()
        for dev in devs:
            ext = ext_map[dev]
            stem = stem_map[dev]
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
                            shot=s,
                            device=dev,
                            ext=ext,
                            variant=variant,
                            device_file_stem=stem,
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
        """
        Re-read the analysis s-file into ``self.data_frame``.

        Notes
        -----
        This is a thin alias for ``load_scalars(source='sfile')`` to make intent explicit.
        """
        self.load_scalars(source="sfile")

    def copy_fresh_sfile_to_analysis(self) -> None:
        """
        Replace the analysis s-file with the fresh copy from the scan folder.

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
        """
        Load and parse the ECS Live Dump file for this scan via ``ScanPaths``.

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
