"""Base class for scan analyzers that process a single device's data.

This module provides :class:`SingleDeviceScanAnalyzer`, an abstract base class
that handles the orchestration of running an ImageAnalyzer across all shots in a scan.
It provides:

- Robust data file discovery per shot via filename pattern matching
- A fused per-shot ``analyze_image_file`` pipeline (default) that runs
  load+analyze atomically per task
- A streaming per-bin pipeline for analyses that operate on the
  bin-averaged image (e.g. nonlinear measurements where image-then-mean
  differs from mean-then-image)
- Turnkey post-processing outputs via renderer delegation
- Saving outputs and updating the scan's auxiliary s-file

Subclasses must provide:
- An ImageAnalyzer instance
- A renderer instance (e.g., Image2DRenderer, Line1DRenderer)
- File pattern matching logic (via file_tail parameter)
"""

from __future__ import annotations

# --- Standard Library ---
import logging
import re
import pickle
from abc import ABC
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypedDict, Dict, Literal, Iterable

# --- Third-Party Libraries ---
import pandas as pd
import numpy as np

# --- Local / Project Imports ---
from scan_analysis.base import DataUnavailableWarning, ScanAnalyzer

# --- Type-Checking Imports ---
if TYPE_CHECKING:
    from image_analysis.types import ImageAnalyzerResult
    from image_analysis.base import ImageAnalyzer
    from scan_analysis.analyzers.renderers import BaseRenderer

logger = logging.getLogger(__name__)


# %% TypedDict Definitions
class BinDataEntry(TypedDict):
    """Container for per-bin aggregates."""

    value: float
    result: Optional[ImageAnalyzerResult]


def _apply_prefix_suffix(scalars: dict, prefix: str, suffix: str) -> dict:
    """Return a new scalars dict with ``{prefix}_{key}{suffix}`` keys (#412).

    ImageAnalysis emits bare scalar keys; this function applies the
    namespacing on the way to storage in ``self.results``. Empty / falsy
    prefix means no underscore is prepended (explicit unprefixed opt-in,
    distinct from the default ``device_name`` fallback applied at
    constructor time).
    """
    if not scalars:
        return scalars
    pre = f"{prefix}_" if prefix else ""
    return {f"{pre}{k}{suffix}": v for k, v in scalars.items()}


# %% Base Class
class SingleDeviceScanAnalyzer(ScanAnalyzer, ABC):
    """
    Base class for scan analyzers that process a single device's data.

    This class orchestrates running an ImageAnalyzer across all shots in
    a scan, handling parallelized I/O, binning, averaging, and rendering.

    The ImageAnalyzer must implement:
    - ``load_image(path) -> np.ndarray`` (or compatible array type)
    - ``analyze_image(image, auxiliary_data: dict|None) -> AnalyzerResultDict``

    Parameters
    ----------
    device_name : str
        Device subfolder name used to locate data files within a scan directory.
    image_analyzer : ImageAnalyzer
        ImageAnalyzer instance to apply per image.
    renderer : BaseRenderer
        Renderer instance for visualization outputs.
    file_tail : str, optional
        Suffix/extension used to match data files (e.g., ".png", ".himg").
    skip_plt_show : bool, default=True
        Passed to :class:`ScanAnalyzer` to control interactive plotting.
    flag_save_data : bool, default=True
        If True, saves outputs to the analysis directory.

    Notes
    -----
    - If ``image_analyzer`` cannot be pickled, the analyzer automatically switches
      to threaded execution instead of multiprocessing.
    """

    def __init__(
        self,
        device_name: str,
        image_analyzer: ImageAnalyzer,
        renderer: BaseRenderer,
        file_tail: Optional[str] = ".png",
        skip_plt_show: bool = True,
        flag_save_data: bool = True,
        analysis_mode: Literal["per_shot", "per_bin"] = "per_shot",
        data_device_name: Optional[str] = None,
        use_injected_data: bool = False,
        output_name: Optional[str] = None,
        metric_suffix: str = "",
    ):
        """Initialize the analyzer and validate concurrency constraints."""
        if not device_name:
            raise ValueError("SingleDeviceScanAnalyzer requires a device_name.")

        super().__init__(
            device_name=device_name,
            skip_plt_show=skip_plt_show,
            use_injected_data=use_injected_data,
        )

        self.image_analyzer = image_analyzer
        self.renderer = renderer

        self.max_workers = 16
        self.saved_avg_data_paths: Dict[int, Path] = {}
        self.data_device_name: str = data_device_name or device_name
        self.results: dict = {}

        # Define flags
        self.flag_save_data = flag_save_data
        self.file_tail = file_tail
        self.analysis_mode = analysis_mode

        # Output identifier (#412). The diagnostic factory passes
        # ``effective_output_name`` (= the diagnostic's ``output_name`` if
        # set, else the device name) and an optional ``metric_suffix``.
        # When constructed directly (Mode 1, notebook use), ``output_name``
        # defaults to None and we fall back to ``device_name`` here — same
        # convention. ``ImageAnalyzer`` emits bare scalar keys; this class
        # applies the ``output_name`` prefix + ``metric_suffix`` in
        # ``_consume_result`` before storing results, so all downstream
        # consumers (s-file writer, in-memory optimizer evaluator, render
        # data) see consistently namespaced keys. Per-analyzer output
        # directories are also keyed off ``_output_name`` so the on-disk
        # layout matches the s-file columns.
        self._output_name: str = output_name if output_name is not None else device_name
        self._metric_suffix: str = metric_suffix or ""

        # Check if image_analyzer is pickleable
        try:
            pickle.dumps(self.image_analyzer)
        except (pickle.PicklingError, TypeError) as e:
            # Mark that we cannot send the image analyzer through multiprocessing
            self.image_analyzer.run_analyze_image_asynchronously = True
            logger.warning(
                f"[{self.__class__.__name__}] ImageAnalyzer instance is not pickleable "
                f"(reason: {e}). Falling back to threaded analysis."
            )

    def _establish_additional_paths(self):
        """Compute input/output paths and validate data presence."""
        # Get the analyzer-specific output identifier if available
        # (Standard family exposes ``output_name``); fall back to the
        # device name when the analyzer doesn't carry one.
        analyzer_name = (
            getattr(self.image_analyzer, "output_name", None) or self.device_name
        )

        # Organize various paths for location of saved data
        self.path_dict = {
            "data": Path(self.scan_directory) / f"{self.data_device_name}",
            "save": (
                self.scan_directory.parents[1]
                / "analysis"
                / self.scan_directory.name
                / f"{analyzer_name}"
                / self.__class__.__name__
            ),
        }

        # Raise a clean sentinel if the data directory is missing or empty so
        # _run_analysis_core can skip gracefully without printing a full traceback.
        data_dir = self.path_dict["data"]
        if not data_dir.exists() or not any(data_dir.iterdir()):
            raise DataUnavailableWarning(
                f"Data directory '{data_dir}' does not exist or is empty for "
                f"device '{self.device_name}'. Skipping analysis."
            )

    def _run_analysis_core(self):
        """
        Main analysis pipeline executed by the orchestrator.

        Workflow
        --------
        1. Establish paths and create the output directory (if saving is enabled).
        2. Run mode-specific analysis via :meth:`_process_all_shots`:
           ``per_shot`` fuses load+analyze in one task per shot;
           ``per_bin`` streams bin-by-bin (load → average → analyze).
           Both paths start with a one-time background pre-pass.
        3. Post-process:
           - If ``noscan``: average + animation.
           - Else: per-bin averaging + summary figure.
        4. Persist auxiliary s-file updates and return list of display artifacts.

        Returns
        -------
        list[str | pathlib.Path]
            Paths to generated artifacts to display.

        Raises
        ------
        DataUnavailableWarning
            If the device data directory is missing or empty. Callers are
            expected to log this as a non-error skip.
        Exception
            Any unhandled error from the analysis pipeline propagates so the
            task queue marks the task as ``failed`` (with a captured error
            message) rather than silently writing ``done`` with no artifacts.
        """
        try:
            self.renderer.display_contents = []
            self._establish_additional_paths()

            if self.flag_save_data and not self.path_dict["save"].exists():
                self.path_dict["save"].mkdir(parents=True)

            # Run the data analyzer on every shot
            self._process_all_shots()

            # Depending on the scan type, perform additional processing
            if len(self.results) > 2:
                if self.noscan:
                    self._postprocess_noscan()
                else:
                    self._postprocess_scan()

            if not self.use_injected_data:
                pending = getattr(self, "_pending_aux_updates", [])
                if pending:
                    df_updates = pd.DataFrame(pending)
                    if not df_updates.empty:
                        self.append_to_sfile(df_updates)
                self._pending_aux_updates = []
            return self.renderer.display_contents

        except DataUnavailableWarning as e:
            logger.warning(str(e))
            raise

    def _process_all_shots(self):
        """Run mode-specific per-shot or per-bin analysis.

        ``per_shot`` mode fuses load+analyze in a single
        ``analyze_image_file`` call per shot. ``per_bin`` mode streams
        bin-by-bin: load just one bin's files, average, analyze, release.
        Neither path materialises all shots in memory simultaneously, and
        per-shot data never has to shuttle through analyzer-instance state
        between separate pipeline phases.
        """
        # One-time pre-pass: resolve {scan_dir} placeholders and apply
        # any scan.background_source directive. Mutates
        # `image_analyzer.camera_config.background` in-place so the
        # downstream per-shot pipeline picks up a static FROM_FILE bg.
        self._resolve_background_paths()

        self._build_data_file_map()

        # Reset per-run state.
        self.results = {}
        self._pending_aux_updates: list[dict] = []

        if self.analysis_mode == "per_shot":
            self._analyze_per_shot()
        else:  # per_bin
            self._analyze_per_bin_streaming()

    def _build_data_file_map(self) -> None:
        """
        Build a mapping from shot number to data file path.

        Two strategies, selected automatically by the metadata present:

        - **acq_timestamp join** (Bluesky-produced scans): when the auxiliary
          frame carries this device's ``acq_timestamp`` column, each shot's
          file is identified by the device's own per-shot timestamp — the
          device names its natively saved files with the same value it
          streams into the event row, so the join is a deterministic lookup
          (canonicalised to integer milliseconds), never a filename guess.
          The join is strictly per-device: this device's column against this
          device's folder; other devices' clocks never enter it.
        - **shot-number filenames** (MC-produced scans): the legacy pattern
          ``Scan<scan_number>_<device_subject>_<shot_number><file_tail>``,
          e.g. ``Scan012_UC_ALineEBeam3_005.png``.

        Only files whose suffix + format matches ``file_tail`` exactly are
        included in either strategy.
        """
        self._data_file_map = {}

        # Check if data directory exists
        if not self.path_dict["data"].exists():
            logger.warning(
                f"Data directory does not exist: {self.path_dict['data']}. "
                f"Skipping file mapping for device '{self.device_name}'."
            )
            return

        logger.info(f"self.file_tail: {self.file_tail}")

        ts_column = self._acq_timestamp_column()
        if ts_column is not None:
            logger.info("Mapping files by device acq_timestamp (column %r)", ts_column)
            self._map_files_by_acq_timestamp(ts_column)
        else:
            logger.info("Mapping matched files")
            self._map_files_by_shot_number()

        expected_shots = set(self.auxiliary_data["Shotnumber"].values)
        found_shots = set(self._data_file_map.keys())
        for m in sorted(expected_shots - found_shots):
            logger.warning(f"No file found for shot {m}")

    def _map_files_by_shot_number(self) -> None:
        """Legacy strategy: parse shot numbers out of MC-convention filenames."""
        data_filename_regex = re.compile(
            r"Scan(?P<scan_number>\d{3,})_"  # scan number
            r"(?P<device_subject>.*?)_"  # non-greedy subject
            r"(?P<shot_number>\d{3,})"  # shot number
            + re.escape(self.file_tail)
            + r"$"  # literal suffix+format
        )

        for file in self.path_dict["data"].iterdir():
            if not file.is_file():
                continue

            m = data_filename_regex.match(file.name)
            if m:
                shot_num = int(m.group("shot_number"))
                if shot_num in self.auxiliary_data["Shotnumber"].values:
                    self._data_file_map[shot_num] = file
                    logger.info(f"Mapped file for shot {shot_num}: {file}")
            else:
                logger.debug(f"Filename {file.name} does not match expected pattern.")

    @staticmethod
    def _normalize_column_token(name: str) -> str:
        """Collapse a name to a lowercase ``_``-separated token for matching."""
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    def _acq_timestamp_column(self) -> Optional[str]:
        """Find this device's ``acq_timestamp`` column in the auxiliary frame.

        Recognises every spelling the column takes across data paths —
        ``"<Device> acq_timestamp"`` (s-file header), ``"<Device>:acq_timestamp"``
        (in-memory frame), ``"<device>-acq_timestamp"`` (raw event key) — by
        normalising both the device name and the column prefix to the same
        token. Returns ``None`` (→ legacy shot-number mapping) when absent.
        """
        if self.auxiliary_data is None:
            return None
        device_token = self._normalize_column_token(self.device_name)
        for column in self.auxiliary_data.columns:
            token = self._normalize_column_token(column)
            if token == f"{device_token}_acq_timestamp":
                return str(column)
        return None

    def _matching_valid_column(self) -> Optional[str]:
        """Find this device's ``valid`` column, if the frame carries one."""
        device_token = self._normalize_column_token(self.device_name)
        for column in self.auxiliary_data.columns:
            if self._normalize_column_token(column) == f"{device_token}_valid":
                return str(column)
        return None

    def _map_files_by_acq_timestamp(self, ts_column: str) -> None:
        """Join shots to files via this device's own per-shot ``acq_timestamp``.

        The device stamps one double per acquisition: streamed into the event
        row and written into the native filename (``<name>_<timestamp><tail>``,
        milliseconds precision). Both representations are canonicalised to
        integer milliseconds and joined exactly; the ±1 ms fallback below is
        float-formatting canonicalisation at the rounding boundary, not a
        physical tolerance window. Rows where the device's ``valid`` column is
        false are skipped — that device's frame belongs to a different
        physical shot, so "no file for this shot" is the correct answer.
        """
        data_dir = self.path_dict["data"]
        file_device = getattr(self, "data_device_name", None) or self.device_name

        # Directory listings over SMB can serve stale (cached) entries for
        # minutes after a file lands — long enough to hide files written
        # seconds ago during a live scan. The expected filename is fully
        # determined by the row timestamp, so probe it with a direct stat
        # (never served from the listing cache) first; the listing-based map
        # below is only a fallback for unconventional names.
        file_ts_regex = re.compile(
            r"_(?P<ts>\d+\.\d+)" + re.escape(self.file_tail) + r"$"
        )
        files_by_ms: dict[int, Path] = {}
        for file in data_dir.iterdir():
            if not file.is_file():
                continue
            m = file_ts_regex.search(file.name)
            if m:
                files_by_ms[round(float(m.group("ts")) * 1000)] = file

        valid_column = self._matching_valid_column()
        for _, row in self.auxiliary_data.iterrows():
            shot_num = int(row["Shotnumber"])
            if valid_column is not None and not bool(row[valid_column]):
                continue
            ts = row[ts_column]
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(ts) or ts <= 0:
                continue
            key = round(ts * 1000)
            file = None
            for k in (key, key - 1, key + 1):
                candidate = data_dir / f"{file_device}_{k / 1000:.3f}{self.file_tail}"
                if candidate.exists():
                    file = candidate
                    break
            if file is None:
                file = (
                    files_by_ms.get(key)
                    or files_by_ms.get(key - 1)
                    or files_by_ms.get(key + 1)
                )
            if file is not None:
                self._data_file_map[shot_num] = file
                logger.info(f"Mapped file for shot {shot_num}: {file}")

    def _resolve_background_paths(self) -> None:
        """
        Resolve background placeholders and any scan-context background source.

        Run once before the per-shot pipeline starts. Mutates
        ``image_analyzer.camera_config.background`` in place. Two
        operations happen here:

        1. Substitute ``{scan_dir}`` in ``file_path`` with the current
           scan's data directory.
        2. If the diagnostic config carried a ``scan.background_source``
           directive (cross-scan dark via ``scan_number`` or current-scan
           dynamic via ``from_current_scan``), compute-and-cache that
           background and rewrite ``bg_config`` to a static ``FROM_FILE``
           pointing at the cache.

        No-op when the image_analyzer has no ``camera_config.background``
        (e.g. 1D analyzers).
        """
        if not hasattr(self.image_analyzer, "camera_config"):
            return

        bg_config = self.image_analyzer.camera_config.background

        if not bg_config:
            return

        scan_dir = self.path_dict["data"]

        if bg_config.file_path and "{scan_dir}" in str(bg_config.file_path):
            resolved = str(bg_config.file_path).replace("{scan_dir}", str(scan_dir))
            bg_config.file_path = Path(resolved)
            logger.info(f"Resolved background file_path: {resolved}")

        directive = getattr(self, "background_source", None)
        if directive is not None:
            self._apply_background_source(directive, bg_config)

    def _apply_background_source(self, directive, bg_config) -> None:
        """Apply a ``scan.background_source`` directive to ``bg_config``.

        Dispatches on which variant of the directive is set
        (``scan_number``, ``from_current_scan``, or ``autodetect``),
        resolves the corresponding background, then rewrites
        ``bg_config`` to point at the result via the ``FROM_FILE``
        method. The downstream per-shot pipeline sees a static
        ``from_file`` background.
        """
        from image_analysis.processing.array2d.background import (
            compute_and_cache_scan_background,
        )
        from image_analysis.config.array2d_processing import BackgroundMethod

        if directive.scan_number is not None:
            from geecs_data_utils import ScanPaths, ScanTag as GeecsDataScanTag

            bg_tag = GeecsDataScanTag(
                year=self.scan_tag.year,
                month=self.scan_tag.month,
                day=self.scan_tag.day,
                number=directive.scan_number,
                experiment=self.scan_tag.experiment,
            )
            bg_scan_paths = ScanPaths(tag=bg_tag, read_mode=True)
            cache_path = compute_and_cache_scan_background(
                image_dir=bg_scan_paths.get_folder() / self.device_name,
                file_tail=self.file_tail,
                image_loader=self.image_analyzer.load_image,
                output_path=(
                    bg_scan_paths.get_analysis_folder()
                    / self.device_name
                    / f"{self.device_name}_background_avg.npy"
                ),
                method="mean",
            )
        elif directive.from_current_scan is not None:
            spec = directive.from_current_scan
            cache_path = compute_and_cache_scan_background(
                image_dir=self.scan_paths.get_folder() / self.device_name,
                file_tail=self.file_tail,
                image_loader=self.image_analyzer.load_image,
                output_path=(
                    self.scan_paths.get_analysis_folder()
                    / self.device_name
                    / "dynamic_background.npy"
                ),
                method=spec.method,
                percentile=spec.percentile,
            )
        elif directive.autodetect is not None:
            cache_path = self._find_autodetected_background_path()
        else:
            # BackgroundSource's model validator requires exactly one
            # source, so this should be unreachable.
            raise ValueError(
                f"background_source directive has no source variant set: {directive}"
            )

        bg_config.file_path = cache_path
        bg_config.method = BackgroundMethod.FROM_FILE

    def _find_autodetected_background_path(self) -> Path:
        """Find the current scan's precomputed averaged background file."""
        scan_number = self.scan_tag.number
        analysis_dir = self.scan_directory.parents[1] / "analysis"
        filename_pattern = re.compile(
            rf"^Scan{scan_number:03d}"
            rf"{re.escape(self.device_name)}"
            r"_averaged\.[^.]+$"
        )

        if not analysis_dir.is_dir():
            raise FileNotFoundError(
                f"Autodetect background directory not found: {analysis_dir}"
            )

        matches = sorted(
            path
            for path in analysis_dir.iterdir()
            if path.is_file() and filename_pattern.match(path.name)
        )

        if not matches:
            raise FileNotFoundError(
                "No autodetected background file found in "
                f"{analysis_dir} matching "
                f"Scan{scan_number:03d}{self.device_name}_averaged.<ext>"
            )
        if len(matches) > 1:
            raise ValueError(
                "Multiple autodetected background files found for "
                f"Scan{scan_number:03d} device {self.device_name}: "
                f"{', '.join(str(path) for path in matches)}"
            )

        logger.info("Autodetected background file: %s", matches[0])
        return matches[0]

    @staticmethod
    def _normalize_aux_value(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _aux_row_for_shot(self, shot_num: int) -> dict:
        if self.auxiliary_data is None:
            return {}
        if "Shotnumber" not in self.auxiliary_data.columns:
            return {}

        row = self.auxiliary_data.loc[self.auxiliary_data["Shotnumber"] == shot_num]
        if row.empty:
            return {}

        row_dict = row.iloc[0].to_dict()
        return {k: self._normalize_aux_value(v) for k, v in row_dict.items()}

    def _aux_mean_for_shots(self, shot_nums: Iterable[int]) -> dict:
        if self.auxiliary_data is None:
            return {}
        if "Shotnumber" not in self.auxiliary_data.columns:
            return {}

        shot_list = list(shot_nums)
        if not shot_list:
            return {}

        rows = self.auxiliary_data[self.auxiliary_data["Shotnumber"].isin(shot_list)]
        if rows.empty:
            return {}

        numeric = rows.select_dtypes(include=[np.number])
        if numeric.empty:
            return {}

        drop_cols = [col for col in ("Shotnumber", "Bin #") if col in numeric.columns]
        if drop_cols:
            numeric = numeric.drop(columns=drop_cols)
        if numeric.empty:
            return {}

        means = numeric.mean(numeric_only=True).to_dict()
        return {k: self._normalize_aux_value(v) for k, v in means.items()}

    def _group_files_by_bin(self) -> dict:
        """Group ``_data_file_map`` entries by ``Bin #`` (or one bin for noscan).

        Returns a mapping ``{bin_key: (shot_nums: list[int], paths: list[Path])}``.
        """
        if "Bin #" not in self.auxiliary_data.columns:
            return {
                0: (
                    list(self._data_file_map.keys()),
                    list(self._data_file_map.values()),
                )
            }

        bins: dict = {}
        for bin_num in self.auxiliary_data["Bin #"].unique():
            bin_shots = self.auxiliary_data[self.auxiliary_data["Bin #"] == bin_num][
                "Shotnumber"
            ].values.tolist()
            bin_paths = [
                self._data_file_map[s] for s in bin_shots if s in self._data_file_map
            ]
            if bin_paths:
                bins[int(bin_num)] = (bin_shots, bin_paths)

        return bins

    def _has_valid_result(self, result: ImageAnalyzerResult) -> bool:
        """
        Check if analyzer result contains valid data.

        Valid data can be in:
        - processed_image (for 2D analyzers)
        - line_data (for 1D analyzers)

        Parameters
        ----------
        result : ImageAnalyzerResult
            Result from analyzer

        Returns
        -------
        bool
            True if result contains valid data
        """
        return result.processed_image is not None or result.line_data is not None

    def _analyze_per_shot(self) -> None:
        """Fused per-shot pipeline using ``analyze_image_file``.

        For each shot, a single task calls ``analyze_image_file(path, aux)``
        which loads and analyzes that shot atomically. Per-shot data
        never has to travel through analyzer-instance state between a
        separate load phase and analyze phase — this is the core
        correctness property the refactor enforces.

        Concurrency backend follows
        ``run_analyze_image_asynchronously`` on the ImageAnalyzer.
        """
        if not self._data_file_map:
            logger.warning("No data files mapped; nothing to analyze.")
            return

        # Pre-compute aux dicts in the parent process; they reference
        # auxiliary_data which we do not need to ship to workers.
        tasks = []
        for shot_num, path in self._data_file_map.items():
            aux = self._aux_row_for_shot(shot_num)
            aux["file_path"] = path
            tasks.append((shot_num, path, aux))

        use_threads = self.image_analyzer.run_analyze_image_asynchronously
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        logger.info(
            "per_shot analysis via %s; %d shots",
            "ThreadPoolExecutor" if use_threads else "ProcessPoolExecutor",
            len(tasks),
        )

        with Executor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.image_analyzer.analyze_image_file, path, aux
                ): shot_num
                for shot_num, path, aux in tasks
            }

            for future in as_completed(futures):
                shot_num = futures[future]
                try:
                    result: ImageAnalyzerResult = future.result()
                    self._consume_result(shot_num, [shot_num], result)
                except Exception as e:
                    logger.error(f"Analysis failed for shot {shot_num}: {e}")

    def _analyze_per_bin_streaming(self) -> None:
        """Streaming per-bin pipeline: load → average → analyze, bin by bin.

        Only one bin's images are materialised at a time. Within each
        bin, file loads run in parallel via the configured executor;
        bins themselves are processed serially so memory stays bounded.
        Use ``per_bin`` mode for analyzers where running on the
        bin-averaged image is scientifically distinct from per-shot
        analysis + result averaging (e.g. nonlinear measurements).
        """
        bins = self._group_files_by_bin()
        if not bins:
            logger.warning("No bins to analyze; data file map is empty.")
            return

        use_threads = self.image_analyzer.run_analyze_image_asynchronously
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        logger.info(
            "per_bin streaming analysis via %s; %d bin(s)",
            "ThreadPoolExecutor" if use_threads else "ProcessPoolExecutor",
            len(bins),
        )

        with Executor(max_workers=self.max_workers) as executor:
            for bin_key, (bin_shots, bin_paths) in bins.items():
                # Parallel load this bin's files; discard any that failed.
                load_futures = {
                    executor.submit(self.image_analyzer.load_image, p): p
                    for p in bin_paths
                }
                images = []
                for fut in as_completed(load_futures):
                    path = load_futures[fut]
                    try:
                        img = fut.result()
                        if img is not None:
                            images.append(img)
                    except Exception as e:
                        logger.warning(
                            "Skipping %s in bin %s (load failed: %s)",
                            path,
                            bin_key,
                            e,
                        )

                if not images:
                    logger.warning("Bin %s has no loadable images; skipping", bin_key)
                    continue

                averaged = self.average_data(images)
                aux = self._aux_mean_for_shots(bin_shots)

                try:
                    result: ImageAnalyzerResult = self.image_analyzer.analyze_image(
                        averaged, aux
                    )
                    self._consume_result(bin_key, bin_shots, result)
                except Exception as e:
                    logger.error(f"Analysis failed for bin {bin_key}: {e}")

    def _consume_result(
        self,
        unit_key,
        sfile_keys,
        result: ImageAnalyzerResult,
    ) -> None:
        """Store a result and queue numeric scalars for s-file append.

        Shared between per-shot and per-bin paths. ``sfile_keys`` is the
        list of shot numbers whose s-file rows should receive the
        scalars: ``[shot_num]`` for per-shot, the bin's shot list for
        per-bin.

        Applies the diagnostic's output-naming here (#412):
        ``ImageAnalyzer`` emits bare keys; this is the single layer
        where they get namespaced with ``output_name`` (prefix) and
        ``metric_suffix`` (suffix). After the rewrite, every downstream
        consumer (s-file writer below, optimizer evaluator reading
        ``self.results[shot].scalars`` in memory) sees the same
        ``{output_name}_{key}{metric_suffix}`` shape.
        """
        result.scalars = _apply_prefix_suffix(
            result.scalars, self._output_name, self._metric_suffix
        )
        analysis_results = result.scalars

        if self._has_valid_result(result):
            self.results[unit_key] = result
            logger.info("Unit %s: valid data stored.", unit_key)
            logger.info("Analyzed unit %s and got %s", unit_key, analysis_results)
        else:
            logger.info("Unit %s: no valid data returned from analysis.", unit_key)

        numeric_updates = {
            key: value
            for key, value in analysis_results.items()
            if isinstance(value, (int, float, np.number))
        }
        non_numeric = set(analysis_results) - set(numeric_updates)
        if non_numeric:
            logger.warning(
                "[%s using %s] Non-numeric scalar keys skipped: %s",
                self.__class__.__name__,
                self.image_analyzer.__class__.__name__,
                sorted(non_numeric),
            )

        if not numeric_updates:
            return

        for shot_num in sfile_keys:
            for key, value in numeric_updates.items():
                self.auxiliary_data.loc[
                    self.auxiliary_data["Shotnumber"] == shot_num, key
                ] = value
            self._pending_aux_updates.append(
                {"Shotnumber": shot_num, **numeric_updates}
            )

    def _postprocess_noscan(self) -> None:
        """
        Post-process noscan data (dimension-specific).

        Must be implemented by subclasses to handle dimension-specific visualization:
        - 1D: average with std dev + waterfall plot
        - 2D: average image + animated GIF

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _postprocess_noscan()"
        )

    def _postprocess_scan(self) -> None:
        """
        Post-process scan data (dimension-specific).

        Must be implemented by subclasses to handle dimension-specific visualization:
        - 1D: waterfall plot from binned data
        - 2D: image grid from binned data

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _postprocess_scan()"
        )

    def _get_renderer_config(self):
        """
        Get renderer configuration for this analyzer.

        Subclasses should override this to provide their specific config type.
        This base implementation returns a generic config with common settings.

        Returns
        -------
        BaseRendererConfig
            Renderer configuration object
        """
        from scan_analysis.analyzers.renderers.config import BaseRendererConfig

        # Get renderer_kwargs if available (for backward compatibility)
        renderer_kwargs = getattr(self, "renderer_kwargs", {})

        # Handle legacy plot_scale parameter
        plot_scale = (getattr(self, "camera_analysis_settings", {}) or {}).get(
            "Plot Scale", None
        )
        if plot_scale is not None and "vmax" not in renderer_kwargs:
            renderer_kwargs["vmax"] = plot_scale

        # Create config from kwargs
        try:
            return BaseRendererConfig(**renderer_kwargs)
        except Exception as e:
            logger.warning(f"Error creating renderer config: {e}. Using defaults.")
            return BaseRendererConfig()

    def get_binned_data(self) -> dict[int, BinDataEntry]:
        """
        Get binned data, handling both per_shot and per_bin analysis modes.

        In per_shot mode: bins and averages the per-shot results.
        In per_bin mode: reformats the already-binned results.

        Returns
        -------
        dict[int, BinDataEntry]
            Binned data ready for rendering.
        """
        if self.analysis_mode == "per_bin":
            # Results are already binned, just reformat
            return self._convert_per_bin_results_to_binned_format()
        else:
            # Results are per-shot, need to bin them
            return self.bin_data_from_results()

    def _convert_per_bin_results_to_binned_format(self) -> dict[int, BinDataEntry]:
        """
        Convert per_bin results to binned_data format.

        In per_bin mode, self.results is already keyed by bin numbers with
        averaged data. This method just adds the scan parameter values.

        Returns
        -------
        dict[int, BinDataEntry]
            Binned data with scan parameter values.
        """
        binned_data = {}

        # Sort bin numbers to ensure consistent ordering
        for bin_num in sorted(self.results.keys()):
            result = self.results[bin_num]
            # Get the scan parameter value for this bin
            column_full_name, _ = self.find_scan_param_column()
            param_value = self.auxiliary_data.loc[
                self.auxiliary_data["Bin #"] == bin_num, column_full_name
            ].mean()

            binned_data[bin_num] = {"value": float(param_value), "result": result}

        return binned_data

    def bin_data_from_results(self) -> dict[int, BinDataEntry]:
        """
        Bin processed data by scan parameter and compute per-bin averages.

        For each unique "Bin #" value in the auxiliary s-file:
        - Select shots with valid processed data
        - Average all ImageAnalyzerResult objects for the bin
        - Preserve all render_data including projections and overlays

        Returns
        -------
        dict[int, BinDataEntry]
            Mapping from bin number to a dictionary with:
            - ``"value"`` : representative scan parameter value for the bin (float)
            - ``"result"`` : ImageAnalyzerResult with averaged data
        """
        if "Bin #" not in self.auxiliary_data.columns:
            logger.warning("Missing 'Bin #' column in auxiliary data.")
            return {}

        from image_analysis.types import ImageAnalyzerResult

        unique_bins = [int(b) for b in np.unique(self.auxiliary_data["Bin #"].values)]
        logger.info(f"Unique bins from auxiliary data: {unique_bins}")

        binned_data: dict[int, BinDataEntry] = {}

        for bin_val in unique_bins:
            # Get shot numbers in this bin
            bin_shots = self.auxiliary_data[self.auxiliary_data["Bin #"] == bin_val][
                "Shotnumber"
            ].values

            # Filter to shots with valid results
            valid_shots = [
                sn
                for sn in bin_shots
                if sn in self.results and self._has_valid_result(self.results[sn])
            ]

            if not valid_shots:
                logger.warning(f"No data found for bin {bin_val}.")
                continue

            # Simply average the ImageAnalyzerResult objects!
            results_to_average = [self.results[sn] for sn in valid_shots]
            binned_result = ImageAnalyzerResult.average(results_to_average)

            # Get scan parameter value for this bin
            column_full_name, _ = self.find_scan_param_column()
            param_value = self.auxiliary_data.loc[
                self.auxiliary_data["Bin #"] == bin_val, column_full_name
            ].mean()

            binned_data[bin_val] = {
                "value": float(param_value),
                "result": binned_result,
            }

        return binned_data

    def cleanup(self) -> None:
        """
        Free memory held by loaded and analyzed data after analysis is complete.

        Clears per-scan attributes that may hold large numpy arrays or result objects:

        - ``results`` — per-shot/bin ImageAnalyzerResult objects
        - ``_data_file_map`` — shot-to-path mapping
        - ``_pending_aux_updates`` — queued s-file row updates

        Also delegates to ``renderer.cleanup()``.
        """
        self.results = {}
        self._data_file_map = {}
        self._pending_aux_updates = []

        self.renderer.cleanup()

        logger.debug(f"[{self.__class__.__name__}] cleanup() complete.")

    @staticmethod
    def average_data(data_list: list[np.ndarray]) -> Optional[np.ndarray]:
        """Return the element-wise mean of a list of data arrays.

        Returns ``None`` for an empty list, or when the arrays have
        inhomogeneous shapes (which numpy cannot stack into a single mean).
        The latter happens for 1D analyzers whose per-shot lineouts depend
        on ROI/threshold masking — e.g. FROG spectral phase, where each
        shot's valid wavelength coverage differs.
        """
        if len(data_list) == 0:
            return None

        shapes = {arr.shape for arr in data_list}
        if len(shapes) > 1:
            logger.warning(
                "Cannot average %d arrays with inhomogeneous shapes %s; "
                "returning None so the caller can skip the averaged figure.",
                len(data_list),
                sorted(shapes),
            )
            return None

        return np.mean(data_list, axis=0)
