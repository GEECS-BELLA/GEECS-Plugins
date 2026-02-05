"""Base class for scan analyzers that process a single device's data.

This module provides :class:`SingleDeviceScanAnalyzer`, an abstract base class
that handles the orchestration of running an ImageAnalyzer across all shots in a scan.
It provides:

- Robust data file discovery per shot via filename pattern matching
- Parallelized loading and per-shot analysis (threaded vs. process pools)
- Optional batch-level preprocessing via the ImageAnalyzer API
- Binning by scan parameter and per-bin averaging
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
import traceback
import pickle
from abc import ABC
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypedDict, Dict, Literal

# --- Third-Party Libraries ---
import pandas as pd
import numpy as np

# --- Local / Project Imports ---
from scan_analysis.base import ScanAnalyzer
from scan_analysis.provenance.capture import extract_config_from_analyzer

# --- Type-Checking Imports ---
if TYPE_CHECKING:
    from image_analysis.types import ImageAnalyzerResult
    from image_analysis.base import ImageAnalyzer
    from scan_analysis.analyzers.renderers import BaseRenderer

PRINT_TRACEBACK = True

logger = logging.getLogger(__name__)


# %% TypedDict Definitions
class BinDataEntry(TypedDict):
    """Container for per-bin aggregates."""

    value: float
    result: Optional[ImageAnalyzerResult]


# %% Base Class
class SingleDeviceScanAnalyzer(ScanAnalyzer, ABC):
    """
    Base class for scan analyzers that process a single device's data.

    This class orchestrates running an ImageAnalyzer across all shots in a scan,
    handling parallelized I/O, batch preprocessing, binning, averaging, and rendering.

    The ImageAnalyzer must implement:
    - ``load_image(path) -> np.ndarray`` (or compatible array type)
    - ``analyze_image(image, auxiliary_data: dict|None) -> AnalyzerResultDict``
    - Optionally ``analyze_image_batch(images: list) -> (list, dict)``

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
    ):
        """Initialize the analyzer and validate concurrency constraints."""
        if not device_name:
            raise ValueError("SingleDeviceScanAnalyzer requires a device_name.")

        super().__init__(device_name=device_name, skip_plt_show=skip_plt_show)

        self.image_analyzer = image_analyzer
        self.renderer = renderer

        self.max_workers = 16
        self.saved_avg_data_paths: Dict[int, Path] = {}

        # Define flags
        self.flag_save_data = flag_save_data
        self.file_tail = file_tail
        self.analysis_mode = analysis_mode

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
        # Get the analyzer-specific name if available
        analyzer_name = (
            getattr(self.image_analyzer, "camera_name", None) or self.device_name
        )

        # Organize various paths for location of saved data
        self.path_dict = {
            "data": Path(self.scan_directory) / f"{self.device_name}",
            "save": (
                self.scan_directory.parents[1]
                / "analysis"
                / self.scan_directory.name
                / f"{analyzer_name}"
                / self.__class__.__name__
            ),
        }

        # Check if data directory exists and is not empty
        if not self.path_dict["data"].exists() or not any(
            self.path_dict["data"].iterdir()
        ):
            logger.warning(
                f"Data directory '{self.path_dict['data']}' does not exist or is empty. Skipping"
            )

        if self.path_dict["data"] is None or self.auxiliary_data is None:
            logger.info("Skipping analysis due to missing data or auxiliary file.")
            return

    def _run_analysis_core(self):
        """
        Main analysis pipeline executed by the orchestrator.

        Workflow
        --------
        1. Establish paths and create the output directory (if saving is enabled).
        2. Load all data files in parallel (threaded/process).
        3. Optionally run batch-level preprocessing via ``analyze_image_batch``.
        4. Run per-shot ``analyze_image`` in parallel and collect results.
        5. Post-process:
           - If ``noscan``: average + animation.
           - Else: per-bin averaging + summary figure.
        6. Persist auxiliary s-file updates and return list of display artifacts.

        Returns
        -------
        list[str | pathlib.Path] or None
            Paths to generated artifacts to display, or None on failure.
        """
        self._establish_additional_paths()

        if self.flag_save_data and not self.path_dict["save"].exists():
            self.path_dict["save"].mkdir(parents=True)

        try:
            # Run the data analyzer on every shot
            self._process_all_shots()

            # Depending on the scan type, perform additional processing
            if len(self.results) > 2:
                if self.noscan:
                    self._postprocess_noscan()
                else:
                    self._postprocess_scan()

            if not self.live_analysis:
                pending = getattr(self, "_pending_aux_updates", [])
                if pending:
                    df_updates = pd.DataFrame(pending)
                    if not df_updates.empty:
                        self.append_to_sfile(df_updates)
                self._pending_aux_updates = []
            return self.renderer.display_contents

        except Exception as e:
            if PRINT_TRACEBACK:
                print(traceback.format_exc())
            logger.warning(f"Warning: Data analysis failed due to: {e}")
            return

    def _process_all_shots(self):
        """Load data files, run batch analysis (optional), then mode-based analysis."""
        self._load_all_data()
        self._run_batch_analysis()

        # Prepare analysis units based on mode
        if self.analysis_mode == "per_shot":
            analysis_units = self._prepare_per_shot_units()
        else:  # per_bin
            analysis_units = self._prepare_per_bin_units()

        # Run unified analysis loop
        self._analyze_units(analysis_units)

    def _build_data_file_map(self) -> None:
        """
        Build a mapping from shot number to data file path using a flexible filename regex.

        Only files whose suffix + format matches ``file_tail`` exactly are included.

        Notes
        -----
        Expected filename pattern::

            Scan<scan_number>_<device_subject>_<shot_number><file_tail>

        Examples
        --------
        ``Scan012_UC_ALineEBeam3_005.png`` or ``Scan016_U_HasoLift_001_postprocessed.tsv``
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
        data_filename_regex = re.compile(
            r"Scan(?P<scan_number>\d{3,})_"  # scan number
            r"(?P<device_subject>.*?)_"  # non-greedy subject
            r"(?P<shot_number>\d{3,})"  # shot number
            + re.escape(self.file_tail)
            + r"$"  # literal suffix+format
        )

        logger.info("Mapping matched files")
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

        expected_shots = set(self.auxiliary_data["Shotnumber"].values)
        found_shots = set(self._data_file_map.keys())
        for m in sorted(expected_shots - found_shots):
            logger.warning(f"No file found for shot {m}")

    def _load_all_data(self) -> None:
        """
        Load all data files in parallel and store them in ``self.raw_data``.

        This identifies the data file path for each shot number using the configured
        filename pattern, then loads data via the ImageAnalyzer's ``load_image`` method.

        Concurrency is selected by the analyzer's ``run_analyze_image_asynchronously`` flag.

        Side Effects
        ------------
        Populates:

        - ``self._data_file_map`` : ``{shot_number: Path}``
        - ``self.raw_data`` : ``{shot_number: (data, Path)}``
        """
        self.raw_data = {}
        self._build_data_file_map()

        use_threads = self.image_analyzer.run_analyze_image_asynchronously
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

        with Executor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.image_analyzer.load_image, path): shot_num
                for shot_num, path in self._data_file_map.items()
            }

            for future in as_completed(futures):
                shot_num = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        file_path = self._data_file_map[shot_num]
                        self.raw_data[shot_num] = (data, file_path)
                except Exception as e:
                    logger.error(f"Error loading data for shot {shot_num}: {e}")

    def _run_batch_analysis(self) -> None:
        """
        Optionally run batch-level preprocessing across all loaded images.

        Calls the ImageAnalyzer's ``analyze_image_batch`` method with the list of raw images.
        If it returns a list of processed images and a state dict, those processed
        images replace the originals for subsequent per-shot analysis.

        Raises
        ------
        RuntimeError
            If no data has been loaded yet.
        ValueError
            If the returned number of processed images does not match the input.
        """
        if not hasattr(self, "raw_data") or not self.raw_data:
            raise RuntimeError("No data loaded. Run _load_all_data first.")

        # Resolve {scan_dir} placeholders in background config BEFORE batch processing
        self._resolve_background_paths()

        try:
            # Extract keys and separate data + path tuples
            shot_nums = list(self.raw_data.keys())
            data_path_tuples = list(self.raw_data.values())
            data_list = [data for data, _ in data_path_tuples]
            file_paths = [path for _, path in data_path_tuples]

            # Run batch analysis
            processed_data, stateful_results = self.image_analyzer.analyze_image_batch(
                data_list
            )
            self.stateful_results = stateful_results
            logger.info(
                "Finished batch processing with %s stateful results", stateful_results
            )

            if processed_data is None:
                logger.warning("analyze_image_batch() returned None. Skipping.")
                self.raw_data = {}
                return

            if len(processed_data) != len(shot_nums):
                raise ValueError(
                    f"analyze_image_batch() returned {len(processed_data)} items, "
                    f"but {len(shot_nums)} were expected."
                )

            self.raw_data = dict(zip(shot_nums, zip(processed_data, file_paths)))

        except Exception as e:
            logger.warning(f"Batch analysis skipped or failed: {e}")

    def _resolve_background_paths(self) -> None:
        """
        Resolve {scan_dir} placeholders in ImageAnalyzer's background configuration.

        This is called once before batch processing, ensuring all subsequent
        parallel workers use concrete paths.
        """
        # Check if analyzer has camera_config with background settings
        if not hasattr(self.image_analyzer, "camera_config"):
            return

        bg_config = self.image_analyzer.camera_config.background

        if not bg_config:
            return

        scan_dir = self.path_dict["data"]

        # Resolve file_path
        if bg_config.file_path and "{scan_dir}" in str(bg_config.file_path):
            resolved = str(bg_config.file_path).replace("{scan_dir}", str(scan_dir))
            bg_config.file_path = Path(resolved)
            logger.info(f"Resolved background file_path: {resolved}")

        # Resolve auto_save_path in dynamic_computation
        if (
            bg_config.dynamic_computation
            and bg_config.dynamic_computation.auto_save_path
            and "{scan_dir}" in str(bg_config.dynamic_computation.auto_save_path)
        ):
            resolved = str(bg_config.dynamic_computation.auto_save_path).replace(
                "{scan_dir}", str(scan_dir)
            )
            bg_config.dynamic_computation.auto_save_path = Path(resolved)
            logger.info(f"Resolved background auto_save_path: {resolved}")

    def _prepare_per_shot_units(self) -> dict:
        """
        Prepare per-shot analysis units (current behavior).

        Returns
        -------
        dict
            Dictionary mapping shot numbers to unit data containing:
            - 'image': image data to analyze
            - 'auxiliary': auxiliary data dict for analyzer
            - 'sfile_keys': list containing just this shot number
        """
        return {
            shot_num: {
                "image": data,
                "auxiliary": {"file_path": path, **self.stateful_results},
                "sfile_keys": [shot_num],
            }
            for shot_num, (data, path) in self.raw_data.items()
        }

    def _prepare_per_bin_units(self) -> dict:
        """
        Prepare per-bin analysis units (bin first, then analyze).

        Returns
        -------
        dict
            Dictionary mapping bin numbers to unit data containing:
            - 'image': averaged image data for the bin
            - 'auxiliary': auxiliary data dict for analyzer
            - 'sfile_keys': list of all shot numbers in this bin
        """
        if "Bin #" not in self.auxiliary_data.columns:
            # Noscan: treat all as one bin
            all_images = [data for data, _ in self.raw_data.values()]
            all_shots = list(self.raw_data.keys())
            return {
                0: {
                    "image": self.average_data(all_images),
                    "auxiliary": self.stateful_results,
                    "sfile_keys": all_shots,
                }
            }

        units = {}
        for bin_num in self.auxiliary_data["Bin #"].unique():
            # Get shots in this bin
            bin_shots = self.auxiliary_data[self.auxiliary_data["Bin #"] == bin_num][
                "Shotnumber"
            ].values

            # Collect images for these shots
            images = [
                self.raw_data[shot][0] for shot in bin_shots if shot in self.raw_data
            ]

            if images:
                units[int(bin_num)] = {
                    "image": self.average_data(images),
                    "auxiliary": self.stateful_results,
                    "sfile_keys": bin_shots.tolist(),
                }

        return units

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

    def _analyze_units(self, analysis_units: dict) -> None:
        """
        Analyze units in parallel (threaded or multi-processed).

        Uses the ImageAnalyzer's ``analyze_image`` method on each unit, forwarding
        auxiliary data. Numeric scalars from the analyzer's return dictionary are
        persisted into the s-file for all relevant shots.

        Parameters
        ----------
        analysis_units : dict
            Dictionary mapping unit keys to unit data containing:
            - 'image': image data to analyze
            - 'auxiliary': auxiliary data dict for analyzer
            - 'sfile_keys': list of shot numbers to update in s-file

        Notes
        -----
        Concurrency backend follows ``run_analyze_image_asynchronously`` on
        the ImageAnalyzer instance.
        """
        logger.info(f"Starting analysis in {self.analysis_mode} mode")
        self.results: dict[int, ImageAnalyzerResult] = {}
        self._pending_aux_updates: list[dict] = []

        use_threads = self.image_analyzer.run_analyze_image_asynchronously
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        logger.info(
            f"Using {'ThreadPoolExecutor' if use_threads else 'ProcessPoolExecutor'}"
        )

        with Executor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.image_analyzer.analyze_image,
                    unit["image"],
                    unit["auxiliary"],
                ): (unit_key, unit["sfile_keys"])
                for unit_key, unit in analysis_units.items()
            }

            logger.info("Submitted analysis tasks.")

            for future in as_completed(futures):
                unit_key, sfile_keys = futures[future]
                try:
                    result: ImageAnalyzerResult = future.result()
                    analysis_results = result.scalars

                    if self._has_valid_result(result):
                        self.results[unit_key] = result
                        logger.info(f"Unit {unit_key}: valid data stored.")
                        logger.info(
                            f"Analyzed unit {unit_key} and got {analysis_results}"
                        )
                    else:
                        logger.info(
                            f"Unit {unit_key}: no valid data returned from analysis."
                        )

                    # Collect numeric scalars and update cached/queued s-file data
                    numeric_updates = {
                        key: value
                        for key, value in analysis_results.items()
                        if isinstance(value, (int, float, np.number))
                    }
                    non_numeric = set(analysis_results) - set(numeric_updates)
                    if non_numeric:
                        logger.warning(
                            f"[{self.__class__.__name__} using {self.image_analyzer.__class__.__name__}] "
                            f"Non-numeric scalar keys skipped: {sorted(non_numeric)}"
                        )

                    if numeric_updates:
                        for shot_num in sfile_keys:
                            # keep in-memory copy current for any downstream calc
                            for key, value in numeric_updates.items():
                                self.auxiliary_data.loc[
                                    self.auxiliary_data["Shotnumber"] == shot_num, key
                                ] = value
                            self._pending_aux_updates.append(
                                {"Shotnumber": shot_num, **numeric_updates}
                            )

                except Exception as e:
                    logger.error(f"Analysis failed for unit {unit_key}: {e}")

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

    def _log_provenance(
        self, columns_written: list[str], config: dict | None = None
    ) -> None:
        """Log provenance with comprehensive config automatically extracted.

        Overrides base class to automatically build a configuration dictionary
        that includes both the scan analyzer config (from factory) and the
        image analyzer config.

        Parameters
        ----------
        columns_written : list[str]
            List of column names that were written to the s-file.
        config : dict, optional
            Configuration dictionary. If None, will build a comprehensive
            config from self.analyzer_config (if available from factory) and
            self.image_analyzer.
        """
        if config is None:
            config = self._build_provenance_config()

        # Call the parent implementation with the extracted config
        super()._log_provenance(columns_written, config=config)

    def _build_provenance_config(self) -> dict | None:
        """Build a comprehensive provenance config dictionary.

        Combines scan analyzer config (from factory, if available) with
        image analyzer config extracted from the instance.

        Returns
        -------
        dict or None
            Comprehensive config dictionary containing:
            - scan_analyzer: class name, module, and config (if from factory)
            - image_analyzer: class name, module, and runtime config
        """
        config = {}

        # Add scan analyzer info
        scan_analyzer_info = {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
        }

        # If we have analyzer_config from factory, include it
        if hasattr(self, "analyzer_config") and self.analyzer_config is not None:
            # Pydantic model - convert to dict, excluding the nested image_analyzer
            # since we'll handle that separately with runtime values
            analyzer_dict = self.analyzer_config.model_dump(
                exclude={"image_analyzer"}, exclude_none=True
            )
            scan_analyzer_info["config"] = analyzer_dict

        config["scan_analyzer"] = scan_analyzer_info

        # Add image analyzer info
        if hasattr(self, "image_analyzer") and self.image_analyzer is not None:
            image_analyzer_info = {
                "class": self.image_analyzer.__class__.__name__,
                "module": self.image_analyzer.__class__.__module__,
            }

            # Extract runtime config from the image analyzer instance
            runtime_config = extract_config_from_analyzer(self.image_analyzer)
            if runtime_config:
                image_analyzer_info["config"] = runtime_config

            config["image_analyzer"] = image_analyzer_info

        return config if config else None

    @staticmethod
    def average_data(data_list: list[np.ndarray]) -> Optional[np.ndarray]:
        """Return the element-wise mean of a list of data arrays."""
        if len(data_list) == 0:
            return None

        return np.mean(data_list, axis=0)
