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
from typing import TYPE_CHECKING, Optional, TypedDict, Dict

# --- Third-Party Libraries ---
import numpy as np
from collections import defaultdict

# --- Local / Project Imports ---
from scan_analysis.base import ScanAnalyzer
from scan_analysis.analyzers.renderers.config import RenderContext

# --- Type-Checking Imports ---
if TYPE_CHECKING:
    from image_analysis.types import AnalyzerResultDict
    from image_analysis.base import ImageAnalyzer
    from scan_analysis.analyzers.renderers import BaseRenderer

PRINT_TRACEBACK = True

logger = logging.getLogger(__name__)


# %% TypedDict Definitions
class BinDataEntry(TypedDict):
    """Container for per-bin aggregates."""

    value: float
    result: Optional[AnalyzerResultDict]


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
            # Run the data analyzer on every shot in parallel
            self._process_all_shots_parallel()

            # Depending on the scan type, perform additional processing
            if len(self.results) > 2:
                if self.noscan:
                    self._postprocess_noscan()
                else:
                    self._postprocess_scan_parallel()

            if not self.live_analysis:
                self.auxiliary_data.to_csv(
                    self.auxiliary_file_path, sep="\t", index=False
                )
            return self.renderer.display_contents

        except Exception as e:
            if PRINT_TRACEBACK:
                print(traceback.format_exc())
            logger.warning(f"Warning: Data analysis failed due to: {e}")
            return

    def _process_all_shots_parallel(self):
        """Load data files, run batch analysis (optional), then per-shot analysis."""
        self._load_all_data_parallel()
        self._run_batch_analysis()
        self._run_data_analysis_parallel()

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

    def _load_all_data_parallel(self) -> None:
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
            raise RuntimeError("No data loaded. Run _load_all_data_parallel first.")

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

    def _run_data_analysis_parallel(self) -> None:
        """
        Analyze each image in parallel (threaded or multi-processed).

        Uses the ImageAnalyzer's ``analyze_image`` method per shot, forwarding the
        file path and any batch-state via ``auxiliary_data``. Numeric scalars from
        the analyzer's return dictionary are persisted into the s-file for the shot.

        Notes
        -----
        Concurrency backend follows ``run_analyze_image_asynchronously`` on
        the ImageAnalyzer instance.
        """
        logger.info("Starting the individual image analysis")
        self.results: dict[int, AnalyzerResultDict] = {}

        use_threads = self.image_analyzer.run_analyze_image_asynchronously
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        logger.info(
            f"Using {'ThreadPoolExecutor' if use_threads else 'ProcessPoolExecutor'}"
        )

        with Executor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.image_analyzer.analyze_image,
                    data,
                    {"file_path": path, **self.stateful_results},
                ): shot_num
                for shot_num, (data, path) in self.raw_data.items()
            }

            logger.info("Submitted data analysis tasks.")

            for future in as_completed(futures):
                shot_num = futures[future]
                try:
                    result: AnalyzerResultDict = future.result()
                    processed_data = result.get("processed_image")
                    analysis_results = result.get("analyzer_return_dictionary", {})

                    if processed_data is not None:
                        self.results[shot_num] = result
                        logger.info(f"Shot {shot_num}: processed data stored.")
                        logger.info(
                            f"Analyzed shot {shot_num} and got {analysis_results}"
                        )
                    else:
                        logger.info(f"Shot {shot_num}: no data returned from analysis.")

                    for key, value in analysis_results.items():
                        if not isinstance(value, (int, float, np.number)):
                            logger.warning(
                                f"[{self.__class__.__name__} using {self.image_analyzer.__class__.__name__}] "
                                f"Analysis result for shot {shot_num} key '{key}' is not numeric (got {type(value).__name__}). Skipping."
                            )
                        else:
                            self.auxiliary_data.loc[
                                self.auxiliary_data["Shotnumber"] == shot_num, key
                            ] = value

                except Exception as e:
                    logger.error(f"Analysis failed for shot {shot_num}: {e}")

    def _postprocess_noscan(self) -> None:
        """Average over shots and create an animation when no parameter is scanned."""
        # Extract processed data from results dict
        data_list = [res["processed_image"] for res in self.results.values()]
        avg_data = self.average_data(data_list)

        if self.flag_save_data:
            # Average scalar results
            analysis_results = [
                res.get("analyzer_return_dictionary", {})
                for res in self.results.values()
            ]
            if analysis_results and analysis_results[0]:
                from collections import defaultdict

                sums = defaultdict(list)
                for d in analysis_results:
                    for k, v in d.items():
                        sums[k].append(v)
                avg_scalars = {k: np.mean(v, axis=0) for k, v in sums.items()}
            else:
                avg_scalars = {}

            # Create RenderContext for average
            avg_context = RenderContext(
                data=avg_data,
                input_parameters={"analyzer_return_dictionary": avg_scalars},
                device_name=self.device_name,
                identifier="average",
            )

            config = self._get_renderer_config()

            # Save average using new interface
            self.renderer.render_single(avg_context, config, self.path_dict["save"])

            # Create animation from all results
            contexts = [
                RenderContext(
                    data=result["processed_image"],
                    input_parameters=result.get("analyzer_input_parameters", {}),
                    device_name=self.device_name,
                    identifier=f"shot_{shot_num}",
                )
                for shot_num, result in self.results.items()
            ]
            animation_path = self.path_dict["save"] / "noscan.gif"
            self.renderer.render_animation(contexts, config, animation_path)

    def _postprocess_scan_parallel(self) -> None:
        """
        Post-process a scanned variable by binning and saving averaged data in parallel.

        Uses the new RenderContext-based renderer interface for clean, type-safe rendering.

        The method:
        1) Bins data using ``bin_data_from_results``.
        2) Creates RenderContext objects for each bin.
        3) Renders individual bins in parallel using renderer.render_single().
        4) Generates a summary figure if >1 bin using renderer.render_summary().
        """
        # Bin data
        binned_data = self.bin_data_from_results()
        if not binned_data:
            logger.warning("No binned data to postprocess")
            return

        # Build render contexts from binned data
        contexts = [
            RenderContext.from_bin_result(
                bin_key=bin_key,
                bin_entry=bin_entry,
                device_name=self.device_name,
                scan_parameter=self.scan_parameter,
            )
            for bin_key, bin_entry in binned_data.items()
        ]

        # Get renderer config (subclasses should provide this)
        config = self._get_renderer_config()

        # Render individual bins in parallel
        if self.flag_save_data:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.renderer.render_single,
                        ctx,
                        config,
                        self.path_dict["save"],
                    )
                    for ctx in contexts
                ]
                for future in as_completed(futures):
                    try:
                        paths = future.result()
                        # Track saved paths for first bin (for backward compatibility)
                        if paths and len(paths) > 0:
                            # Extract bin_key from the first path's filename
                            filename = paths[0].name
                            # Parse bin_key from filename like "device_0_processed.h5"
                            parts = filename.split("_")
                            if len(parts) >= 2 and parts[-2].isdigit():
                                bin_key = int(parts[-2])
                                self.saved_avg_data_paths[bin_key] = paths[0]
                    except Exception as e:
                        logger.error(f"Error rendering bin data: {e}")

        # Create summary figure if multiple bins exist
        if len(contexts) > 1 and self.flag_save_data:
            try:
                summary_path = self.renderer.render_summary(
                    contexts, config, self.path_dict["save"]
                )
                if summary_path:
                    logger.info(f"Created summary figure at {summary_path}")
            except Exception as e:
                logger.error(f"Error creating summary figure: {e}")

        self.binned_data = binned_data

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

    def bin_data_from_results(self) -> dict[int, BinDataEntry]:
        """
        Bin processed data by scan parameter and compute per-bin averages.

        For each unique "Bin #" value in the auxiliary s-file:
        - Select shots with valid processed data.
        - Average the data for that bin.
        - Average numeric analyzer scalar outputs across shots in the bin.
        - Optionally, average analyzer lineouts (if supplied as arrays).

        Returns
        -------
        dict[int, BinDataEntry]
            Mapping from bin number to a dictionary with:
            - ``"value"`` : representative scan parameter value for the bin (float)
            - ``"result"`` : AnalyzerResultDict with averaged data/scalars/lineout
        """
        if "Bin #" not in self.auxiliary_data.columns:
            logger.warning("Missing 'Bin #' column in auxiliary data.")
            return {}

        unique_bins = [int(b) for b in np.unique(self.auxiliary_data["Bin #"].values)]

        logger.info(f"Unique bins from auxiliary data: {unique_bins}")

        binned_data: dict[int, BinDataEntry] = {}

        for bin_val in unique_bins:
            # Get shot numbers in this bin
            bin_shots = self.auxiliary_data[self.auxiliary_data["Bin #"] == bin_val][
                "Shotnumber"
            ].values

            # Filter shot numbers that have valid results
            valid_shots = [
                sn
                for sn in bin_shots
                if sn in self.results
                and self.results[sn].get("processed_image") is not None
            ]

            if not valid_shots:
                logger.warning(f"No data found for bin {bin_val}.")
                continue

            # Collect data and scalar results
            data_list = [self.results[sn]["processed_image"] for sn in valid_shots]
            analysis_results = [
                self.results[sn].get("analyzer_return_dictionary", {})
                for sn in valid_shots
            ]
            lineouts = [
                self.results[sn].get("analyzer_return_lineouts") for sn in valid_shots
            ]
            # Extract the first entry in the input parameters
            input_params = self.results[valid_shots[0]].get(
                "analyzer_input_parameters", {}
            )

            avg_data = self.average_data(data_list)
            if avg_data is None:
                continue

            # Calculate the average value for each key in analysis_results dict
            sums = defaultdict(list)
            for d in analysis_results:
                for k, v in d.items():
                    sums[k].append(v)
            avg_vals = {k: np.mean(v, axis=0) for k, v in sums.items()}

            if lineouts:
                # Unzip into two lists: all x-lineouts, all y-lineouts
                try:
                    x_list = [lo[0] for lo in lineouts if lo is not None]
                    y_list = [lo[1] for lo in lineouts if lo is not None]

                    avg_x = (
                        np.mean(np.stack(x_list, axis=0), axis=0) if x_list else None
                    )
                    avg_y = (
                        np.mean(np.stack(y_list, axis=0), axis=0) if y_list else None
                    )

                    average_lineouts = [avg_x, avg_y]
                except IndexError:
                    logger.warning("Lineouts do not have expected shape for overlays")
                    average_lineouts = None
            else:
                average_lineouts = None

            # Get representative scan parameter value
            column_full_name, _ = self.find_scan_param_column()
            param_value = self.auxiliary_data.loc[
                self.auxiliary_data["Bin #"] == bin_val, column_full_name
            ].mean()

            binned_data[bin_val] = {
                "value": float(param_value),
                "result": {
                    "processed_image": avg_data,
                    "analyzer_return_dictionary": avg_vals,
                    "analyzer_input_parameters": input_params,
                    "analyzer_return_lineouts": average_lineouts,
                },
            }

        return binned_data

    @staticmethod
    def average_data(data_list: list[np.ndarray]) -> Optional[np.ndarray]:
        """Return the element-wise mean of a list of data arrays."""
        if len(data_list) == 0:
            return None

        return np.mean(data_list, axis=0)
