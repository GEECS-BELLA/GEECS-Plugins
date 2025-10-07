"""General-purpose scan analyzer for 2D array data.

This module provides :class:`Array2DScanAnalyzer`, a robust child of
:class:`scan_analysis.base.ScanAnalyzer` that orchestrates running an
:class:`image_analysis.base.ImageAnalyzer` over all shots in a scan, handling:

- Robust image discovery per shot via filename pattern matching.
- Parallelized loading and per-shot analysis (threaded vs. process pools).
- Optional batch-level preprocessing via the ImageAnalyzer API.
- Binning by scan parameter and per‑bin averaging of images and scalars.
- Turnkey post-processing outputs:
  - For "noscan": averaged image + animated GIF over shots.
  - For parameter scans: averaged image per bin + grid montage.
- Saving outputs (HDF5 for data, PNG for visualization) and updating the
  scan's auxiliary s-file with analyzer scalar results.

Notes
-----
- Concurrency backend is selected automatically based on whether the
  provided ImageAnalyzer instance is pickleable. Unpickleable analyzers
  (e.g., those holding active handles) trigger a fallback to threads by
  setting ``image_analyzer.run_analyze_image_asynchronously = True``.
- The analyzer expects the wrapped ImageAnalyzer to return results using
  :meth:`image_analysis.base.ImageAnalyzer.build_return_dictionary`.
- Binning requires a "Bin #" column in the auxiliary s-file.
"""

from __future__ import annotations

# --- Standard Library ---
import logging
import re
import traceback
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypedDict, Dict

# --- Third-Party Libraries ---
import numpy as np
import matplotlib
from collections import defaultdict


# --- Local / Project Imports ---
from scan_analysis.base import ScanAnalyzer
from scan_analysis.analyzers.renderers import Image2DRenderer
from image_analysis.base import ImageAnalyzer
from image_analysis.tools.rendering import base_render_image

# --- Type-Checking Imports ---
if TYPE_CHECKING:
    from geecs_data_utils import ScanTag
    from image_analysis.types import AnalyzerResultDict

# --- Global Config ---
use_interactive = False
if not use_interactive:
    matplotlib.use("Agg")


PRINT_TRACEBACK = True


# --- TypedDict Definitions ---
class BinImageEntry(TypedDict):
    """Container for per‑bin aggregates."""

    value: float
    result: Optional[AnalyzerResultDict]


logger = logging.getLogger(__name__)


# %% classes
class Array2DScanAnalyzer(ScanAnalyzer):
    """Scan analyzer for generic 2D array images.

    This class adapts any :class:`ImageAnalyzer` to run across a scan. It handles
    parallelized I/O and per‑shot analysis, optional batch preprocessing, binning
    and averaging, and post‑processing visualizations.

    The ImageAnalyzer must implement:

    - ``load_image(path) -> np.ndarray`` (or compatible Array2D)
    - ``analyze_image(image, auxiliary_data: dict|None) -> AnalyzerResultDict``
    - Optionally ``analyze_image_batch(images: list[np.ndarray]) -> (list[np.ndarray], dict)``
    - Optionally ``render_image(...) -> (Figure, Axes)`` (used for plots/GIFs)

    Parameters
    ----------
    device_name : str
        Device subfolder name used to locate images within a scan directory.
    image_analyzer : ImageAnalyzer, optional
        Analyzer instance to apply per image. If omitted, a default ``ImageAnalyzer()``
        is constructed.
    file_tail : str, optional
        Suffix/extension used to match image files (e.g., ".png", ".himg",
        "_postprocessed.tsv"). Only files ending with this literal tail are used.
    skip_plt_show : bool, default=True
        Passed to :class:`ScanAnalyzer` to control interactive plotting in parents.
    flag_save_images : bool, default=True
        If True, saves HDF5/PNG outputs to the analysis directory.

    Notes
    -----
    - If ``image_analyzer`` cannot be pickled (e.g., due to open handles), the analyzer
      automatically switches to threaded execution instead of multiprocessing.
    """

    def __init__(
        self,
        device_name: str,
        image_analyzer: Optional[ImageAnalyzer] = None,
        file_tail: Optional[str] = ".png",
        skip_plt_show: bool = True,
        flag_save_images: bool = True,
    ):
        """Initialize the analyzer and validate concurrency constraints."""
        if not device_name:
            raise ValueError("Array2DScanAnalyzer requires a device_name.")

        super().__init__(device_name=device_name, skip_plt_show=skip_plt_show)

        self.image_analyzer = image_analyzer or ImageAnalyzer()
        self.renderer = Image2DRenderer()

        self.max_workers = 16
        self.saved_avg_image_paths: Dict[int, Path] = {}

        # define flags
        self.flag_save_images = flag_save_images

        self.file_tail = file_tail

        try:
            pickle.dumps(self.image_analyzer)
        except (pickle.PicklingError, TypeError) as e:
            # Mark that we cannot send the ImageAnalyzer through multiprocessing,
            # so we’ll fall back to threading instead.
            self.image_analyzer.run_analyze_image_asynchronously = True
            logger.warning(
                f"[Array2DScanAnalyzer] ImageAnalyzer instance is not pickleable "
                f"(reason: {e}). Falling back to threaded analysis."
            )

    def _establish_additional_paths(self):
        """Compute input/output paths and validate data presence."""
        # Get the camera-specific name if available (includes suffix like "_variation")
        # This allows multiple analysis variants of the same camera to have separate output directories
        camera_name = (
            getattr(self.image_analyzer, "camera_name", None) or self.device_name
        )

        # organize various paths for location of saved data
        self.path_dict = {
            "data_img": Path(self.scan_directory) / f"{self.device_name}",
            "save": (
                self.scan_directory.parents[1]
                / "analysis"
                / self.scan_directory.name
                / f"{camera_name}"  # Use camera_name instead of device_name
                / "Array2DScanAnalyzer"
            ),
        }

        # Check if data directory exists and is not empty
        if not self.path_dict["data_img"].exists() or not any(
            self.path_dict["data_img"].iterdir()
        ):
            logger.warning(
                f"Data directory '{self.path_dict['data_img']}' does not exist or is empty. Skipping"
            )

        if self.path_dict["data_img"] is None or self.auxiliary_data is None:
            logger.info("Skipping analysis due to missing data or auxiliary file.")
            return

    def _run_analysis_core(self):
        """Main analysis pipeline executed by the orchestrator.

        Workflow
        --------
        1. Establish paths and create the output directory (if saving is enabled).
        2. Load all images in parallel (threaded/process).
        3. Optionally run batch-level preprocessing via ``analyze_image_batch``.
        4. Run per-shot ``analyze_image`` in parallel and collect results.
        5. Post-process:
           - If ``noscan``: average + GIF.
           - Else: per‑bin averaging + grid montage.
        6. Persist auxiliary s-file updates and return list of display artifacts.

        Returns
        -------
        list[str | pathlib.Path] or None
            Paths to generated artifacts to display (e.g., images/GIFs), or None on failure.
        """
        self._establish_additional_paths()

        if self.flag_save_images and not self.path_dict["save"].exists():
            self.path_dict["save"].mkdir(parents=True)

        try:
            # Run the image analyzer on every shot in parallel.
            self._process_all_shots_parallel()

            # Depending on the scan type, perform additional processing.
            # self.results is a dict that only gets updated if the ImageAnalyzer
            # returns an image.
            if len(self.results) > 2:
                if self.noscan:
                    self._postprocess_noscan()
                else:
                    if use_interactive:
                        self._postprocess_scan_interactive()
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
            logger.warning(f"Warning: Image analysis failed due to: {e}")
            return

    def _process_all_shots_parallel(self):
        """Load images, run batch analysis (optional), then per‑shot analysis."""
        self._load_all_images_parallel()
        self._run_batch_analysis()
        self._run_image_analysis_parallel()

    def _build_image_file_map(self) -> None:
        """
        Build a mapping from shot number to image file path using a flexible filename regex.

        Only files whose suffix + format matches ``file_tail`` exactly are included.

        Notes
        -----
        Expected filename pattern::

            Scan<scan_number>_<device_subject>_<shot_number><file_tail>

        Examples
        --------
        ``Scan012_UC_ALineEBeam3_005.png`` or ``Scan016_U_HasoLift_001_postprocessed.tsv``
        """
        self._image_file_map = {}

        logger.info(f"self.file_tail: {self.file_tail}")
        image_filename_regex = re.compile(
            r"Scan(?P<scan_number>\d{3,})_"  # scan number
            r"(?P<device_subject>.*?)_"  # non-greedy subject
            r"(?P<shot_number>\d{3,})"  # shot number
            + re.escape(self.file_tail)
            + r"$"  # literal suffix+format
        )

        logger.info("mapping matched files")
        for file in self.path_dict["data_img"].iterdir():
            if not file.is_file():
                continue

            m = image_filename_regex.match(file.name)
            if m:
                shot_num = int(m.group("shot_number"))
                if shot_num in self.auxiliary_data["Shotnumber"].values:
                    self._image_file_map[shot_num] = file
                    logger.info(f"Mapped file for shot {shot_num}: {file}")
            else:
                logger.debug(f"Filename {file.name} does not match expected pattern.")

        expected_shots = set(self.auxiliary_data["Shotnumber"].values)
        found_shots = set(self._image_file_map.keys())
        for m in sorted(expected_shots - found_shots):
            logger.warning(f"No file found for shot {m}")

    def _load_all_images_parallel(self) -> None:
        """
        Load all images in parallel (threaded or multi-processed) and store them in ``self.raw_images``.

        This identifies the image file path for each shot number using the configured filename
        pattern, then loads images via :meth:`ImageAnalyzer.load_image`.

        Concurrency is selected by the analyzer's ``run_analyze_image_asynchronously`` flag.

        Side Effects
        ------------
        Populates:

        - ``self._image_file_map`` : ``{shot_number: Path}``
        - ``self.raw_images`` : ``{shot_number: (np.ndarray, Path)}``
        """
        self.raw_images = {}
        self._build_image_file_map()

        use_threads = self.image_analyzer.run_analyze_image_asynchronously
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

        with Executor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.image_analyzer.load_image, path): shot_num
                for shot_num, path in self._image_file_map.items()
            }

            for future in as_completed(futures):
                shot_num = futures[future]
                try:
                    image = future.result()
                    if image is not None:
                        file_path = self._image_file_map[shot_num]
                        self.raw_images[shot_num] = (image, file_path)
                except Exception as e:
                    logger.error(f"Error loading image for shot {shot_num}: {e}")

    def _run_batch_analysis(self) -> None:
        """
        Optionally run batch-level preprocessing across all loaded images.

        Calls :meth:`ImageAnalyzer.analyze_image_batch` with the list of raw images.
        If it returns a list of processed images and a state dict, those processed
        images replace the originals for subsequent per‑shot analysis, and the
        state dict is forwarded via the auxiliary_data argument to each
        :meth:`ImageAnalyzer.analyze_image` call.

        Raises
        ------
        RuntimeError
            If no images have been loaded yet.
        ValueError
            If the returned number of processed images does not match the input.
        """
        if not hasattr(self, "raw_images") or not self.raw_images:
            raise RuntimeError("No images loaded. Run _load_all_images_parallel first.")

        # Resolve {scan_dir} placeholders in background config BEFORE batch processing
        self._resolve_background_paths()

        try:
            # Extract keys and separate image + path tuples
            shot_nums = list(self.raw_images.keys())
            image_path_tuples = list(self.raw_images.values())
            image_list = [img for img, _ in image_path_tuples]  # extract only images
            file_paths = [
                path for _, path in image_path_tuples
            ]  # Reconstruct raw_images keeping original paths

            # Run batch analysis. This returns the processed images which will get handled
            # by analyze_image. It also returns additional otherwise stateful config type
            # results that should be used by analyze_image, but which are not explicitly
            # available to the multiprocessing based instances of the analyzer. Thus, they need
            # to be passed with the aux data

            processed_images, stateful_results = (
                self.image_analyzer.analyze_image_batch(image_list)
            )
            self.stateful_results = stateful_results
            logger.info(
                "finished batch processing with %s stateful results", stateful_results
            )

            if processed_images is None:
                logger.warning("analyze_image_batch() returned None. Skipping.")
                self.raw_images = {}
                return

            if len(processed_images) != len(shot_nums):
                raise ValueError(
                    f"analyze_image_batch() returned {len(processed_images)} images, "
                    f"but {len(shot_nums)} were expected."
                )

            self.raw_images = dict(zip(shot_nums, zip(processed_images, file_paths)))

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

        scan_dir = self.path_dict["data_img"]

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

    def _run_image_analysis_parallel(self) -> None:
        """
        Analyze each image in parallel (threaded or multi-processed).

        Uses :meth:`ImageAnalyzer.analyze_image` per shot, forwarding the file path and
        any batch‑state via ``auxiliary_data``. Numeric scalars from the analyzer's
        return dictionary are persisted into the s-file for the shot.

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
                    img,
                    {"file_path": path, **self.stateful_results},
                ): shot_num
                for shot_num, (img, path) in self.raw_images.items()
            }

            logger.info("Submitted image analysis tasks.")

            for future in as_completed(futures):
                shot_num = futures[future]
                try:
                    result: AnalyzerResultDict = future.result()
                    processed_image = result.get("processed_image")
                    analysis_results = result.get("analyzer_return_dictionary", {})

                    if processed_image is not None:
                        self.results[shot_num] = result
                        logger.info(f"Shot {shot_num}: processed image stored.")
                        logger.info(
                            f"analyzed shot {shot_num} and got {analysis_results}"
                        )

                    else:
                        logger.info(
                            f"Shot {shot_num}: no image returned from analysis."
                        )

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
        """Average over shots and create a GIF when no parameter is scanned."""
        # Extract processed images and shot numbers from results dict
        images = [res["processed_image"] for res in self.results.values()]
        avg_image = self.average_images(images)

        if self.flag_save_images:
            # Save average image as HDF5
            self.renderer.save_data(
                avg_image,
                save_dir=self.path_dict["save"],
                save_name=f"{self.device_name}_average_processed.h5",
            )

            # Save normalized PNG
            save_name = f"{self.device_name}_average_processed_visual.png"
            self.renderer.save_visualization(
                avg_image,
                save_dir=self.path_dict["save"],
                save_name=save_name,
                label=save_name,
            )

            # Create and store GIF
            gif_path = self.path_dict["save"] / "noscan.gif"
            render_fn = getattr(self.image_analyzer, "render_image", base_render_image)
            self.renderer.create_animation(
                data_dict=self.results,
                output_file=gif_path,
                render_fn=render_fn,
            )

    def _save_bin_images(self, bin_key: int, processed_image: np.ndarray) -> None:
        """Save per‑bin averaged image in HDF5 (data) and PNG (visual) formats."""
        save_name_scaled = f"{self.device_name}_{bin_key}_processed.h5"
        save_name_normalized = f"{self.device_name}_{bin_key}_processed_visual.png"
        self.renderer.save_data(
            processed_image, save_dir=self.path_dict["save"], save_name=save_name_scaled
        )
        self.saved_avg_image_paths[bin_key] = self.path_dict["save"] / save_name_scaled

        self.renderer.save_visualization(
            processed_image,
            save_dir=self.path_dict["save"],
            save_name=save_name_normalized,
        )

        logger.info(
            f"Saved bin {bin_key} images: {save_name_scaled} and {save_name_normalized}"
        )

    def _postprocess_scan_parallel(self) -> None:
        """
        Post-process a scanned variable by binning and saving averaged images in parallel.

        The method:
        1) Bins images using ``bin_images_from_data``.
        2) Saves each bin’s HDF5 and PNG concurrently.
        3) Generates a grid montage of binned averages if >1 bin.
        """
        # Use your existing parallel (or sequential) binning method to create binned_data.
        binned_data = self.bin_images_from_data(flag_save=False)

        # Save each bin's images concurrently using a thread pool.
        if self.flag_save_images:
            with ThreadPoolExecutor() as executor:
                futures = []
                for bin_key, bin_item in binned_data.items():
                    processed_image = bin_item["result"]["processed_image"]
                    futures.append(
                        executor.submit(self._save_bin_images, bin_key, processed_image)
                    )
                for future in as_completed(futures):
                    # Optionally handle exceptions:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error saving images for a bin: {e}")

        # Create an image grid if more than one bin exists.
        if len(binned_data) > 1 and self.flag_save_images:
            plot_scale = (getattr(self, "camera_analysis_settings", {}) or {}).get(
                "Plot Scale", None
            )
            save_path = (
                Path(self.path_dict["save"])
                / f"{self.device_name}_averaged_image_grid.png"
            )
            render_fn = getattr(self.image_analyzer, "render_image", base_render_image)
            self.renderer.create_summary_figure(
                binned_data,
                plot_scale=plot_scale,
                save_path=save_path,
                device_name=self.device_name,
                scan_parameter=self.scan_parameter,
                render_fn=render_fn,
            )
        self.binned_data = binned_data

    def _postprocess_scan_interactive(self) -> None:
        """Post-process a scanned variable (sequential save; useful for interactive runs)."""
        # Bin images from the already processed data.
        binned_data = self.bin_images_from_data(flag_save=False)

        # Process each bin sequentially.
        for bin_key, bin_item in binned_data.items():
            processed_image = bin_item["result"].get("processed_image")
            if self.flag_save_images and processed_image is not None:
                self._save_bin_images(bin_key, processed_image)
            elif processed_image is None:
                logger.warning(
                    f"Bin {bin_key} has no processed image; skipping saving for this bin."
                )

        # If more than one bin exists, create an image grid.
        if len(binned_data) > 1 and self.flag_save_images:
            plot_scale = (getattr(self, "camera_analysis_settings", {}) or {}).get(
                "Plot Scale", None
            )
            save_path = (
                Path(self.path_dict["save"])
                / f"{self.device_name}_averaged_image_grid.png"
            )
            render_fn = getattr(self.image_analyzer, "render_image", base_render_image)
            self.renderer.create_summary_figure(
                binned_data,
                plot_scale=plot_scale,
                save_path=save_path,
                device_name=self.device_name,
                scan_parameter=self.scan_parameter,
                render_fn=render_fn,
            )

        self.binned_data = binned_data

    def bin_images_from_data(
        self, flag_save: Optional[bool] = None
    ) -> dict[int, BinImageEntry]:
        """
        Bin processed images by scan parameter and compute per‑bin averages.

        For each unique "Bin #" value in the auxiliary s-file:
        - Select shots with valid processed images.
        - Average the images for that bin.
        - Average numeric analyzer scalar outputs across shots in the bin.
        - Optionally, average analyzer lineouts (if supplied as arrays).
        - Save per‑bin averaged images if requested.

        Parameters
        ----------
        flag_save : bool, optional
            Whether to save the averaged image per bin. Defaults to the instance's
            ``flag_save_images``.

        Returns
        -------
        dict[int, BinImageEntry]
            Mapping from bin number to a dictionary with:
            - ``"value"`` : representative scan parameter value for the bin (float)
            - ``"result"`` : AnalyzerResultDict with averaged image/scalars/lineout
        """
        if flag_save is None:
            flag_save = self.flag_save_images

        if "Bin #" not in self.auxiliary_data.columns:
            logger.warning("Missing 'Bin #' column in auxiliary data.")
            return {}

        unique_bins = [int(b) for b in np.unique(self.auxiliary_data["Bin #"].values)]

        logger.info(f"Unique bins from auxiliary data: {unique_bins}")

        binned_data: dict[int, BinImageEntry] = {}

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
                logger.warning(f"No images found for bin {bin_val}.")
                continue

            # Collect images and scalar results
            images = [self.results[sn]["processed_image"] for sn in valid_shots]
            analysis_results = [
                self.results[sn].get("analyzer_return_dictionary", {})
                for sn in valid_shots
            ]
            lineouts = [
                self.results[sn].get("analyzer_return_lineouts") for sn in valid_shots
            ]
            # just extract the first entry in the input parameters, as it isn't expected to change
            input_params = self.results[valid_shots[0]].get(
                "analyzer_input_parameters", {}
            )

            avg_image = self.average_images(images)
            if avg_image is None:
                continue

            # calculate the average value for each key,item in analysis_results dict for the bin
            sums = defaultdict(list)
            for d in analysis_results:
                for k, v in d.items():
                    sums[k].append(v)
            avg_vals = {k: np.mean(v, axis=0) for k, v in sums.items()}

            if lineouts:
                # unzip into two lists: all x-lineouts, all y-lineouts
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
                    logger.warning(
                        "lineouts do not have expected shape for image overlays"
                    )
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
                    "processed_image": avg_image,
                    "analyzer_return_dictionary": avg_vals,
                    "analyzer_input_parameters": input_params,
                    "analyzer_return_lineouts": average_lineouts,
                },
            }

            if flag_save:
                save_name = f"{self.device_name}_{bin_val}.h5"
                self.renderer.save_data(
                    avg_image, save_dir=self.path_dict["save"], save_name=save_name
                )

                logger.info(
                    f"Binned and averaged images for bin {bin_val} saved as {save_name}."
                )

        return binned_data

    @staticmethod
    def average_images(images: list[np.ndarray]) -> Optional[np.ndarray]:
        """Return the pixelwise mean of a list of images."""
        if len(images) == 0:
            return None

        return np.mean(images, axis=0)


if __name__ == "__main__":
    from scan_analysis.base import ScanAnalyzerInfo as Info
    from scan_analysis.execute_scan_analysis import instantiate_scan_analyzer
    from image_analysis.offline_analyzers.beam_analyzer import (
        BeamAnalyzer,
    )
    from image_analysis.config_loader import set_config_base_dir
    from geecs_data_utils import ScanTag

    current_dir = Path(__file__).resolve().parent.parent
    geecs_plugins_dir = current_dir.parent.parent.parent
    set_config_base_dir(geecs_plugins_dir / "image_analysis_configs")

    dev_name = "UC_ALineEBeam3"
    config_dict = {"camera_config_name": dev_name}
    analyzer_info = Info(
        scan_analyzer_class=Array2DScanAnalyzer,
        requirements={dev_name},
        device_name=dev_name,
        scan_analyzer_kwargs={"image_analyzer": BeamAnalyzer(**config_dict)},
    )

    import time

    t0 = time.monotonic()
    test_tag = ScanTag(year=2025, month=6, day=10, number=29, experiment="Undulator")
    scan_analyzer = instantiate_scan_analyzer(scan_analyzer_info=analyzer_info)
    scan_analyzer.run_analysis(scan_tag=test_tag)
    t1 = time.monotonic()
    logger.info(f"execution time: {t1 - t0}")
