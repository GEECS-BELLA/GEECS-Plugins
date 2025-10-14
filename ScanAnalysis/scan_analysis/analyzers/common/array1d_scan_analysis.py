"""Scan analyzer for 1D array (line/spectrum) data.

This module provides :class:`Array1DScanAnalyzer`, a specialized scan analyzer
for processing 1D line/spectrum data across all shots in a scan. It inherits from
:class:`SingleDeviceScanAnalyzer` and adds 1D-specific rendering via
:class:`Line1DRenderer`.

The analyzer handles:
- Robust data file discovery per shot via filename pattern matching
- Parallelized loading and per-shot analysis (threaded vs. process pools)
- Optional batch-level preprocessing via the ImageAnalyzer API
- Binning by scan parameter and per-bin averaging of 1D data and scalars
- Turnkey post-processing outputs:
  - For "noscan": averaged spectrum + animated GIF over shots
  - For parameter scans: averaged spectrum per bin + waterfall heatmap
- Saving outputs (CSV for data, PNG for visualization) and updating the
  scan's auxiliary s-file with analyzer scalar results
"""

from __future__ import annotations

# --- Standard Library ---
import logging
from typing import TYPE_CHECKING, Optional, Dict, Any, Literal

# --- Local / Project Imports ---
from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)
from scan_analysis.analyzers.renderers import Line1DRenderer, Line1DRendererConfig
from image_analysis.offline_analyzers.standard_1d_analyzer import Standard1DAnalyzer

# --- Type-Checking Imports ---
if TYPE_CHECKING:
    from image_analysis.base import ImageAnalyzer

logger = logging.getLogger(__name__)


# %% classes
class Array1DScanAnalyzer(SingleDeviceScanAnalyzer):
    """
    Scan analyzer for 1D line/spectrum data.

    This class adapts any 1D :class:`ImageAnalyzer` to run across a scan. It handles
    parallelized I/O and per-shot analysis, optional batch preprocessing, binning
    and averaging, and post-processing visualizations using :class:`Line1DRenderer`.

    The ImageAnalyzer must implement:

    - ``load_image(path) -> np.ndarray`` returning Nx2 array (x, y pairs)
    - ``analyze_image(image, auxiliary_data: dict|None) -> AnalyzerResultDict``
    - Optionally ``analyze_image_batch(images: list[np.ndarray]) -> (list[np.ndarray], dict)``

    Parameters
    ----------
    device_name : str
        Device subfolder name used to locate data files within a scan directory.
    image_analyzer : ImageAnalyzer, optional
        Analyzer instance to apply per spectrum/line. If omitted, a default
        ``Standard1DAnalyzer()`` is constructed.
    file_tail : str, optional
        Suffix/extension used to match data files (e.g., ".csv", ".txt", ".tsv").
        Only files ending with this literal tail are used.
    skip_plt_show : bool, default=True
        Passed to :class:`ScanAnalyzer` to control interactive plotting in parents.
    flag_save_data : bool, default=True
        If True, saves CSV/PNG outputs to the analysis directory.

    Notes
    -----
    - If ``image_analyzer`` cannot be pickled (e.g., due to open handles), the analyzer
      automatically switches to threaded execution instead of multiprocessing.
    - This class is a thin wrapper around :class:`SingleDeviceScanAnalyzer` that
      provides the :class:`Line1DRenderer` for 1D-specific visualization.
    - All 1D data is expected to be in Nx2 format where column 0 is x-values and
      column 1 is y-values.

    Examples
    --------
    Basic usage with default analyzer::

        analyzer = Array1DScanAnalyzer(
            device_name="TekScope",
            file_tail=".csv"
        )
        analyzer.run_analysis(scan_tag=my_scan_tag)

    With custom analyzer::

        from image_analysis.offline_analyzers.standard_1d_analyzer import Standard1DAnalyzer

        custom_analyzer = Standard1DAnalyzer(line_config_name="my_scope_config")
        analyzer = Array1DScanAnalyzer(
            device_name="TekScope",
            image_analyzer=custom_analyzer,
            file_tail=".csv"
        )
        analyzer.run_analysis(scan_tag=my_scan_tag)
    """

    def __init__(
        self,
        device_name: str,
        image_analyzer: Optional[ImageAnalyzer] = None,
        file_tail: Optional[str] = ".csv",
        skip_plt_show: bool = True,
        flag_save_data: bool = True,
        renderer_kwargs: Optional[Dict[str, Any]] = None,
        analysis_mode: Literal["per_shot", "per_bin"] = "per_shot",
    ):
        """Initialize the analyzer with an ImageAnalyzer and Line1DRenderer.

        Parameters
        ----------
        renderer_kwargs : dict, optional
            Additional keyword arguments to pass to the renderer's create_summary_figure method.
            Useful options include:

            - ``colormap_mode`` : str, default="sequential"
                Colormap normalization mode:

                - "sequential": Standard 0 to max (default, uses 'plasma')
                - "diverging": Symmetric around zero for bipolar data (uses 'RdBu_r')
                - "custom": User-defined vmin/vmax and cmap

            - ``cmap`` : str, optional
                Matplotlib colormap name (e.g., 'plasma', 'RdBu_r', 'coolwarm')
            - ``mode`` : str, default="waterfall"
                Visualization mode: "waterfall", "overlay", or "grid"
            - ``vmin``, ``vmax`` : float, optional
                Custom colormap limits (only used with colormap_mode="custom")

        Examples
        --------
        For scope traces with bipolar signals::

            analyzer = Array1DScanAnalyzer(
                device_name="TekScope",
                renderer_kwargs={"colormap_mode": "diverging"}
            )

        For custom colormap and limits::

            analyzer = Array1DScanAnalyzer(
                device_name="MyDevice",
                renderer_kwargs={
                    "colormap_mode": "custom",
                    "cmap": "coolwarm",
                    "vmin": -10,
                    "vmax": 10
                }
            )
        """
        if not device_name:
            raise ValueError("Array1DScanAnalyzer requires a device_name.")

        # Store renderer kwargs for later use
        self.renderer_kwargs = renderer_kwargs or {}

        # Create image analyzer if not provided
        image_analyzer = image_analyzer or Standard1DAnalyzer()

        # Create 1D renderer
        renderer = Line1DRenderer()

        # Initialize base class with the analyzer and renderer
        super().__init__(
            device_name=device_name,
            image_analyzer=image_analyzer,
            renderer=renderer,
            file_tail=file_tail,
            skip_plt_show=skip_plt_show,
            flag_save_data=flag_save_data,
            analysis_mode=analysis_mode,
        )

    def _get_renderer_config(self):
        """
        Get Line1DRendererConfig for this analyzer.

        Returns
        -------
        Line1DRendererConfig
            Renderer configuration with 1D-specific settings
        """
        from scan_analysis.analyzers.renderers.config import Line1DRendererConfig

        # Create config from renderer_kwargs
        try:
            return Line1DRendererConfig(**self.renderer_kwargs)
        except Exception as e:
            logger.warning(
                f"Error creating Line1DRendererConfig from {self.renderer_kwargs}: {e}. "
                f"Using defaults."
            )
            return Line1DRendererConfig()

    def _postprocess_noscan(self) -> None:
        """
        Post-process noscan 1D data: create average with std dev + waterfall plot.

        For 1D line/spectrum data without a scanned parameter, this creates:
        - An averaged spectrum (for now, just the average line)
        - A waterfall plot showing all shots in chronological order

        TODO: Add standard deviation visualization to the average plot
        """
        from scan_analysis.analyzers.renderers.config import RenderContext
        from collections import defaultdict
        import numpy as np

        # Extract data from lineouts (1D analyzers store data there)
        data_list = []
        for res in self.results.values():
            lineouts = res.get("analyzer_return_lineouts")
            if lineouts:
                # Reconstruct Nx2 array from lineouts [x_array, y_array]
                data = np.column_stack([lineouts[0], lineouts[1]])
                data_list.append(data)

        avg_data = self.average_data(data_list)

        if self.flag_save_data:
            # Average scalar results
            analysis_results = [
                res.get("analyzer_return_dictionary", {})
                for res in self.results.values()
            ]
            if analysis_results and analysis_results[0]:
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

            # Save average using renderer
            self.renderer.render_single(avg_context, config, self.path_dict["save"])

            # Create waterfall plot from all results (chronological order)
            contexts = []
            for shot_num, result in sorted(self.results.items()):
                lineouts = result.get("analyzer_return_lineouts")
                if lineouts:
                    # Reconstruct Nx2 array from lineouts
                    data = np.column_stack([lineouts[0], lineouts[1]])
                    contexts.append(
                        RenderContext(
                            data=data,
                            input_parameters=result.get(
                                "analyzer_input_parameters", {}
                            ),
                            device_name=self.device_name,
                            identifier=shot_num,
                            parameter_value=float(
                                shot_num
                            ),  # Use shot number as y-axis
                            scan_parameter="Shot Number",
                        )
                    )

            # Use waterfall mode for summary
            waterfall_config = Line1DRendererConfig(
                **{**self.renderer_kwargs, "mode": "waterfall"}
            )
            # Save to parent directory and let render_summary create the file
            self.renderer.render_summary(
                contexts, waterfall_config, self.path_dict["save"]
            )

    def _postprocess_scan(self) -> None:
        """
        Post-process scanned 1D data: create waterfall plot from binned data.

        For 1D line/spectrum data with a scanned parameter, this creates:
        - Individual averaged spectra per bin
        - A waterfall plot showing all bins sorted by scan parameter
        """
        from scan_analysis.analyzers.renderers.config import RenderContext
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Get binned data (handles both per_shot and per_bin modes)
        binned_data = self.get_binned_data()
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

        # Get renderer config with waterfall mode
        config = self._get_renderer_config()
        # Ensure waterfall mode for scans
        waterfall_config = Line1DRendererConfig(
            **{**self.renderer_kwargs, "mode": "waterfall"}
        )

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

        # Create waterfall summary figure if multiple bins exist
        if len(contexts) > 1 and self.flag_save_data:
            try:
                summary_path = self.renderer.render_summary(
                    contexts, waterfall_config, self.path_dict["save"]
                )
                if summary_path:
                    logger.info(f"Created waterfall summary at {summary_path}")
            except Exception as e:
                logger.error(f"Error creating waterfall summary: {e}")

        self.binned_data = binned_data
