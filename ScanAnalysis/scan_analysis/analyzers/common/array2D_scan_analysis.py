"""Scan analyzer for 2D array (image) data.

This module provides :class:`Array2DScanAnalyzer`, a specialized scan analyzer
for processing 2D image data across all shots in a scan. It inherits from
:class:`SingleDeviceScanAnalyzer` and adds 2D-specific rendering via
:class:`Image2DRenderer`.

The analyzer handles:
- Robust image discovery per shot via filename pattern matching
- Parallelized loading and per-shot analysis (threaded vs. process pools)
- Optional batch-level preprocessing via the ImageAnalyzer API
- Binning by scan parameter and per-bin averaging of images and scalars
- Turnkey post-processing outputs:
  - For "noscan": averaged image + animated GIF over shots
  - For parameter scans: averaged image per bin + grid montage
- Saving outputs (HDF5 for data, PNG for visualization) and updating the
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


from scan_analysis.analyzers.renderers import Image2DRenderer
from image_analysis.base import ImageAnalyzer

# --- Type-Checking Imports ---
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# %% classes
class Array2DScanAnalyzer(SingleDeviceScanAnalyzer):
    """
    Scan analyzer for generic 2D array images.

    This class adapts any :class:`ImageAnalyzer` to run across a scan. It handles
    parallelized I/O and per-shot analysis, optional batch preprocessing, binning
    and averaging, and post-processing visualizations using :class:`Image2DRenderer`.

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
    - This class is a thin wrapper around :class:`SingleDeviceScanAnalyzer` that
      provides the :class:`Image2DRenderer` for 2D-specific visualization.
    """

    def __init__(
        self,
        device_name: str,
        image_analyzer: Optional[ImageAnalyzer] = None,
        file_tail: Optional[str] = ".png",
        skip_plt_show: bool = True,
        flag_save_images: bool = True,
        renderer_kwargs: Optional[Dict[str, Any]] = None,
        analysis_mode: Literal["per_shot", "per_bin"] = "per_shot",
    ):
        """Initialize the analyzer with an ImageAnalyzer and Image2DRenderer.

        Parameters
        ----------
        renderer_kwargs : dict, optional
            Additional keyword arguments to pass to the renderer's methods.
            Useful options include:

            - ``colormap_mode`` : str, default="sequential"
                Colormap normalization mode:

                - "sequential": Standard 0 to max (default, uses 'plasma')
                - "diverging": Symmetric around zero for bipolar data (uses 'RdBu_r')
                - "custom": User-defined vmin/vmax and cmap

            - ``cmap`` : str, optional
                Matplotlib colormap name (e.g., 'plasma', 'RdBu_r', 'coolwarm')
            - ``vmax`` : float, optional
                Maximum value for colormap (replaces legacy 'plot_scale')
            - ``vmin`` : float, optional
                Minimum value for colormap
            - ``figsize`` : tuple, optional
                Panel width and height in inches for grid montages
            - ``figsize_inches`` : float, optional
                Width/height for square animation frames

        """
        if not device_name:
            raise ValueError("Array2DScanAnalyzer requires a device_name.")

        # Store renderer kwargs for later use
        self.renderer_kwargs = renderer_kwargs or {}

        # Create image analyzer if not provided
        image_analyzer = image_analyzer or ImageAnalyzer()

        # Create 2D renderer
        renderer = Image2DRenderer()

        # Initialize base class with the analyzer and renderer
        super().__init__(
            device_name=device_name,
            image_analyzer=image_analyzer,
            renderer=renderer,
            file_tail=file_tail,
            skip_plt_show=skip_plt_show,
            flag_save_data=flag_save_images,
            analysis_mode=analysis_mode,
        )

    def _get_renderer_config(self):
        """
        Get Image2DRendererConfig for this analyzer.

        Returns
        -------
        Image2DRendererConfig
            Renderer configuration with 2D-specific settings
        """
        from scan_analysis.analyzers.renderers.config import Image2DRendererConfig

        # Get renderer_kwargs if available
        renderer_kwargs = getattr(self, "renderer_kwargs", {})

        # Handle legacy plot_scale parameter from camera_analysis_settings
        plot_scale = (getattr(self, "camera_analysis_settings", {}) or {}).get(
            "Plot Scale", None
        )
        if plot_scale is not None and "vmax" not in renderer_kwargs:
            renderer_kwargs["vmax"] = plot_scale

        # Create config from kwargs
        try:
            return Image2DRendererConfig(**renderer_kwargs)
        except Exception as e:
            logger.warning(
                f"Error creating Image2DRendererConfig from {renderer_kwargs}: {e}. "
                f"Using defaults."
            )
            return Image2DRendererConfig()

    def _postprocess_noscan(self) -> None:
        """
        Post-process noscan 2D data: create average image + animated GIF.

        For 2D image data without a scanned parameter, this creates:
        - An averaged image across all shots
        - An animated GIF showing temporal evolution
        """
        from scan_analysis.analyzers.renderers.config import RenderContext
        from collections import defaultdict
        import numpy as np

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

            # Create animation from all results
            contexts = [
                RenderContext.from_analyzer_result(
                    shot_number=shot_num,
                    result=result,
                    device_name=self.device_name,
                )
                for shot_num, result in self.results.items()
            ]
            animation_path = self.path_dict["save"] / "noscan.gif"
            self.renderer.render_animation(contexts, config, animation_path)

    def _postprocess_scan(self) -> None:
        """
        Post-process scanned 2D data: create image grid from binned data.

        For 2D image data with a scanned parameter, this creates:
        - Individual averaged images per bin
        - A grid montage showing all bins
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

        # Get renderer config
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

        # Create summary figure (grid montage) if multiple bins exist
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
