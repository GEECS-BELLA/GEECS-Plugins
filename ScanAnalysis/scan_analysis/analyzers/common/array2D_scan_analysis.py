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
from typing import TYPE_CHECKING, Optional, Dict, Any

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
