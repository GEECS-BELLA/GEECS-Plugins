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
from typing import TYPE_CHECKING, Optional, Dict, Any

# --- Local / Project Imports ---
from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)
from scan_analysis.analyzers.renderers import Line1DRenderer
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
        )


if __name__ == "__main__":
    """Example usage of Array1DScanAnalyzer."""
    from scan_analysis.base import ScanAnalyzerInfo as Info
    from scan_analysis.execute_scan_analysis import instantiate_scan_analyzer
    from image_analysis.config_loader import set_config_base_dir
    from geecs_data_utils import ScanTag
    from pathlib import Path

    # Set up config directory
    current_dir = Path(__file__).resolve().parent.parent
    geecs_plugins_dir = current_dir.parent.parent.parent
    set_config_base_dir(geecs_plugins_dir / "image_analysis_configs")

    # Example device and config
    dev_name = "TekScope"
    config_dict = {"line_config_name": "Z_Test_Scope"}

    # Create analyzer info
    analyzer_info = Info(
        scan_analyzer_class=Array1DScanAnalyzer,
        requirements={dev_name},
        device_name=dev_name,
        scan_analyzer_kwargs={
            "image_analyzer": Standard1DAnalyzer(**config_dict),
            "file_tail": ".csv",
        },
    )

    # Run analysis on a test scan
    import time

    t0 = time.monotonic()
    test_tag = ScanTag(year=2025, month=1, day=15, number=1, experiment="HTU")
    scan_analyzer = instantiate_scan_analyzer(scan_analyzer_info=analyzer_info)
    scan_analyzer.run_analysis(scan_tag=test_tag)
    t1 = time.monotonic()
    logger.info(f"Execution time: {t1 - t0:.2f} seconds")
