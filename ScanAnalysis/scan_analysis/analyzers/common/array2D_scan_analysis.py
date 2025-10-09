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
from typing import TYPE_CHECKING, Optional

# --- Local / Project Imports ---
from scan_analysis.analyzers.common.single_device_scan_analyzer import (
    SingleDeviceScanAnalyzer,
)


from scan_analysis.analyzers.renderers import Image2DRenderer
from image_analysis.base import ImageAnalyzer

# --- Type-Checking Imports ---
if TYPE_CHECKING:
    from geecs_data_utils import ScanTag

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
    ):
        """Initialize the analyzer with an ImageAnalyzer and Image2DRenderer."""
        if not device_name:
            raise ValueError("Array2DScanAnalyzer requires a device_name.")

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

        # Backward compatibility: map flag_save_images to flag_save_data
        self.flag_save_images = flag_save_images


if __name__ == "__main__":
    from scan_analysis.base import ScanAnalyzerInfo as Info
    from scan_analysis.execute_scan_analysis import instantiate_scan_analyzer
    from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
    from image_analysis.config_loader import set_config_base_dir
    from geecs_data_utils import ScanTag
    from pathlib import Path

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
