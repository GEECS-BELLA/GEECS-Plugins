"""
General camera image analyzer.
Child to ScanAnalysis (./scan_analysis/base.py)
"""
# %% imports
from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional, List

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from geecs_paths_utils.scan_paths import ScanTag
    from numpy.typing import NDArray

from pathlib import Path
import logging

from scan_analysis.analyzers.common.array2D_scan_analysis import Array2DScanAnalysis

import traceback
PRINT_TRACEBACK = True


class HIMGWithAveraging(Array2DScanAnalysis):
    """
    A child class of Array2DScanAnalysis that overrides the noscan postprocessing.
    This version may, for example, save a custom average image and skip creating a GIF.
    """

    def _postprocess_noscan(self) -> None:
        """
        Custom post-processing for a no-scan.

        This method computes the average image from the processed images,
        saves a custom-named version of the average image (and optionally a normalized version),
        and updates the display contents.
        """
        # Compute the average image from the processed images.
        avg_image = np.mean(self.data['images'], axis=0)

        if avg_image is None:
            logging.warning("No images available to process in _postprocess_noscan.")
            return

        df = pd.DataFrame(avg_image)
        df.to_csv( self.path_dict['data_img'] / "average_phase.tsv", sep="\t", index=False, header=False)

    def _postprocess_scan_parallel(self) -> None:
        """
        Post-process a scanned variable by binning the images (from self.data) and then
        saving the resulting images in parallel.
        """
        pass

    def _postprocess_scan_interactive(self) -> None:
        """Perform post-processing for a scan: bin images and create an image grid."""
        pass


