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
    from geecs_python_api.controls.api_defs import ScanTag
    from numpy.typing import NDArray

from pathlib import Path
import logging

from scan_analysis.analyzers.Undulator.array2D_scan_analysis import Array2DScanAnalysis

import traceback
PRINT_TRACEBACK = True


class HIMGWithAveraging(Array2DScanAnalysis):
    """
    A child class of Array2DScanAnalysis that overrides the noscan postprocessing.
    This version may, for example, save a custom average image and skip creating a GIF.
    """

    @staticmethod
    def process_shot_parallel(
            shot_num: int, file_path: Path, analyzer_class: type[BasicImageAnalyzer]
    ) -> tuple[int, Optional[np.ndarray], dict]:
        """
        Helper function for parallel processing in a separate process.
        Creates a new analyzer instance from analyzer_class, processes the image,
        and returns the shot number, processed image, and analysis results.

        If the analyzer's return value is not as expected (e.g., not a dict, missing
        keys, or values are None), it logs a warning and returns safe defaults.
        """
        try:
            analyzer = analyzer_class()
            results_dict = analyzer.analyze_image(file_path=file_path)
        except Exception as e:
            logging.error(f"Error during analysis for shot {shot_num}: {e}")
            return shot_num, None, {}

        if not isinstance(results_dict, dict):
            logging.warning(f"Analyzer returned non-dict result for shot {shot_num}.")
            return shot_num, None, {}

        if "analyzer_return_lineouts" not in results_dict:
            logging.warning(f"Shot {shot_num}: 'analyzer_return_lineouts' key not found in analyzer result.")
            image = None
        else:
            image = results_dict.get("analyzer_return_lineouts")
        print(np.nanmax(image))
        print(np.nanmin(image))

        analysis = results_dict.get("analyzer_return_dictionary", {})
        if not isinstance(analysis, dict):
            logging.warning(f"Shot {shot_num} analysis returned non-dict 'analyzer_return_dictionary'.")
            analysis = {}

        if image is None and analysis:
            logging.info(f"Shot {shot_num} returned no processed image, but analysis results are available.")
        elif image is None:
            logging.warning(f"Shot {shot_num} returned no processed image or analysis results.")
        print(np.nanmax(image))
        print(np.nanmin(image))


        return shot_num, image, analysis

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


