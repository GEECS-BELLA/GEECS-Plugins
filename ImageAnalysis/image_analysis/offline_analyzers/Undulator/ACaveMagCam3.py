"""Image analyzer for ACaveMagCam3 beamline camera.

This module defines `ACaveMagCam3ImageAnalyzer`, an `ImageAnalyzer` subclass for
extracting basic ROI statistics (average, total, maximum counts) from a fixed set
of rectangular regions in ACaveMagCam3 images.
"""

from __future__ import annotations

from typing import Union, Optional
from pathlib import Path

import numpy as np
from image_analysis.base import ImageAnalyzer


class ACaveMagCam3ImageAnalyzer(ImageAnalyzer):
    """Analyzer for ACaveMagCam3 images using fixed ROIs."""

    def __init__(self):
        """Initialize analyzer with synchronous execution and logging enabled."""
        self.run_analyze_image_asynchronously = False
        self.flag_logging = True
        super().__init__()

    @staticmethod
    def analyze_roi(image, roi):
        """Extract ROI subarray and compute mean, sum, and max counts."""
        x, y, w, h = roi
        roi_image = image[y : y + h, x : x + w]
        avg_count = np.mean(roi_image)
        total_count = np.sum(roi_image)
        max_count = np.max(roi_image)
        return avg_count, total_count, max_count, roi_image

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[dict] = None
    ) -> dict[str, Union[float, int, str, np.ndarray]]:
        """Analyze an ACaveMagCam3 image by computing fixed ROI statistics.

        Parameters
        ----------
        image : numpy.ndarray
            Input image array.
        auxiliary_data : dict, optional
            Additional analysis metadata (unused in this analyzer).

        Returns
        -------
        dict
            Standard `ImageAnalyzer` result dictionary containing:
            - ROI averages (float64)
            - ROI totals
            - ROI maxima
            - The last ROI image (if available)
        """
        rois = {
            "ACaveMagCam3 High Energy ROI": (665, 130, 110, 110),
            "ACaveMagCam3 Low Energy ROI": (815, 130, 110, 110),
            "ACaveMagCam3 Total ROI": (660, 125, 270, 120),
        }

        roi_analysis_result = {}
        roi_image = None

        for roi_name, roi in rois.items():
            avg, total, max_val, roi_image = self.analyze_roi(image, roi)
            roi_analysis_result[f"{roi_name}:avg"] = avg.astype(np.float64)
            roi_analysis_result[f"{roi_name}:total"] = total
            roi_analysis_result[f"{roi_name}:max"] = max_val

        if roi_image is None:
            return_dictionary = self.build_return_dictionary(
                return_scalars=roi_analysis_result
            )
        else:
            return_dictionary = self.build_return_dictionary(
                return_image=roi_image, return_scalars=roi_analysis_result
            )

        return return_dictionary


if __name__ == "__main__":
    image_analyzer = ACaveMagCam3ImageAnalyzer()
    file_path = Path(
        "Z:/data/Undulator/Y2025/03-Mar/25_0306/scans/Scan039/UC_ACaveMagCam3/Scan039_UC_ACaveMagCam3_001.png"
    )
    results = image_analyzer.analyze_image_file(image_filepath=file_path)
    print(results)
