from __future__ import annotations

from typing import TYPE_CHECKING, Union
from pathlib import Path

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..types import Array2D

import numpy as np
from image_analysis.analyzers.basic_image_analysis import BasicImageAnalyzer

class ACaveMagCam3ImageAnalyzer(BasicImageAnalyzer):

    def __init__(self):
        """
        Parameters
        ----------

        """
        self.run_analyze_image_asynchronously = True
        self.flag_logging = True

        super().__init__()

    @staticmethod
    def analyze_roi(image, roi):
        x, y, w, h = roi
        roi_image = image[y:y + h, x:x + w]
        avg_count = np.mean(roi_image)
        total_count = np.sum(roi_image)
        max_count = np.max(roi_image)
        return avg_count, total_count, max_count, roi_image

    def analyze_image(
            self,
            image: Array2D = None,
            file_path: Path = None
    ) -> dict[str, Union[float | NDArray, dict[str, float]]]:
        """
        Analyze an image by simply loading it (if not already loaded) and returning it as the processed image.

        Parameters
        ----------
        image : Array2D, optional
            The image array to process. If not provided, it is loaded from file_path.
        file_path : Path, optional
            The path to the image file to load if image is None.

        Returns
        -------
        dict
            A dictionary with the processed image and placeholder for analysis results.
        """

        if image is None:
            if file_path is None:
                raise ValueError("Either an image or file_path must be provided.")
            image = self.load_image(file_path)

        # Define the ROIs (x, y, width, height) as a dictionary
        rois = {
            'ACaveMagCam3 High Energy ROI': (665, 130, 110, 110),
            'ACaveMagCam3 Low Energy ROI': (815, 130, 110, 110),
            'ACaveMagCam3 Total ROI': (660, 125, 270, 120)
        }

        # Start with a dictionary that includes the shot number
        roi_analysis_result = {}

        # Loop over each ROI and analyze it, then flatten the results into one dictionary
        for roi_name, roi in rois.items():
            avg, total, max_val, roi_image = self.analyze_roi(image, roi)
            roi_analysis_result[f'{roi_name}:avg'] = avg.astype(np.float64)
            roi_analysis_result[f'{roi_name}:total'] = total
            roi_analysis_result[f'{roi_name}:max'] = max_val

        # Append the flattened result dictionary to your list
        return {'processed_image':roi_image, 'analysis_results': roi_analysis_result}

if __name__ == "__main__":
    image_analyzer  = ACaveMagCam3ImageAnalyzer()
    file_path = Path('/Volumes/hdna2/data/Undulator/Y2025/03-Mar/25_0306/scans/Scan039/UC_ACaveMagCam3/Scan039_UC_ACaveMagCam3_001.png')
    print(file_path.exists())
    results = image_analyzer.analyze_image(file_path=file_path)
    print(results['processed_image'])