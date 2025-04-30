from __future__ import annotations

from typing import Union, Optional, Any
from pathlib import Path

import numpy as np
from image_analysis.base import ImageAnalyzer

class ACaveMagCam3ImageAnalyzer(ImageAnalyzer):

    def __init__(self, config: Optional[Any] = None):
        """
        Parameters
        ----------

        """
        self.run_analyze_image_asynchronously = True
        self.flag_logging = True

        super().__init__(config=config)

    @staticmethod
    def analyze_roi(image, roi):
        x, y, w, h = roi
        roi_image = image[y:y + h, x:x + w]
        avg_count = np.mean(roi_image)
        total_count = np.sum(roi_image)
        max_count = np.max(roi_image)
        return avg_count, total_count, max_count, roi_image


    def analyze_image_file(self, image_filepath: Path, auxiliary_data: Optional[dict] = None) -> dict[
            str, Union[float, int, str, np.ndarray]]:

        """
        Analyze an image by simply loading it (if not already loaded) and returning it as the processed image.

        Parameters
        ----------
        image_filepath : Path, optional
            The path to the image file to load if image is None.
        auxiliary_data: dict, containing any additional imformation needed for analysis

        Returns
        -------
        dict
            A dictionary with the processed image and placeholder for analysis results.
        """

        image = self.load_image(image_filepath)

        # Define the ROIs (x, y, width, height) as a dictionary
        rois = {
            'ACaveMagCam3 High Energy ROI': (665, 130, 110, 110),
            'ACaveMagCam3 Low Energy ROI': (815, 130, 110, 110),
            'ACaveMagCam3 Total ROI': (660, 125, 270, 120)
        }

        # Start with a dictionary that includes the shot number
        roi_analysis_result = {}
        roi_image = None

        # Loop over each ROI and analyze it, then flatten the results into one dictionary
        for roi_name, roi in rois.items():
            avg, total, max_val, roi_image = self.analyze_roi(image, roi)
            roi_analysis_result[f'{roi_name}:avg'] = avg.astype(np.float64)
            roi_analysis_result[f'{roi_name}:total'] = total
            roi_analysis_result[f'{roi_name}:max'] = max_val

        # Append the flattened result dictionary to your list
        if roi_image is None:
            return_dictionary = self.build_return_dictionary(return_scalars=roi_analysis_result)
        else:
            return_dictionary = self.build_return_dictionary(return_image=roi_image, return_scalars=roi_analysis_result)

        return return_dictionary

if __name__ == "__main__":
    image_analyzer  = ACaveMagCam3ImageAnalyzer()
    file_path = Path('/Volumes/hdna2/data/Undulator/Y2025/03-Mar/25_0306/scans/Scan039/UC_ACaveMagCam3/Scan039_UC_ACaveMagCam3_001.png')
    print(file_path.exists())
    results = image_analyzer.analyze_image_file(image_filepath=file_path)
    print(results)
