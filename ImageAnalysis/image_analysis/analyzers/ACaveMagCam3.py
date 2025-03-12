from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union
from pathlib import Path

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..types import Array2D, QuantityArray2D

import logging
import numpy as np
from image_analysis.base import ImageAnalyzer
from image_analysis.utils import ROI, read_imaq_image

class ACaveMagCam3ImageAnalyzer(ImageAnalyzer):

    def __init__(self,
                 roi: ROI = ROI(top=None, bottom=None, left=None, right=None),
                 medium: str = 'plasma',
                 background_path: Path = None,
                 on_no_background: str = 'warn',

                 laser_wavelength: float = 800, #in nanmeter
                ):
        """
        Parameters
        ----------
        roi : ROI
            Region of interest, as top, bottom (where top < bottom), left, right.
        medium : str
            One of 'plasma', 'gas/He', 'gas/N', for calculating density from Abel-
            inverted wavefront.
        background_path : Path
            A file or folder containing interferograms to use as background.
        on_no_background : str
            What to do if no background is set explicitly and no background path is
            given.
                'raise': raise ValueError
                'warn': return wavefront with no background subtraction and issue warning
                'ignore': return wavefrtont with no background subtraction and don't
                          issue warning.

        laser_wavelength : [length] Quantity
            of imaging laser

        """

        self.roi = roi
        self.flag_logging = True

        super().__init__()

    def _log_info(self, message: str, *args, **kwargs):
        """Log an info message if logging is enabled."""
        if self.flag_logging:
            logging.info(message, *args, **kwargs)

    def _log_warning(self, message: str, *args, **kwargs):
        """Log a warning message if logging is enabled."""
        if self.flag_logging:
            logging.warning(message, *args, **kwargs)

    @staticmethod
    def analyze_roi(image, roi):
        x, y, w, h = roi
        roi_image = image[y:y + h, x:x + w]
        avg_count = np.mean(roi_image)
        total_count = np.sum(roi_image)
        max_count = np.max(roi_image)
        return avg_count, total_count, max_count, roi_image

    def analyze_image(self, image: NDArray = None, file_path: Path = None) -> dict[str, Union[float, int, str, np.ndarray]]:
        """
        Create phase map from a .himg or .has file.

        Parameters:
            image (NDArray): None. This part of the signature for the base class, but this analyzer
                requires loading of the image and processing with a third party SDK
            file_path: Path to the image file.

        Returns:
            A dictionary containing results (e.g., phase map and/or related parameters).

        Raises:
            ValueError: If the file type is not supported.
        """

        # Define the ROIs (x, y, width, height) as a dictionary
        rois = {
            'ACaveMagCam3 High Energy ROI': (665, 130, 110, 110),
            'ACaveMagCam3 Low Energy ROI': (815, 130, 110, 110),
            'ACaveMagCam3 Total ROI': (660, 125, 270, 120)
        }

        # Read the image once
        logging.info(f'file path passed to image analyzer is {file_path}')
        try:
            image = read_imaq_image(file_path) * 1.0
        except:
            logging.warning(f'file {file_path} does not exist')
            return {'processed_image':None, 'analysis_results': None}
        # Start with a dictionary that includes the shot number
        roi_analysis_result = {}

        # Loop over each ROI and analyze it, then flatten the results into one dictionary
        for roi_name, roi in rois.items():
            avg, total, max_val, roi_image = self.analyze_roi(image, roi)
            roi_analysis_result[f'{roi_name}:avg'] = avg
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