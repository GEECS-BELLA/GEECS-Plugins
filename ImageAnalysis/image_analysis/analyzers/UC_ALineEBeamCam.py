from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Union, List
import numpy as np
from scipy.ndimage import median_filter
import time

if TYPE_CHECKING:
    from ..types import Array2D

from ..base import LabviewImageAnalyzer
from .online_analysis_modules import image_processing_funcs as process


class UC_ALineEBeamCamAnalyzer(LabviewImageAnalyzer):
    def __init__(self,
                 noise_threshold: int = 100,
                 roi: List[int] = [None, None, None, None],  # ROI(top, bottom, left, right)
                 circular_crop_center_x: int = 525,
                 circular_crop_center_y: int = 475,
                 circular_crop_radius: int = 475,
                 saturation_value: int = 4095,
                 spatial_calibration: float = 24.4, ):
        super().__init__()

        self.noise_threshold = noise_threshold
        self.roi = roi
        self.circular_crop_center_x = circular_crop_center_x
        self.circular_crop_center_y = circular_crop_center_y
        self.circular_crop_radius = circular_crop_radius
        self.saturation_value = saturation_value
        self.spatial_calibration = spatial_calibration

        self.do_print = False
        self.computational_clock_time = time.perf_counter()

    def circular_crop(self, image):
        if (self.circular_crop_radius > 0) and all(value is not None for value in (self.circular_crop_center_x,
                                                                                   self.circular_crop_center_y,
                                                                                   self.circular_crop_radius)):
            x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            r = np.sqrt((x - self.circular_crop_center_x) ** 2 + (y - self.circular_crop_center_y) ** 2)
            mask = r > self.circular_crop_radius

            cropped_image = np.copy(image)
            cropped_image[mask] = 0
        else:
            cropped_image = np.copy(image)
        return cropped_image

    def analyze_image(self, input_image: Array2D, auxiliary_data: Optional[dict] = None,
                      ) -> dict[str, Union[dict, np.ndarray]]:
        """
        roi_image = self.roi_image(input_image)
        filter_image = median_filter(roi_image, size=3)
        crop_image = self.circular_crop(filter_image)
        image = process.threshold_reduction(crop_image, threshold=self.noise_threshold)
        """
        image = np.copy(input_image)
        roi_image = np.copy(input_image)
        crop_image = np.copy(input_image)

        saturation_number = process.saturation_check(roi_image, saturation_value=self.saturation_value)
        peak_counts = np.max(crop_image)  # Before threshold subtraction
        total_counts = np.sum(image)  # After threshold subtraction

        if total_counts == 0:
            centroid_x = 0
            centroid_y = 0
            fwhm_x = 0
            fwhm_y = 0
            sigma_x = 0
            sigma_y = 0

        else:
            centroid_x, centroid_y = process.calculate_centroid_center_of_mass(image, total_counts)

            x_projection = np.sum(image, axis=0)
            y_projection = np.sum(image, axis=1)

            fwhm_x = process.calculate_fwhm(x_projection, threshold=self.noise_threshold)
            fwhm_y = process.calculate_fwhm(y_projection, threshold=self.noise_threshold)

            sigma_x = process.calculate_standard_deviation(x_projection, np.arange(len(x_projection)))
            sigma_y = process.calculate_standard_deviation(y_projection, np.arange(len(y_projection)))

        alinecam_dict = {
            "camera_saturation_counts": saturation_number,
            "camera_total_intensity_counts": total_counts,
            "peak_intensity_counts": peak_counts,
            "centroid_x_um": centroid_x * self.spatial_calibration,
            "centroid_y_um": centroid_y * self.spatial_calibration,
            "fwhm_x_um": fwhm_x * self.spatial_calibration,
            "fwhm_y_um": fwhm_y * self.spatial_calibration,
            "sigma_x_um": sigma_x * self.spatial_calibration,
            "sigma_y_um": sigma_y * self.spatial_calibration
        }
        uint_image = crop_image.astype(np.uint16)
        input_params = self.build_input_parameter_dictionary()

        return_dictionary = {
            "processed_image_uint16": uint_image,
            "analyzer_return_dictionary": alinecam_dict,
            "analyzer_return_lineouts": np.zeros((2, 2)),
            "analyzer_input_parameters": input_params
        }
        return return_dictionary

    def build_input_parameter_dictionary(self) -> dict:
        input_params = {
            "noise_threshold_int": self.noise_threshold,
            "roi_bounds_pixel": self.roi,
            "circular_crop_center_x_int": self.circular_crop_center_x,
            "circular_crop_center_y_int": self.circular_crop_center_y,
            "circular_crop_radius_int": self.circular_crop_radius,
            "saturation_value_int": self.saturation_value,
            "spatial_calibration_um/pix": self.spatial_calibration,
        }
        return input_params

    def print_time(self, label):
        if self.do_print:
            print(label, time.perf_counter() - self.computational_clock_time)
            self.computational_clock_time = time.perf_counter()
