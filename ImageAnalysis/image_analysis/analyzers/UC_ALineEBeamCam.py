from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Union
import numpy as np
from scipy.ndimage import median_filter
import time

if TYPE_CHECKING:
    from ..types import Array2D

from ..base import LabviewImageAnalyzer
from .online_analysis_modules import image_processing_funcs as process

class UC_ALineEBeamCamAnalyzer(LabviewImageAnalyzer):
    def __init__(self, config_file=None, **kwargs):
        self.noise_threshold = None
        self.circular_crop_center_x = None
        self.circular_crop_center_y = None
        self.circular_crop_radius = None
        self.saturation_value = None
        self.spatial_calibration = None

        self.do_print = False
        self.computational_clock_time = time.perf_counter()

        super().__init__(config_file, **kwargs)

    def configure(self,
                  noise_threshold: int = 100,
                  circular_crop_center_x: int = 525,
                  circular_crop_center_y: int = 475,
                  circular_crop_radius: int = 475,
                  saturation_value: int = 4095,
                  spatial_calibration: float = 24.4,
                  ):
        self.noise_threshold = int(noise_threshold)
        self.circular_crop_center_x = int(circular_crop_center_x)
        self.circular_crop_center_y = int(circular_crop_center_y)
        self.circular_crop_radius = int(circular_crop_radius)
        self.saturation_value = int(saturation_value)
        self.spatial_calibration = float(spatial_calibration)

    def circular_crop(self, image):
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        r = np.sqrt((x - self.circular_crop_center_x) ** 2 + (y - self.circular_crop_center_y) ** 2)
        mask = r > self.circular_crop_radius

        cropped_image = np.copy(image)
        cropped_image[mask] = 0
        return cropped_image

    def analyze_image(self, input_image: Array2D, auxiliary_data: Optional[dict] = None,
                      ) -> dict[str, Union[dict, np.ndarray]]:
        roi_image = self.roi_image(input_image)
        filter_image = median_filter(roi_image, size=3)
        crop_image = self.circular_crop(filter_image)
        image = process.threshold_reduction(crop_image, threshold=self.noise_threshold)

        saturation_number = process.saturation_check(roi_image, saturation_value=self.saturation_value)
        peak_counts = np.max(crop_image)  # Before threshold subtraction
        total_counts = np.sum(image)  # After threshold subtraction

        if total_counts == 0:
            centroid_x = 0
            centroid_y = 0
            fwhm_x = 0
            fwhm_y = 0

        else:
            centroid_x = np.sum(np.arange(image.shape[1]) * image) / total_counts
            centroid_y = np.sum(np.arange(image.shape[0]) * np.transpose(image)) / total_counts

            x_projection = np.sum(image, axis=0)
            y_projection = np.sum(image, axis=1)

            fwhm_x = process.calculate_fwhm(x_projection, threshold=self.noise_threshold)
            fwhm_y = process.calculate_fwhm(y_projection, threshold=self.noise_threshold)

        alinecam_dict = {
            "camera_saturation_counts": saturation_number,
            "camera_total_intensity_counts": total_counts,
            "peak_intensity_counts": peak_counts,
            "centroid_x_um": centroid_x * self.spatial_calibration,
            "centroid_y_um": centroid_y * self.spatial_calibration,
            "fwhm_x_um": fwhm_x * self.spatial_calibration,
            "fwhm_y_um": fwhm_y * self.spatial_calibration
        }
        uint_image = crop_image.astype(np.uint16)
        input_params = self.build_input_parameter_dictionary()

        return_dictionary = {
            "processed_image_uint16": uint_image,
            "analyzer_return_dictionary": alinecam_dict,
            "analyzer_return_lineouts": np.zeros(1),
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
