"""
Wednesday 9-6-2023

Will eventually flesh this out more.  Just writing out what I want the analysis to do.  Mostly just rotate the image
and then do a projection.  Then turn this projection into a spectra.

@Chris
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Union, List
import numpy as np
import time

if TYPE_CHECKING:
    from ..types import Array2D

from ..base import LabviewImageAnalyzer
from .online_analysis_modules import photon_spectrometer_analyzer as analyze


class UC_LightSpectrometerCamAnalyzer(LabviewImageAnalyzer):
    def __init__(self, config_file=None, **kwargs):
        self.noise_threshold = None
        self.saturation_value = None
        self.calibration_image_tilt = None
        self.calibration_wavelength_pixel = None
        self.calibration_0th_order_pixel = None
        self.minimum_wavelength_analysis = None
        self.optimization_central_wavelength = None
        self.optimization_bandwidth_wavelength = None

        # Set do_print to True for debugging information
        self.do_print = False
        self.computational_clock_time = time.perf_counter()

        super().__init__(config_file, **kwargs)

    def configure(self,
                  noise_threshold: int = 50,
                  roi: List[int] = [None, None, None, None],  # ROI(top, bottom, left, right)
                  saturation_value: int = 4095,
                  calibration_image_tilt: float = 1.1253,
                  calibration_wavelength_pixel: float = 0.463658,
                  calibration_0th_order_pixel: float = 1186.24,
                  minimum_wavelength_analysis: float = 150.0,
                  optimization_central_wavelength: float = 420.0,
                  optimization_bandwidth_wavelength: float = 10.0
                  ):
        """
        Parameters
        ----------
        noise_threshold: int
            Large enough to remove noise level
        roi: List[int]
            The bounds by which the input image is cropped when the analyze_image function is called.  Given as a list
            of four integers, arranged as [top, bottom, left, right].
        saturation_value: int
            The minimum value by which a pixel can be considered as "Saturated." The default is 2^12 - 1
        calibration_image_tilt: float
            Tilt angle of the image in degrees
        calibration_wavelength_pixel: float
            Linear calibration of the wavelength per pixel in nm/pixel
        calibration_0th_order_pixel: float
            Calibrated position of the 0th order in pixel number
        minimum_wavelength_analysis: float
            Minimum wavelength in which to crop the spectrum, in nm
        optimization_central_wavelength: float
            For the XOpt algorithm, the central wavelength to use for the Gaussian weight function, in nm
        optimization_bandwidth_wavelength: float
            For the XOpt algorithm, the standard deviation from the central wavelength for the Gaussian weight function
        """
        if self.roi is None:
            self.roi = roi
        self.noise_threshold = int(noise_threshold)
        self.saturation_value = int(saturation_value)
        self.calibration_image_tilt = float(calibration_image_tilt)
        self.calibration_wavelength_pixel = float(calibration_wavelength_pixel)
        self.calibration_0th_order_pixel = float(calibration_0th_order_pixel)
        self.minimum_wavelength_analysis = float(minimum_wavelength_analysis)
        self.optimization_central_wavelength = float(optimization_central_wavelength)
        self.optimization_bandwidth_wavelength = float(optimization_bandwidth_wavelength)

    def analyze_image(self, input_image: Array2D, auxiliary_data: Optional[dict] = None,
                      ) -> dict[str, Union[dict, np.ndarray]]:

        processed_image = self.roi_image(input_image.astype(np.float32))

        saturation_number = analyze.saturation_check(processed_image, self.saturation_value)
        self.print_time(" Saturation Check:")

        image = analyze.threshold_reduction(processed_image, self.noise_threshold)
        self.print_time(" Threshold Subtraction")

        total_photons = np.sum(image)
        self.print_time(" Total Photon Counts")

        rotated_image = analyze.rotate_image(image, self.calibration_image_tilt)
        self.print_time(" Image Rotation")

        wavelength_array, spectrum_array = analyze.get_spectrum_lineouts(rotated_image,
                                                                         self.calibration_wavelength_pixel,
                                                                         self.calibration_0th_order_pixel)
        self.print_time(" Spectrum Lineouts")

        crop_wavelength_array, crop_spectrum_array = analyze.crop_spectrum(wavelength_array, spectrum_array,
                                                                           self.minimum_wavelength_analysis)

        if np.sum(crop_spectrum_array) == 0:
            peak_wavelength = 0
            average_wavelength = 0
            wavelength_spread = 0
            optimization_factor = 0

        else:
            peak_wavelength = analyze.get_peak_wavelength(crop_wavelength_array, crop_spectrum_array)
            average_wavelength = analyze.get_average_wavelength(crop_wavelength_array, crop_spectrum_array)
            wavelength_spread = analyze.get_wavelength_spread(crop_wavelength_array, crop_spectrum_array,
                                                              average_wavelength=average_wavelength)
            self.print_time(" Spectrum Stats")

            optimization_factor = analyze.calculate_optimization_factor(crop_wavelength_array, crop_spectrum_array,
                                                                        self.optimization_central_wavelength,
                                                                        self.optimization_bandwidth_wavelength)
            self.print_time(" Optimization Factor")

        exit_cam_dict = {
            "camera_saturation_counts": saturation_number,
            "camera_total_intensity_counts": total_photons,
            "peak_wavelength_nm": peak_wavelength,
            "average_wavelength_nm": average_wavelength,
            "wavelength_spread_weighted_rms_nm": wavelength_spread,
            "optimization_factor": optimization_factor,
        }
        uint_image = rotated_image.astype(np.uint16)
        input_params = self.build_input_parameter_dictionary()

        return_dictionary = {
            "processed_image_uint16": uint_image,
            "analyzer_return_dictionary": exit_cam_dict,
            "analyzer_return_lineouts": np.vstack((wavelength_array, spectrum_array)),
            "analyzer_input_parameters": input_params
        }
        return return_dictionary

    def build_input_parameter_dictionary(self) -> dict:
        input_params = {
            "noise_threshold_int": self.noise_threshold,
            "roi_bounds_pixel": self.roi,
            "saturation_value_int": self.saturation_value,
            "image_tilt_calibration_degrees": self.calibration_image_tilt,
            "wavelength_calibration_nm/pixel": self.calibration_wavelength_pixel,
            "zeroth_order_calibration_pixel": self.calibration_0th_order_pixel,
            "minimum_wavelength_nm": self.minimum_wavelength_analysis,
            "optimization_central_wavelength_nm": self.optimization_central_wavelength,
            "optimization_bandwidth_wavelength_nm": self.optimization_bandwidth_wavelength,
        }
        return input_params

    def print_time(self, label):
        if self.do_print:
            print(label, time.perf_counter() - self.computational_clock_time)
            self.computational_clock_time = time.perf_counter()
