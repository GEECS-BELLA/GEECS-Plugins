"""
Wednesday 9-6-2023

Will eventually flesh this out more.  Just writing out what I want the analysis to do.  Mostly just rotate the image
and then do a projection.  Then turn this projection into a spectra.

@Chris
"""
from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING, Union
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..types import Array2D

from ..base import ImageAnalyzer
from .online_analysis_modules import photon_spectrometer_analyzer as analyze


class UC_UndulatorExitCam(ImageAnalyzer):
    def __init__(self,
                 noise_threshold: int = 50,
                 edge_pixel_crop: int = 0,
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
        edge_pixel_crop: int
            Number of edge pixels to crop
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
        super().__init__()
        self.noise_threshold = noise_threshold
        self.edge_pixel_crop = edge_pixel_crop
        self.saturation_value = saturation_value
        self.calibration_image_tilt = calibration_image_tilt
        self.calibration_wavelength_pixel = calibration_wavelength_pixel
        self.calibration_0th_order_pixel = calibration_0th_order_pixel
        self.minimum_wavelength_analysis = minimum_wavelength_analysis
        self.optimization_central_wavelength = optimization_central_wavelength
        self.optimization_bandwidth_wavelength = optimization_bandwidth_wavelength

    def analyze_image(self, image: Array2D, auxiliary_data: Optional[dict] = None,
                      ) -> dict[str, Union[float, np.ndarray]]:
        input_params = self.build_input_parameter_dictionary()
        processed_image = image.astype(np.float32)
        returned_image, spec_dict, lineouts = analyze.analyze_image(processed_image, input_params)
        uint_image = returned_image.astype(np.uint16)

        return_dictionary = {
            "processed_image_uint16": uint_image,
            "analyzer_return_dictionary": spec_dict,
            "analyzer_return_lineouts": lineouts,
            "analyzer_input_parameters": input_params
        }
        return return_dictionary

    def build_input_parameter_dictionary(self) -> dict:
        input_params = {
            "noise_threshold_int": self.noise_threshold,
            "edge_crop_pixels": self.edge_pixel_crop,
            "saturation_value_int": self.saturation_value,
            "image_tilt_calibration_degrees": self.calibration_image_tilt,
            "wavelength_calibration_nm/pixel": self.calibration_wavelength_pixel,
            "zeroth_order_calibration_pixel": self.calibration_0th_order_pixel,
            "minimum_wavelength_nm": self.minimum_wavelength_analysis,
            "optimization_central_wavelength_nm": self.optimization_central_wavelength,
            "optimization_bandwidth_wavelength_nm": self.optimization_bandwidth_wavelength,
        }
        return input_params
