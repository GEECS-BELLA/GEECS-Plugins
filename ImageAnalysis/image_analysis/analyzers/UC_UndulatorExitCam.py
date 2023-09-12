"""
Wednesday 9-6-2023

Will eventually flesh this out more.  Just writing out what I want the analysis to do.  Mostly just rotate the image
and then do a projection.  Then turn this projection into a spectra.

@Chris
"""
from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..types import Array2D

from ..base import ImageAnalyzer
from .online_analysis_modules import photon_spectrometer_analyzer as analyze


class UC_UndulatorExitCam(ImageAnalyzer):
    def __init__(self,
                 noise_threshold: int = 100,
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
        optimization_central_wavelength: float
            For the XOpt algorithm, the central wavelength by which to optimize the beam onto for the Gaussian weight
        calibration_image_tilt: float
            Tilt angle of the image in degrees
        calibration_wavelength_pixel: float
            Linear calibration of the wavelength per pixel in nm/pixel
        calibration_0th_order_pixel: float
            Calibrated position of the 0th order in pixel number
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

    def analyze_image(self, image: Array2D, auxiliary_data: Optional[dict] = None) -> tuple[
        NDArray[np.uint16], dict[str, Any], dict[str, Any]]:
        input_params = {
            "Threshold-Value": self.noise_threshold,
            "Pixel-Crop": self.edge_pixel_crop,
            "Saturation-Value": self.saturation_value,
            "Image-Tilt": self.calibration_image_tilt,
            "Wavelength-Calibration": self.calibration_wavelength_pixel,
            "0th-Order-Calibration": self.calibration_0th_order_pixel,
            "Minimum-Wavelength": self.minimum_wavelength_analysis,
            "Optimization-Central-Wavelength": self.optimization_central_wavelength,
            "Optimization-Bandwidth-Wavelength": self.optimization_bandwidth_wavelength
        }
        processed_image = image.astype(np.float32)
        returned_image, mag_spec_dict, lineouts = analyze.analyze_image(processed_image, input_params)
        uint_image = returned_image.astype(np.uint16)
        return uint_image, mag_spec_dict, input_params, lineouts
