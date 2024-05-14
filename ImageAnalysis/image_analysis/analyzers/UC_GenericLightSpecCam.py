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
from .online_analysis_modules import image_processing_funcs as process
from .online_analysis_modules import photon_spectrometer_analyzer as analyze


class UC_LightSpectrometerCamAnalyzer(LabviewImageAnalyzer):
    def __init__(self,
                 noise_threshold: int = 50,
                 roi: List[int] = [None, None, None, None],  # ROI(top, bottom, left, right)
                 saturation_value: int = 4095,
                 calibration_image_tilt: float = 1.1253,
                 calibration_wavelength_pixel: float = 0.463658,
                 calibration_0th_order_pixel: float = 1186.24,
                 minimum_wavelength_analysis: float = 150.0,
                 optimization_central_wavelength: float = 420.0,
                 optimization_bandwidth_wavelength: float = 10.0,
                 spec_correction_bool: bool = True,
                 spec_correction_files: List[str] = []
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
        super().__init__()

        self.roi = roi
        self.noise_threshold = int(noise_threshold)
        self.saturation_value = int(saturation_value)
        self.calibration_image_tilt = float(calibration_image_tilt)
        self.calibration_wavelength_pixel = float(calibration_wavelength_pixel)
        self.calibration_0th_order_pixel = float(calibration_0th_order_pixel)
        self.minimum_wavelength_analysis = float(minimum_wavelength_analysis)
        self.optimization_central_wavelength = float(optimization_central_wavelength)
        self.optimization_bandwidth_wavelength = float(optimization_bandwidth_wavelength)
        self.spec_correction_bool = bool(spec_correction_bool)
        self.spec_correction_files = list(spec_correction_files)

        # Set do_print to True for debugging information
        self.do_print = False
        self.computational_clock_time = time.perf_counter()

    def analyze_image(self, input_image: Array2D, auxiliary_data: Optional[dict] = None,
                      ) -> dict[str, Union[dict, np.ndarray]]:

        processed_image = self.roi_image(input_image.astype(np.float32))

        saturation_number = process.saturation_check(processed_image, self.saturation_value)
        self.print_time(" Saturation Check:")

        image = process.threshold_reduction(processed_image, self.noise_threshold)
        self.print_time(" Threshold Subtraction")

        total_photons = np.sum(image)
        self.print_time(" Total Photon Counts")

        rotated_image = analyze.rotate_image(image, self.calibration_image_tilt)
        self.print_time(" Image Rotation")

        wavelength_array, spectrum_array = analyze.get_spectrum_lineouts(rotated_image,
                                                                         self.calibration_wavelength_pixel,
                                                                         self.calibration_0th_order_pixel)
        self.print_time(" Spectrum Lineouts")

        if self.spec_correction_bool:
            (spectrum_array,
             rotated_image) = self.perform_spectral_correction(wavelength_array,
                                                               spectrum_array,
                                                               rotated_image)
            self.print_time(" Spec Correction")

        crop_wavelength_array, crop_spectrum_array = analyze.crop_spectrum(wavelength_array, spectrum_array,
                                                                           self.minimum_wavelength_analysis)

        if np.sum(crop_spectrum_array) == 0:
            peak_wavelength = 0
            average_wavelength = 0
            wavelength_spread = 0
            optimization_factor = 0
            total_1st_order = 0

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
            total_1st_order = np.sum(crop_spectrum_array)

        exit_cam_dict = {
            "camera_saturation_counts": saturation_number,
            "camera_total_intensity_counts": total_1st_order,  # total_photons,
            "peak_wavelength_nm": peak_wavelength,
            "average_wavelength_nm": average_wavelength,
            "wavelength_spread_weighted_rms_nm": wavelength_spread,
            "optimization_factor": optimization_factor,
        }
        return_dictionary = self.build_return_dictionary(return_image=rotated_image,
                                                         return_scalars=exit_cam_dict,
                                                         return_lineouts=[wavelength_array, spectrum_array],
                                                         input_parameters=self.build_input_parameter_dictionary())
        return return_dictionary

    def perform_spectral_correction(self, wavelength, spectrum, image,
                                    eff_low_bound=0.025):

        # load detection efficiency
        spectral_correction = self.construct_spectral_correction(wavelength)

        # if any efficiency too low, don't correct
        spectral_correction['efficiency'][spectral_correction['efficiency'] < eff_low_bound] = 1.0

        # correct spectrum
        cor_spectrum = spectrum / spectral_correction['efficiency']

        # correct image
        cor_image = image / spectral_correction['efficiency']

        return cor_spectrum, cor_image

    def construct_spectral_correction(self, input_wavelength):

        # define relative path
        rel_path = os.path.join('..', 'data')

        # initialize storage
        spectral_correction = {'wavelength': input_wavelength.copy(),
                               'efficiency': np.ones_like(input_wavelength)}

        # loop through correction files
        for file in self.spec_correction_files:

            # load file data
            output = np.loadtxt(os.path.join(rel_path, file),
                                delimiter='\t', skiprows=1, dtype=float)

            # store
            file_wavelength = output[:, 0]
            file_efficiency = output[:, 1]

            # convert from percent to decimal
            if file in ('alvium_uvcam_qe.tsv'):
                file_efficiency = file_efficiency / 100.

            # interpolate and compile on spectral correction
            interp = interp1d(file_wavelength, file_efficiency,
                              kind='linear', bounds_error=False, fill_value=(1.0, 1.0))
            spectral_correction['efficiency'] *= interp(spectral_correction['wavelength'])

        return spectral_correction

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
