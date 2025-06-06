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

from pathlib import Path
from scipy.interpolate import interp1d

from ..base import LabviewImageAnalyzer
from .online_analysis_modules import image_processing_funcs as process
from .online_analysis_modules import photon_spectrometer_analyzer as analyze

from ..utils import read_imaq_image, ROI
from .generic_beam_analyzer import BeamSpotAnalyzer


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
                 pointing_correction_bool: bool = False,
                 spectral_correction_bool: bool = False,
                 spectral_correction_files: List[str] = None
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
            pointing_correction_bool: bool
                Boolean to toggle on and off the 0th order pointing correction functionality.
            spectral_correction_bool: bool
                Boolean to toggle on and off the spectral correction functionality.
            spectral_correction_files: list of strings
                List of filenames for optic transmission and detection efficiencies. Simultaneously defines which optics are necessary for proper correction.
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

        if not isinstance(pointing_correction_bool, bool):
            raise TypeError("pointing_correction_bool should be of bool type")
        self.pointing_correction_bool = pointing_correction_bool

        if not isinstance(spectral_correction_bool, bool):
            raise TypeError("spectral_correction_bool should be of bool type")
        self.spectral_correction_bool = spectral_correction_bool

        self.spectral_correction_files = list()
        self.spectral_correction_data = None

        # Set do_print to True for debugging information
        self.do_print = False
        self.computational_clock_time = time.perf_counter()

    def configure(self, **kwargs):
        super().configure(**kwargs)
        if self.spectral_correction_bool:
            self.spectral_correction_data = self.get_spectral_correction_data()


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

        if self.pointing_correction_bool:
            self.calibration_0th_order_pixel = self.perform_pointing_correction(rotated_image)
            self.print_time(" Pointing Correction")

        wavelength_array, spectrum_array = analyze.get_spectrum_lineouts(rotated_image,
                                                                         self.calibration_wavelength_pixel,
                                                                         self.calibration_0th_order_pixel)
        self.print_time(" Spectrum Lineouts")

        if self.spectral_correction_bool:
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

    def perform_pointing_correction(self, image):

        # copy image
        image = image.copy()

        # initialize beam spot image_analyzer
        roi = ROI(top=None, bottom=None, left=None,
                  right=int(self.calibration_0th_order_pixel + abs(self.minimum_wavelength_analysis / self.calibration_wavelength_pixel)),
                  bad_index_order='invert')
        analyzer = BeamSpotAnalyzer(roi=roi,
                                    bool_hp=True,
                                    hp_median=int(2),
                                    hp_thresh=float(3.0),
                                    bool_thresh=True,
                                    thresh_median=2,
                                    thresh_coeff=0.1)

        # crop
        image = roi.crop(image)

        # analyze for centroid of 0th order
        output = analyzer.analyze_image(image)
        centx_0th_order = output['analyzer_return_dictionary']['centroid'][1]

        return centx_0th_order

    def perform_spectral_correction(self, wavelength, spectrum, image,
                                    efficiency_lower_bound=0.025):

        # calculate total efficiency
        interp = interp1d(self.spectral_correction_data['wavelength'],
                          self.spectral_correction_data['efficiency'],
                          kind='linear', bounds_error=False, fill_value=(1.0, 1.0))
        spectral_correction = interp(wavelength)

        # if any efficiency too low, don't correct
        spectral_correction[spectral_correction < efficiency_lower_bound] = 1.0

        # correct spectrum
        corrected_spectrum = spectrum / spectral_correction

        # correct image
        corrected_image = image / spectral_correction

        return corrected_spectrum, corrected_image

    def get_spectral_correction_data(self):

        # define relative path
        spectral_correction_files_folder: Path = Path(__file__).parents[1] / 'data' # for GEECS-Plugins

        # initialize output dict
        return_dict = {'wavelength': None, 'efficiency': None}

        # import response data files
        correction_data = {
            file: np.loadtxt(spectral_correction_files_folder / file, delimiter='\t', skiprows=1, dtype=float)
            for file in self.spectral_correction_files
            }

        # process data and convert efficiency if necessary
        for file, data in correction_data.items():
            correction_data[file] = {'wavelength': data[:, 0], 'efficiency': data[:, 1]}

        # construct compiled wavelength array from imported data
        # range bound to common range limits, include all unique wavelength points
        min_bound = max(np.min(data['wavelength']) for data in correction_data.values())
        max_bound = min(np.max(data['wavelength']) for data in correction_data.values())

        all_wavelengths = np.concatenate([
            data['wavelength'][(data['wavelength'] >= min_bound) & (data['wavelength'] <= max_bound)]
            for data in correction_data.values()])
        unique_wavelengths = np.sort(np.unique(all_wavelengths))

        if len(unique_wavelengths) == 0:
            raise Exception('Spectral correction file wavelengths not compatible.')

        # construct total response efficiency curve
        total_efficiency = np.ones_like(unique_wavelengths)
        for data in correction_data.values():
            interp = interp1d(data['wavelength'], data['efficiency'],
                              kind='linear', bounds_error=False, fill_value=(1.0, 1.0))
            total_efficiency *= interp(unique_wavelengths)

        return_dict = {'wavelength': unique_wavelengths,
                       'efficiency': total_efficiency}

        return return_dict

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
            "pointing_correction_bool": self.pointing_correction_bool,
            "spec_correction_bool": self.spectral_correction_bool,
            "spec_correction_files": self.spectral_correction_files,
        }
        return input_params

    def print_time(self, label):
        if self.do_print:
            print(label, time.perf_counter() - self.computational_clock_time)
            self.computational_clock_time = time.perf_counter()
