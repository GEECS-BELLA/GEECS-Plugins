"""
Class definition for mag spec cam

Analyzer can be initialized through either providing the input parameters through __init__, by providing a config file
that defines the input parameters, or by calling a respective function in default_analyzer_generators to initialize an
analyzer class through a default config file.

@ Chris
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Union, List
import numpy as np
import time

if TYPE_CHECKING:
    from ..types import Array2D

from ..base import LabviewImageAnalyzer
from .online_analysis_modules import mag_spec_analysis as analyze
from .online_analysis_modules import mag_spec_energy_axis as energy_axis_lookup


class UC_GenericMagSpecCamAnalyzer(LabviewImageAnalyzer):
    def __init__(self, config_file=None, **kwargs):
        self.mag_spec_name = None
        self.noise_threshold = None
        self.saturation_value = None
        self.normalization_factor = None
        self.transverse_calibration = None
        self.do_transverse_calculation = None
        self.transverse_slice_threshold = None
        self.transverse_slice_binsize = None
        self.optimization_central_energy = None
        self.optimization_bandwidth_energy = None

        # Set do_print to True for debugging information
        self.do_print = False
        self.computational_clock_time = time.perf_counter()

        super().__init__(config_file, **kwargs)

    def configure(self,
                  mag_spec_name: str = 'NA',
                  roi: List[int] = [None, None, None, None],  # ROI(top, bottom, left, right)
                  noise_threshold: int = 0,  # CONFIRM IF THIS WORKS
                  saturation_value: int = 4095,
                  normalization_factor: float = 1.0,  # 7.643283839778091e-07,   # NEED TO CALCULATE
                  transverse_calibration: float = 1.0,
                  do_transverse_calculation: bool = True,  # IS THIS ANALYSIS USEFUL HERE?
                  transverse_slice_threshold: float = 0,  # ^^
                  transverse_slice_binsize: int = 10,  #
                  optimization_central_energy: float = 100.0,  # IS THIS ANALYSIS USEFUL HERE?
                  optimization_bandwidth_energy: float = 2.0  # ^^
                  ):
        """
        Parameters
        ----------
        mag_spec_name: str
            Name of the mag spec camera.  Needed to look up roi and energy axis information.
            Current accepted values are:
                hires
                acave3
        roi: List[int]
            The bounds by which the input image is cropped when the analyze_image function is called.  Given as a list
            of four integers, arranged as [top, bottom, left, right].
        noise_threshold: int
            Large enough to remove noise level
        saturation_value: int
            The minimum value by which a pixel can be considered as "Saturated." The default is 2^12 - 1
        normalization_factor: float
            Factor to go from camera counts to pC/MeV. Depends on trigger delay, exposure, and the threshold value for
            magspec analysis. See calibration_scripts/scripts_charge_calibration for how this is calculated
            default value comes from July 25th, Scan 24, HiResMagSpec,
                normalization_triggerdelay = 15.497208
                normalization_exposure = 0.010000
                normalization_thresholdvalue = 100
        transverse_calibration: float
            Transverse resolution of the image in ums / pixel
        do_transverse_calculation: bool
            Set as True to perform the transverse beam size and beam tilt calculation.  Set as False to skip the
            calculation.  This is the most time-intensive step, especially if the bin size is small.
        transverse_slice_threshold: float
            Only perform a transverse beam size calculation on a particular bin if the maximum of this bin is greater
            than the maximum of the entire image multiplied by this factor.  Larger values result in more transverse
            bins being skipped, and a factor of 0.0 results in every bin in the image being included.
        transverse_slice_binsize: int
            The bin size in number of transverse slices for the transverse calculation algorithm.  Larger bin size leads
            to a faster calculation at the expense of a decreased resolution.
        optimization_central_energy: float
            For the XOpt algorithm, the central energy by which to optimize the beam onto for the Gaussian weight func.
        optimization_bandwidth_energy: float
            For the XOpt algorithm, the standard deviation from the central energy for the Gaussian weight function
        """
        if self.roi is None:
            self.roi = roi
        self.mag_spec_name = str(mag_spec_name)
        self.noise_threshold = int(noise_threshold)
        self.saturation_value = int(saturation_value)
        self.normalization_factor = float(normalization_factor)
        self.transverse_calibration = float(transverse_calibration)
        self.do_transverse_calculation = bool(do_transverse_calculation)
        self.transverse_slice_threshold = float(transverse_slice_threshold)
        self.transverse_slice_binsize = int(transverse_slice_binsize)
        self.optimization_central_energy = float(optimization_central_energy)
        self.optimization_bandwidth_energy = float(optimization_bandwidth_energy)

    def analyze_image(self, input_image: Array2D, auxiliary_data: Optional[dict] = None,
                      ) -> dict[str, Union[dict, np.ndarray]]:
        processed_image = self.roi_image(input_image.astype(np.float32))

        saturation_number = analyze.saturation_check(processed_image, self.saturation_value)
        self.print_time(" Saturation Check:")

        image = analyze.threshold_reduction(processed_image, self.noise_threshold)
        self.print_time(" Threshold Subtraction")

        image = analyze.normalize_image(image, self.normalization_factor)
        self.print_time(" Normalize Image:")

        image = np.copy(image[::-1, ::-1])
        self.print_time(" Rotate Image")

        image_width = np.shape(image)[1]
        pixel_arr = np.linspace(0, image_width, image_width)
        energy_arr = energy_axis_lookup.return_energy_axis(pixel_arr, self.mag_spec_name)
        self.print_time(" Calculated Energy Axis:")

        charge_on_camera = np.sum(image)
        self.print_time(" Charge on Camera")

        clipped_percentage = analyze.calculate_clipped_percentage(image)
        self.print_time(" Calculate Clipped Percentage")

        charge_arr = analyze.calculate_charge_density_distribution(image, self.transverse_calibration)
        self.print_time(" Charge Projection:")

        if charge_on_camera == 0:
            peak_charge = 0.0
            average_energy = 0.0
            energy_spread = 0.0
            energy_spread_percent = 0.0
            peak_charge_energy = 0.0
            average_beam_size = 0.0
            beam_angle = 0.0
            beam_intercept = 0.0
            projected_beam_size = 0.0
            optimization_factor = 0.0
            fwhm_charge_percent = 0.0
        else:
            peak_charge = analyze.calculate_maximum_charge(charge_arr)
            self.print_time(" Peak Charge:")

            average_energy = analyze.calculate_average_energy(charge_arr, energy_arr)
            self.print_time(" Average Energy:")

            energy_spread = analyze.calculate_standard_deviation_energy(charge_arr, energy_arr, average_energy)
            energy_spread_percent = energy_spread / average_energy * 100
            self.print_time(" Energy Spread:")

            peak_charge_energy = analyze.calculate_peak_energy(charge_arr, energy_arr)
            self.print_time(" Energy at Peak Charge:")

            fwhm_charge_percent = analyze.calculate_fwhm_relative(charge_arr, energy_arr, maximum_charge=peak_charge,
                                                                  peak_energy=peak_charge_energy)
            self.print_time(" FWHM Charge Distribution:")

            optimization_factor = analyze.calculate_optimization_factor(
                charge_arr, energy_arr, self.optimization_central_energy, self.optimization_bandwidth_energy)
            self.print_time(" Optimization Factor:")

            if self.do_transverse_calculation:

                sigma_arr, x0_arr, amp_arr, err_arr = analyze.transverse_slice_loop(
                    image,
                    calibration_factor=self.transverse_calibration,
                    threshold=self.transverse_slice_threshold,
                    binsize=self.transverse_slice_binsize)
                self.print_time(" Gaussian Fits for each Slice:")

                if np.sum(amp_arr) > 0:
                    average_beam_size = analyze.calculate_average_size(sigma_arr, amp_arr)
                    self.print_time(" Average Beam Size:")

                    linear_fit = analyze.fit_beam_angle(x0_arr, amp_arr, energy_arr)
                    self.print_time(" Beam Angle Fit:")

                    beam_angle = linear_fit[0]
                    beam_intercept = linear_fit[1]
                    projected_axis, projected_arr, projected_beam_size = analyze.calculate_projected_beam_size(image, self.transverse_calibration)
                    self.print_time(" Projected Size:")
                else:
                    print("Error with transverse calcs.  Some charge on camera but still sum to zero:")
                    print("charge_on_camera=", charge_on_camera)
                    print("sum(amp_arr)=", np.sum(amp_arr))

                    average_beam_size = 0.0
                    projected_beam_size = 0.0
                    beam_angle = 0.0
                    beam_intercept = 0.0
            else:
                average_beam_size = 0.0
                projected_beam_size = 0.0
                beam_angle = 0.0
                beam_intercept = 0.0

        mag_spec_dict = {
            "camera_clipping_factor": clipped_percentage,
            "camera_saturation_counts": saturation_number,
            "total_charge_pC": charge_on_camera,
            "peak_charge_pc/MeV": peak_charge,
            "peak_charge_energy_MeV": peak_charge_energy,
            "weighted_average_energy_MeV": average_energy,
            "energy_spread_weighted_rms_MeV": energy_spread,
            "energy_spread_percent": energy_spread_percent,
            "weighted_average_beam_size_um": average_beam_size,
            "projected_beam_size_um": projected_beam_size,
            "beam_tilt_um/MeV": beam_angle,
            "beam_tilt_intercept_um": beam_intercept,
            "beam_tilt_intercept_100MeV_um": 100 * beam_angle + beam_intercept,
            "optimization_factor": optimization_factor,
            "fwhm_percent": fwhm_charge_percent,
        }

        unnormalized_image = image / self.normalization_factor
        uint_image = unnormalized_image.astype(np.uint16)
        return_dictionary = {
            "processed_image_uint16": uint_image,
            "analyzer_return_dictionary": mag_spec_dict,
            "analyzer_return_lineouts": np.vstack((energy_arr, charge_arr)),
            "analyzer_input_parameters": self.build_input_parameter_dictionary()
        }
        return return_dictionary

    def build_input_parameter_dictionary(self) -> dict:
        input_params = {
            "mag_spec_name_str": self.mag_spec_name,
            "roi_bounds_pixel": self.roi,
            "noise_threshold_int": self.noise_threshold,
            "saturation_value_int": self.saturation_value,
            "charge_normalization_factor_pC/count": self.normalization_factor,
            "transverse_calibration_factor_um/pixel": self.transverse_calibration,
            "optimization_central_energy_MeV": self.optimization_central_energy,
            "optimization_bandwidth_energy_MeV": self.optimization_bandwidth_energy,
            "do_transverse_calibration_bool": self.do_transverse_calculation,
            "transverse_slice_threshold_factor": self.transverse_slice_threshold,
            "transverse_slice_bin_size_pixels": self.transverse_slice_binsize,
        }
        return input_params

    def print_time(self, label):
        if self.do_print:
            print(label, time.perf_counter() - self.computational_clock_time)
            self.computational_clock_time = time.perf_counter()
