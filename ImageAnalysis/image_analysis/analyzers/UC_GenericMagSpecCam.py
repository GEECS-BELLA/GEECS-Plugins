"""Magnet spectrometer camera analyzer for LabVIEW image streams.

This module defines `UC_GenericMagSpecCamAnalyzer`, a `LabviewImageAnalyzer`
specialization that:
- Crops to a user-provided ROI.
- Reduces noise and checks saturation.
- Normalizes counts to pC/MeV.
- Flips image orientation for analysis.
- Builds an energy axis from a named spectrometer configuration.
- Computes charge projection and scalar beam/energy metrics.
- Optionally performs transverse size/tilt analysis via per-slice Gaussian fits.
- Returns a standard result dictionary with image, scalars, lineouts, and inputs.

Initialization can be done programmatically (via `__init__`), from a config file,
or through helper constructors in `default_analyzer_generators`.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Union, List
import numpy as np
import time

if TYPE_CHECKING:
    from ..types import Array2D

from ..base import LabviewImageAnalyzer
from .online_analysis_modules import image_processing_funcs as process
from .online_analysis_modules import mag_spec_analysis as analyze
from .online_analysis_modules import mag_spec_energy_axis as energy_axis_lookup


class UC_GenericMagSpecCamAnalyzer(LabviewImageAnalyzer):
    """Generic analyzer for magnet spectrometer cameras producing charge-vs-energy."""

    def __init__(
        self,
        mag_spec_name: str = "NA",
        roi: List[int] = [None, None, None, None],  # ROI(top, bottom, left, right)
        noise_threshold: int = 0,
        saturation_value: int = 4095,
        normalization_factor: float = 1.0,
        transverse_calibration: float = 1.0,
        do_transverse_calculation: bool = True,
        transverse_slice_threshold: float = 0,
        transverse_slice_binsize: int = 10,
        optimization_central_energy: float = 100.0,
        optimization_bandwidth_energy: float = 2.0,
    ):
        """Configure analyzer and store analysis parameters.

        Parameters
        ----------
        mag_spec_name : str, default='NA'
            Name of the magnet spectrometer used for energy-axis lookup (e.g., 'hires', 'acave3').
        roi : list of int, default=[None, None, None, None]
            Crop bounds as ``[top, bottom, left, right]`` in pixel indices.
        noise_threshold : int, default=0
            Threshold below which pixels are treated as noise and reduced.
        saturation_value : int, default=4095
            Pixel value considered saturated (e.g., for 12-bit cameras: 2**12 - 1).
        normalization_factor : float, default=1.0
            Factor converting camera counts to pC/MeV for magspec analysis.
        transverse_calibration : float, default=1.0
            Transverse scale in µm/pixel used by transverse analysis.
        do_transverse_calculation : bool, default=True
            Whether to run per-slice Gaussian fits for beam size/tilt.
        transverse_slice_threshold : float, default=0
            Relative threshold (0–1) to skip dim transverse bins in the per-slice fits.
        transverse_slice_binsize : int, default=10
            Number of columns per bin for the transverse slice algorithm.
        optimization_central_energy : float, default=100.0
            Center energy (MeV) for an optional Gaussian weight used by optimization scoring.
        optimization_bandwidth_energy : float, default=2.0
            Sigma (MeV) for the Gaussian weighting used in optimization score.
        """
        super().__init__()

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

        # Set do_print to True for debugging timestamps.
        self.do_print = False
        self.computational_clock_time = time.perf_counter()

    def analyze_image(
        self,
        input_image: Array2D,
        auxiliary_data: Optional[dict] = None,
    ) -> dict[str, Union[dict, np.ndarray]]:
        """Run full magspec analysis and return image, scalars, and lineouts.

        Workflow
        --------
        1) Crop to ROI.
        2) Saturation check.
        3) Noise reduction.
        4) Normalize counts to pC/MeV.
        5) Flip image orientation for analysis.
        6) Build energy axis and compute charge projection.
        7) Compute scalar metrics (peak charge, mean energy, energy spread, FWHM, etc.).
        8) Optionally compute transverse size and beam tilt via slice-wise Gaussian fits.

        Parameters
        ----------
        input_image : Array2D
            2D image array (counts).
        auxiliary_data : dict, optional
            Extra info for analysis (currently unused).

        Returns
        -------
        dict
            Result dictionary from `build_return_dictionary` with:
            - ``return_image`` : numpy.ndarray
                Processed image in counts (not normalized), orientation flipped for display.
            - ``return_scalars`` : dict
                Keys include:
                * ``camera_clipping_factor`` : float
                * ``camera_saturation_counts`` : int
                * ``total_charge_pC`` : float
                * ``peak_charge_pc/MeV`` : float
                * ``peak_charge_energy_MeV`` : float
                * ``weighted_average_energy_MeV`` : float
                * ``energy_spread_weighted_rms_MeV`` : float
                * ``energy_spread_percent`` : float
                * ``weighted_average_beam_size_um`` : float
                * ``projected_beam_size_um`` : float
                * ``beam_tilt_um/MeV`` : float
                * ``beam_tilt_intercept_um`` : float
                * ``beam_tilt_intercept_100MeV_um`` : float
                * ``optimization_factor`` : float
                * ``fwhm_percent`` : float
            - ``return_lineouts`` : numpy.ndarray, shape (2, W)
                Stacked array with first row = energy axis (MeV), second row = charge per energy bin.
            - ``input_parameters`` : dict
                A snapshot of the main inputs used for this analysis.
        """
        processed_image = self.roi_image(input_image.astype(np.float32))

        saturation_number = process.saturation_check(
            processed_image, self.saturation_value
        )
        self.print_time(" Saturation Check:")

        image = process.threshold_reduction(processed_image, self.noise_threshold)
        self.print_time(" Threshold Subtraction")

        image = analyze.normalize_image(image, self.normalization_factor)
        self.print_time(" Normalize Image:")

        image = np.copy(image[::-1, ::-1])
        self.print_time(" Rotate Image")

        image_width = np.shape(image)[1]
        pixel_arr = np.linspace(0, image_width, image_width)
        energy_arr = energy_axis_lookup.return_energy_axis(
            pixel_arr, self.mag_spec_name
        )
        self.print_time(" Calculated Energy Axis:")

        charge_on_camera = np.sum(image)
        self.print_time(" Charge on Camera")

        clipped_percentage = analyze.calculate_clipped_percentage(image)
        self.print_time(" Calculate Clipped Percentage")

        charge_arr = analyze.calculate_charge_density_distribution(image, energy_arr)
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
            peak_charge = np.max(charge_arr)
            self.print_time(" Peak Charge:")

            average_energy = process.calculate_axis_average(charge_arr, energy_arr)
            self.print_time(" Average Energy:")

            energy_spread = process.calculate_standard_deviation(
                charge_arr, energy_arr, average_energy
            )
            energy_spread_percent = energy_spread / average_energy * 100
            self.print_time(" Energy Spread:")

            peak_charge_energy = energy_arr[np.argmax(charge_arr)]
            self.print_time(" Energy at Peak Charge:")

            fwhm_charge_percent = analyze.calculate_fwhm_relative(
                charge_arr,
                energy_arr,
                maximum_charge=peak_charge,
                peak_energy=peak_charge_energy,
            )
            self.print_time(" FWHM Charge Distribution:")

            optimization_factor = analyze.calculate_optimization_factor(
                charge_arr,
                energy_arr,
                self.optimization_central_energy,
                self.optimization_bandwidth_energy,
            )
            self.print_time(" Optimization Factor:")

            if self.do_transverse_calculation:
                sigma_arr, x0_arr, amp_arr, err_arr = analyze.transverse_slice_loop(
                    image,
                    calibration_factor=self.transverse_calibration,
                    threshold=self.transverse_slice_threshold,
                    binsize=self.transverse_slice_binsize,
                )
                self.print_time(" Gaussian Fits for each Slice:")

                if np.sum(amp_arr) > 0:
                    average_beam_size = np.average(sigma_arr, weights=amp_arr)
                    self.print_time(" Average Beam Size:")

                    linear_fit = analyze.fit_beam_angle(x0_arr, amp_arr, energy_arr)
                    self.print_time(" Beam Angle Fit:")

                    beam_angle = linear_fit[0]
                    beam_intercept = linear_fit[1]
                    projected_axis, projected_arr, projected_beam_size = (
                        analyze.calculate_projected_beam_size(
                            image, self.transverse_calibration
                        )
                    )
                    self.print_time(" Projected Size:")
                else:
                    print(
                        "Error with transverse calcs.  Some charge on camera but still sum to zero:"
                    )
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

        return_dictionary = self.build_return_dictionary(
            return_image=image / self.normalization_factor,
            return_scalars=mag_spec_dict,
            return_lineouts=np.vstack((energy_arr, charge_arr)),
            input_parameters=self.build_input_parameter_dictionary(),
        )
        return return_dictionary

    def build_input_parameter_dictionary(self) -> dict:
        """Return a dict snapshot of key input parameters used for analysis."""
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
        """Log elapsed time between steps when `do_print` is enabled."""
        if self.do_print:
            print(label, time.perf_counter() - self.computational_clock_time)
            self.computational_clock_time = time.perf_counter()
