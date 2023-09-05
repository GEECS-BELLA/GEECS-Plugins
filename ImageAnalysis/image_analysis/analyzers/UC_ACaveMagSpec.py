"""
Class definition for a cave mag spec analysis

This still needs a lot of work, and most of it is a copy-paste of the hi res version.  If enough of the analysis itself
is a copy-paste, then consider making either a single analysis module with particular inputs or have a class with two
daughter classes for each analysis.

Also, we are working with ACaveMagSpec's Cam3

@ Chris
"""
from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..types import Array2D

from ..base import ImageAnalyzer
from .online_analysis_modules import acave_mag_spec_analysis as analyze


class UC_ACaveMagSpecImageAnalyzer(ImageAnalyzer):

    def __init__(self,
                 noise_threshold: int = 100,                            # CONFIRM IF THIS WORKS
                 edge_pixel_crop: int = 1,
                 saturation_value: int = 4095,
                 normalization_factor: float = 7.643283839778091e-07,   # NEED TO CALCULATE
                 transverse_calibration: int = 129.4,
                 do_transverse_calculation: bool = True,                # IS THIS ANALYSIS USEFUL HERE?
                 transverse_slice_threshold: float = 0.02,              # ^^
                 transverse_slice_binsize: int = 5,                     #
                 optimization_central_energy: float = 100.0,            # IS THIS ANALYSIS USEFUL HERE?
                 optimization_bandwidth_energy: float = 2.0             # ^^
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
        normalization_factor: float
            Factor to go from camera counts to pC/MeV. Depends on trigger delay, exposure, and the threshold value for
            magspec analysis. See post_analysis/scripts_charge_calibration for how this is calculated
            default value comes from July 25th, Scan 24, HiResMagSpec,
                normalization_triggerdelay = 15.497208
                normalization_exposure = 0.010000
                normalization_thresholdvalue = 100
        transverse_calibration: int
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
        self.noise_threshold = noise_threshold
        self.edge_pixel_crop = edge_pixel_crop
        self.saturation_value = saturation_value
        self.normalization_factor = normalization_factor
        self.transverse_calibration = transverse_calibration
        self.do_transverse_calculation = do_transverse_calculation
        self.transverse_slice_threshold = transverse_slice_threshold
        self.transverse_slice_binsize = transverse_slice_binsize
        self.optimization_central_energy = optimization_central_energy
        self.optimization_bandwidth_energy = optimization_bandwidth_energy

    def analyze_image(self, image: Array2D, auxiliary_data: Optional[dict] = None) -> tuple[
        NDArray[np.uint16], dict[str, Any], dict[str, Any]]:
        input_params = {
            "Threshold-Value": self.noise_threshold,
            "Pixel-Crop": self.edge_pixel_crop,
            "Saturation-Value": self.saturation_value,
            "Normalization-Factor": self.normalization_factor,
            "Transverse-Calibration": self.transverse_calibration,
            "Do-Transverse-Calculation": self.do_transverse_calculation,
            "Transverse-Slice-Threshold": self.transverse_slice_threshold,
            "Transverse-Slice-Binsize": self.transverse_slice_binsize,
            "Optimization-Central-Energy": self.optimization_central_energy,
            "Optimization-Bandwidth-Energy": self.optimization_bandwidth_energy
        }
        processed_image = image.astype(np.float32)
        returned_image, mag_spec_dict, lineouts = analyze.analyze_image(processed_image, input_params)
        unnormalized_image = returned_image / self.normalization_factor
        uint_image = unnormalized_image.astype(np.uint16)
        return uint_image, mag_spec_dict, input_params, lineouts
