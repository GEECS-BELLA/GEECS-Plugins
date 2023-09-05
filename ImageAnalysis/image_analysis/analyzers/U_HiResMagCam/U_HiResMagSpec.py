"""
Class definition for mag spec analysis 

@ Chris
"""
from __future__ import annotations

from array import array
from typing import Optional, Any, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ...types import Array2D

from ...base import ImageAnalyzer
# import sys
# sys.path.append(r"C:\GEECS\Developers Version\source\GEECS-Plugins\ImageAnalysis\image_analysis")
# from base import ImageAnalyzer

# Either importing with the path set to GEECS-PythonAPI (as is the case for post-analysis scripts elsewhere)
#  or importing with the path set to this location (which is the case for when run on LabView)

from .OnlineAnalysisModules import HiResMagSpecAnalysis as MagSpecAnalysis


class U_HiResMagSpecImageAnalyzer(ImageAnalyzer):

    def __init__(self, 
                 noise_threshold: int = 100,
                 edge_pixel_crop: int = 1,
                 saturation_value: int = 4095,
                 normalization_factor: float = 7.643283839778091e-07,
                 transverse_calibration: int = 43,
                 do_transverse_calculation: bool = True,
                 transverse_slice_threshold: float = 0.02,
                 transverse_slice_binsize: int = 5,
                 optimization_central_energy: float = 100.0,
                 optimization_bandwidth_energy: float = 2.0
                ):
        """
        Parameters
        ----------
        noise_threshold: int
            Large enough to remove noise level
        edge_pixel_crop: int
            Number of edge pixels to crop
        saturation_value: int
            needs doc
        normalization_factor: float
            Factor to go from camera counts to pC/MeV
            Depends on trigger delay, exposure, and the threshold value for 
            magspec analysis
            default value comes from July 25th, Scan 24, HiResMagSpec, 
                normalization_triggerdelay = 15.497208
                normalization_exposure = 0.010000
                normalization_thresholdvalue = 100 # 230
        transverse_calibration: int
            ums / pixel
        do_transverse_calculation: bool
            needs doc
        transverse_slice_threshold: float
            needs doc
        transverse_slice_binsize: int
            needs doc
        optimization_central_energy: float
            needs doc
        optimization_bandwidth_energy: float
            needs doc
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

    def analyze_image(self, image: Array2D, auxiliary_data: Optional[dict] = None) -> tuple[NDArray[np.uint16], dict[str, Any], dict[str, Any]]:

        inputParams = {
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
        returned_image, MagSpecDict, lineouts = MagSpecAnalysis.AnalyzeImage(processed_image, inputParams)
        unnormalized_image = returned_image / self.normalization_factor
        uint_image = unnormalized_image.astype(np.uint16)
        return uint_image, MagSpecDict, inputParams, lineouts
