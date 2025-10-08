"""

GEECS Plugin - Beam image_analyzer
Kyle Jensen, kjensen@lbl.gov

Adapted from UC_BeamSpot by Guillaume Plateau, grplateau@lbl.gov

"""
# =============================================================================
# %% imports

from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional

if TYPE_CHECKING:
    from ..types import Array2D

import numpy as np
from scipy.ndimage import median_filter
from skimage.measure import regionprops, label

from ..base import LabviewImageAnalyzer
from ..utils import ROI
from image_analysis.processing.array2d.filtering import apply_median_filter


# =============================================================================
# %% beam spot image_analyzer


class BeamSpotAnalyzer(LabviewImageAnalyzer):
    """
    Image analysis for e-beam or laser beam spot profiles.

    Inherits ImageAnalyzer super class.
    """

    def __init__(
        self,
        roi: ROI = ROI(),
        bool_hp: bool = True,
        hp_median: int = 2,
        hp_thresh: float = 3.0,
        bool_thresh: bool = True,
        thresh_median: int = 2,
        thresh_coeff: float = 0.1,
    ):
        super().__init__()

        self.roi = roi
        self.bool_hp = bool(bool_hp)
        self.hp_median = int(hp_median)
        self.hp_thresh = float(hp_thresh)
        self.bool_thresh = bool(bool_thresh)
        self.thresh_median = int(thresh_median)
        self.thresh_coeff = float(thresh_coeff)

    def image_signal_thresholding(self, image: np.ndarray) -> np.ndarray:
        data_type = image.dtype
        image = image.astype("float64")

        # perform median filtering
        blurred = median_filter(image, size=self.thresh_median)

        # threshold with respect to the blurred image max
        image[blurred < blurred.max() * self.thresh_coeff] = 0

        return image.astype(data_type)

    @staticmethod
    def find_beam_properties(image: np.ndarray):
        # initialize beam properties dict
        beam_properties = {}

        # construct binary and label images
        image_binary = image.copy()
        image_binary[image_binary > 0] = 1
        image_binary = image_binary.astype(int)
        image_label = label(image_binary)

        # get beam properties and reduce to the largest region
        props = regionprops(image_label, image)
        areas = [i.area for i in props]
        props = props[areas.index(max(areas))]

        # extract centroid
        beam_properties["centroid"] = props.centroid_weighted

        return beam_properties

    def analyze_image(
        self,
        image: Array2D,
        auxiliary_data: Optional[dict] = None,
    ) -> dict[str, Union[dict, np.ndarray]]:
        # initialize processed image
        image_cor = image.copy()

        # perform hot pixel correction
        if self.bool_hp:
            image_cor = apply_median_filter(
                image_cor,
                kernel_size=self.hp_median,
            )
        # threshold signal
        if self.bool_thresh:
            image_cor = self.image_signal_thresholding(image_cor)

        # extract beam properties
        beam_properties = self.find_beam_properties(image_cor)

        # extract lineouts wrt centroid
        # lineouts = {}
        # lineouts['x'] = image[int(beam_properties['centroid'][0]), :]
        # lineouts['y'] = image[:, int(beam_properties['centroid'][1])]

        # organize results dict
        results_dict = {"centroid": beam_properties["centroid"]}
        # organize return dict
        return_dictionary = self.build_return_dictionary(
            return_image=image_cor,
            return_scalars=results_dict,
            return_lineouts=None,
            input_parameters=self.build_input_parameter_dictionary(),
        )
        return return_dictionary

    def build_input_parameter_dictionary(self) -> dict:
        input_params = {
            "roi": [self.roi.top, self.roi.bottom, self.roi.left, self.roi.right],
            "bool_hp": self.bool_hp,
            "hp_median": self.hp_median,
            "hp_thresh": self.hp_thresh,
            "bool_thresh": self.bool_thresh,
            "thresh_median": self.thresh_median,
            "thresh_coeff": self.thresh_coeff,
        }

        return input_params


# =============================================================================
