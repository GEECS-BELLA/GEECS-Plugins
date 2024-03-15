"""

GEECS Plugin - Camera device analyzer
Kyle Jensen, kjensen@lbl.gov

"""
# =============================================================================
# %% imports

from __future__ import annotations

from typing import Union
import configparser

import numpy as np

from ..utils import ROI
from ..tools.filtering import clip_hot_pixels

from .generic_beam_analyzer import BeamSpotAnalyzer
# =============================================================================
# %% handle config file

def return_analyzer_from_config_file(config_filename) -> AnalyzerAmp2Input:

    config = configparser.ConfigParser()
    config.read(config_filename)

    analyzer_roi = ROI(top=config.get('roi', 'top'),
                       bottom=config.get('roi', 'bottom'),
                       left=config.get('roi', 'left'),
                       right=config.get('roi', 'right'),
                       bad_index_order='invert'
                       )

    analyzer = AnalyzerAmp2Input(
        roi=analyzer_roi,
        bool_hp=bool(config.get('hotpixel', 'bool_hp')),
        hp_median=int(config.get('hotpixel', 'hp_median')),
        hp_thresh=float(config.get('hotpixel', 'hp_thresh')),
        bool_thresh=bool(config.get('thresholding', 'bool_thresh')),
        thresh_median=int(config.get('thresholding', 'thresh_median')),
        thresh_coeff=float(config.get('thresholding', 'thresh_coeff')),
        )

    return analyzer
# =============================================================================
# %% beam spot analyzer

class AnalyzerAmp2Input(BeamSpotAnalyzer):
    """
    Image analysis for amp2 input camera.

    Inherits: BeamSpotAnalyzer, ImageAnalyzer
    """

    def analyze_image(self,
                      image: np.ndarray,
                      ) -> dict[str, Union[float, np.ndarray]]:

        # run super analysis
        return_dict = super().analyze_image(image)

        return return_dict

# =============================================================================
