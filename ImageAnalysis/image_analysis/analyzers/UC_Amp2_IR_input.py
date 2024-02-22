"""

GEECS Plugin - Camera device analyzer
Initial: Guillaume Plateau, grplateau@lbl.gov
Modified by: Kyle Jensen, kjensen@lbl.gov

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

        # initialize processed image
        image_cor = image.copy()

        # perform hot pixel correction
        if self.bool_hp:
            image_cor = clip_hot_pixels(image_cor,
                                        median_filter_size=self.hp_median,
                                        threshold_factor=self.hp_thresh)

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
        results_dict = {"centroidx": beam_properties['centroid'][1],
                        "centroidy": beam_properties['centroid'][0]
                        }

        # organize return dict
        return_dict = {"processed_image_uint16": image_cor,
                       "analyzer_return_dictionary": results_dict,
                       "analyzer_input_parameters": self.build_input_parameter_dictionary()
                       }

        return return_dict

# =============================================================================
