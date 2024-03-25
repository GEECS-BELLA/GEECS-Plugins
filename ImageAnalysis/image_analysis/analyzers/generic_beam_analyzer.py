"""

GEECS Plugin - Beam analyzer
Kyle Jensen, kjensen@lbl.gov

Adapted from UC_BeamSpot by Guillaume Plateau, grplateau@lbl.gov

"""
# =============================================================================
# %% imports

from __future__ import annotations

from typing import Union, Optional
import configparser

import numpy as np
from scipy.ndimage import median_filter
from skimage.measure import regionprops, label

from ..base import ImageAnalyzer
from ..utils import ROI
from ..tools.filtering import clip_hot_pixels
# =============================================================================
# %% handle config file

def return_analyzer_from_config_file(config_filename) -> BeamSpotAnalyzer:

    config = configparser.ConfigParser()
    config.read(config_filename)

    analyzer_roi = ROI(top=config.get('roi', 'top'),
                       bottom=config.get('roi', 'bottom'),
                       left=config.get('roi', 'left'),
                       right=config.get('roi', 'right'),
                       bad_index_order='invert'
                       )

    analyzer = BeamSpotAnalyzer(
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

class BeamSpotAnalyzer(ImageAnalyzer):
    """
    Image analysis for e-beam or laser beam spot profiles.

    Inherits ImageAnalyzer super class.
    """

    # initialization
    def __init__(self,
                 roi: ROI = ROI(),
                 bool_hp: bool = True,
                 hp_median: int = 2,
                 hp_thresh: float = 3.0,
                 bool_thresh: bool = True,
                 thresh_median: int = 2,
                 thresh_coeff: float = 0.1,
                 background: Optional[np.ndarray] = None
                 ):

        super().__init__()

        self.roi = roi
        self.bool_hp = bool_hp
        self.hp_median = hp_median
        self.hp_thresh = hp_thresh
        self.bool_thresh = bool_thresh
        self.thresh_median = thresh_median
        self.thresh_coeff = thresh_coeff
        self.background = background

    def image_signal_thresholding(self, image:np.ndarray) -> np.ndarray:

        data_type = image.dtype
        image = image.astype('float64')

        # perform median filtering
        blurred = median_filter(image, size=self.thresh_median)

        # threshold with respect to the blurred image max
        image[blurred < blurred.max() * self.thresh_coeff] = 0

        return image.astype(data_type)

    def find_beam_properties(self, image:np.ndarray) -> np.ndarray:

        # initialize beam properties dict
        beam_properties = {}

        # construct binary and label images
        image_binary = image.copy()
        image_binary[image_binary > 0] = 1
        image_binary = image_binary.astype(int)
        image_label = label(image_binary)

        # get beam properties and reduce to largest region
        props = regionprops(image_label, image)
        areas = [i.area for i in props]
        props = props[areas.index(max(areas))]

        # extract centroid
        beam_properties['centroid'] = props.centroid_weighted

        return beam_properties

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
        results_dict = {"centroid": beam_properties['centroid']
                        }

        # organize return dict
        return_dict = {"processed_image_uint16": image_cor,
                       "analyzer_return_dictionary": results_dict,
                       "analyzer_input_parameters": self.build_input_parameter_dictionary()
                       }

        return return_dict

    def build_input_parameter_dictionary(self) -> dict:

        input_params = {'roi': [self.roi.top, self.roi.bottom, self.roi.left, self.roi.right],
                        'bool_hp': self.bool_hp,
                        'hp_median': self.hp_median,
                        "hp_thresh": self.hp_thresh,
                        "bool_thresh": self.bool_thresh,
                        "thresh_median": self.thresh_median,
                        "thresh_coeff": self.thresh_coeff
                        }

        return input_params
# =============================================================================
