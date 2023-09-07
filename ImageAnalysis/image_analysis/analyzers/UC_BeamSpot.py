""" Contains UCBeamSpotScanImages class which is applicable to any cameras imaging a beam spot. 
"""

from __future__ import annotations

from typing import Union, Optional
from pathlib import Path

import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.transform import hough_ellipse
from skimage.feature import canny

from ..tools.spot import n_sigma_window

from ..base import ImageAnalyzer
from ..utils import ROI
from ..tools.filtering import basic_filter

class UC_BeamSpotImageAnalyzer(ImageAnalyzer):
    """ Implements image analysis for any e-beam or laser beam spot image

        Used for undulator Visa, Aline, and Phosphor cameras

    """
    def __init__(self,
                 camera_roi: ROI = ROI(),
                 contrast: float = 1.333,
                 hp_median: int = 2,
                 hp_threshold: float = 3.0,
                 denoise_cycles: int = 0,
                 gauss_filter: float = 5.0,
                 com_threshold: float = 0.5,
                 background: Optional[np.ndarray] = None,
                 box: bool = True,
                 ellipse: bool = False,
                 sigma_radius: float = 5.0,
                ):
        """
        Parameters
        ----------
        camera_roi: ROI
            region of interest, to which the image is already cropped. This
            is currently passed only for saving in the analysis.
        contrast: float
            needs doc
        hp_median: int
            hot pixels median filter size
        hp_threshold: float
            hot pixels threshold in # of sta. dev. of the median deviation
        denoise_cycles: int
            needs doc
        gauss_filter: float
            gaussian filter size 
        com_threshold: float
            image threshold for center-of-mass calculation
        background: np.ndarray or None
            not currently used
        box: bool
            needs doc
        ellipse: bool
            needs doc
        sigma_radius: float
            used in n_sigma_window function, needs doc

        """
        self.camera_roi = camera_roi

        # from FilterParameters
        self.contrast = contrast
        self.hp_median = hp_median
        self.hp_threshold = hp_threshold
        self.denoise_cycles = denoise_cycles
        self.gauss_filter = gauss_filter
        self.com_threshold = com_threshold
        self.box = box
        self.ellipse = ellipse

        self.sigma_radius = sigma_radius

        # background. not currently used
        self.background = background

        super().__init__()

    def analyze_image(self, 
                      image: np.ndarray, 
                      auxiliary_data: Optional[dict] = None,
                     ) -> dict[str, Union[float, np.ndarray]]:

        # initial filtering
        analysis = basic_filter(image, self.hp_median, self.hp_threshold,
                                     self.denoise_cycles, self.gauss_filter, self.com_threshold)
        
        analysis.update({'filters': {}, 'metrics': {}, 'flags': {}})
        
        if self.camera_roi is None:
            analysis['filters']['roi'] = \
                np.array([0, image.shape[1] - 1, 0, image.shape[0] - 1])  # left, right, top, bottom
        else:
            # GEECS-PythonAPI uses [left, right, top, bottom], with all indices inclusive
            analysis['filters']['roi'] = np.array([self.camera_roi.left, 
                                                   self.camera_roi.right - 1, 
                                                   self.camera_roi.top, 
                                                   self.camera_roi.bottom - 1, 
                                                 ])

        # stop if low-contrast image
        def is_image_valid() -> bool:
            counts, bins = np.histogram(analysis['arrays']['blurred'],
                                        bins=max(10, 2 * int(np.std(analysis['arrays']['blurred']))))
            analysis['arrays']['histogram'] = (counts, bins)
            # most common value
            analysis['metrics']['bkg_level']: float = bins[np.where(counts == np.max(counts))[0][0]]
            analysis['metrics']['contrast']: float = self.contrast
            analysis['flags']['is_valid']: bool = np.std(bins) > self.contrast * analysis['metrics']['bkg_level']

            return analysis['flags']['is_valid']

        if not is_image_valid():
            return analysis

        try:
            blurred = analysis['arrays']['blurred']

            lr = np.round(n_sigma_window(blurred[analysis['positions']['com_ij'][0], :], self.sigma_radius)).astype(int)
            ud = np.round(n_sigma_window(blurred[:, analysis['positions']['com_ij'][1]], self.sigma_radius)).astype(int)
            analysis['metrics']['roi_com'] = np.concatenate((lr, ud))

            # max roi (left, right, top, bottom)
            lr = np.round(n_sigma_window(blurred[analysis['positions']['max_ij'][0], :], self.sigma_radius)).astype(int)
            ud = np.round(n_sigma_window(blurred[:, analysis['positions']['max_ij'][1]], self.sigma_radius)).astype(int)
            analysis['metrics']['roi_max'] = np.concatenate((lr, ud))

            # beam edges
            image_edges: np.ndarray = canny(analysis['arrays']['thresholded'], sigma=3)
            image_edges = closing(image_edges.astype(float), square(3))

            # boxed image
            if self.box:
                label_image = label(image_edges)
                areas = [box.area for box in regionprops(label_image)]
                roi = regionprops(label_image)[areas.index(max(areas))]

                if roi:
                    gain_factor: float = 0.2
                    roi = np.array([roi.bbox[1], roi.bbox[3], roi.bbox[0], roi.bbox[2]])  # left, right, top, bottom

                    width_gain = int(round((roi[1] - roi[0]) * gain_factor))
                    left_gain = min(roi[0], width_gain)
                    right_gain = min(image_edges.shape[1] - 1 - roi[1], width_gain)

                    height_gain = int(round((roi[3] - roi[2]) * gain_factor))
                    top_gain = min(roi[2], height_gain)
                    bottom_gain = min(image_edges.shape[0] - 1 - roi[3], height_gain)

                    gain_pixels = max(left_gain, right_gain, top_gain, bottom_gain)
                    left_gain = min(roi[0], gain_pixels)
                    right_gain = min(image_edges.shape[1] - 1 - roi[1], gain_pixels)
                    top_gain = min(roi[2], gain_pixels)
                    bottom_gain = min(image_edges.shape[0] - 1 - roi[3], gain_pixels)

                    analysis['metrics']['roi_edges'] = np.array([roi[0] - left_gain, roi[1] + right_gain,
                                                                      roi[2] - top_gain, roi[3] + bottom_gain])

                    pos_box = np.array(
                        [(analysis['metrics']['roi_edges'][2] + analysis['metrics']['roi_edges'][3]) / 2.,
                         (analysis['metrics']['roi_edges'][0] + analysis['metrics']['roi_edges'][1]) / 2.])
                    pos_box = np.round(pos_box).astype(int)

                    analysis['positions']['box_ij'] = tuple(pos_box)  # i, j
                    analysis['positions']['long_names'].append('box')
                    analysis['positions']['short_names'].append('box')

                    # update edges image
                    image_edges[:, :analysis['metrics']['roi_edges'][0]] = 0
                    image_edges[:, analysis['metrics']['roi_edges'][1]+1:] = 0
                    image_edges[:analysis['metrics']['roi_edges'][2], :] = 0
                    image_edges[analysis['metrics']['roi_edges'][3]+1:, :] = 0

            analysis['arrays']['edges'] = image_edges

            # ellipse fit (min_size: min of major axis, max_size: max of minor axis)
            if self.ellipse:
                h_ellipse = hough_ellipse(image_edges, min_size=10, max_size=60, accuracy=8)
                h_ellipse.sort(order='accumulator')
                best = list(h_ellipse[-1])
                yc, xc, a, b = (int(round(x)) for x in best[1:5])
                ellipse = (yc, xc, a, b, best[5])
                analysis['metrics']['ellipse'] = ellipse  # (i, j, major, minor, orientation)

                # cy, cx = ellipse_perimeter(*ellipse)
                pos_ellipse = np.array([yc, xc])

                analysis['positions']['ellipse_ij'] = tuple(pos_ellipse)  # i, j
                analysis['positions']['long_names'].append('ellipse')
                analysis['positions']['short_names'].append('ellipse')

            return analysis

        except Exception as ex:
            print(f'Failed to analyze image: {ex}')
            return analysis
