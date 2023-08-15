""" Contains UCBeamSpotScanImages class which is applicable to any cameras imaging a beam spot. 
"""

from __future__ import annotations

from typing import Union, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ....controls.devices.HTU.diagnostics.cameras.camera import Camera

import numpy as np

from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.transform import hough_ellipse
from skimage.feature import canny

from ..scans.scan_images import ScanImages, ScanData
from ....tools.images.filtering import basic_filter, FiltersParameters
from ....tools.images.spot import n_sigma_window
from ....controls.interface import api_error

class UC_BeamSpotScanImages(ScanImages):
    """ Implements image analysis for any e-beam or laser beam spot image

        Used for undulator Visa, Aline, and Phosphor cameras

    """
    def __init__(self, scan: ScanData, camera: Union[int, Camera, str]):
        """
        Parameters
        ----------
        sigma_radius : float 
            used in n_sigma_window function

        """
        super().__init__(scan, camera)

    def actually_analyze_image(self, image: np.ndarray, filtering=FiltersParameters(), sigma_radius: float = 5):
        """
        Parameters
        ----------
        filtering : FiltersParameters
        sigma_radius : float

        """

        # initial filtering
        self.analysis = basic_filter(image, self.analysis, filtering.hp_median, filtering.hp_threshold,
                                     filtering.denoise_cycles, filtering.gauss_filter, filtering.com_threshold)
        if self.camera_roi is None:
            self.analysis['filters']['roi'] = \
                np.array([0, image.shape[1] - 1, 0, image.shape[0] - 1])  # left, right, top, bottom
        else:
            self.analysis['filters']['roi'] = self.camera_roi.copy()

        # stop if low-contrast image
        if not self.is_image_valid(filtering.contrast):
            return

        try:
            blurred = self.analysis['arrays']['blurred']

            lr = np.round(n_sigma_window(blurred[self.analysis['positions']['com_ij'][0], :], sigma_radius)).astype(int)
            ud = np.round(n_sigma_window(blurred[:, self.analysis['positions']['com_ij'][1]], sigma_radius)).astype(int)
            self.analysis['metrics']['roi_com'] = np.concatenate((lr, ud))

            # max roi (left, right, top, bottom)
            lr = np.round(n_sigma_window(blurred[self.analysis['positions']['max_ij'][0], :], sigma_radius)).astype(int)
            ud = np.round(n_sigma_window(blurred[:, self.analysis['positions']['max_ij'][1]], sigma_radius)).astype(int)
            self.analysis['metrics']['roi_max'] = np.concatenate((lr, ud))

            # beam edges
            image_edges: np.ndarray = canny(self.analysis['arrays']['thresholded'], sigma=3)
            image_edges = closing(image_edges.astype(float), square(3))

            # boxed image
            if filtering.box:
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

                    self.analysis['metrics']['roi_edges'] = np.array([roi[0] - left_gain, roi[1] + right_gain,
                                                                      roi[2] - top_gain, roi[3] + bottom_gain])

                    pos_box = np.array(
                        [(self.analysis['metrics']['roi_edges'][2] + self.analysis['metrics']['roi_edges'][3]) / 2.,
                         (self.analysis['metrics']['roi_edges'][0] + self.analysis['metrics']['roi_edges'][1]) / 2.])
                    pos_box = np.round(pos_box).astype(int)

                    self.analysis['positions']['box_ij'] = tuple(pos_box)  # i, j
                    self.analysis['positions']['long_names'].append('box')
                    self.analysis['positions']['short_names'].append('box')

                    # update edges image
                    image_edges[:, :self.analysis['metrics']['roi_edges'][0]] = 0
                    image_edges[:, self.analysis['metrics']['roi_edges'][1]+1:] = 0
                    image_edges[:self.analysis['metrics']['roi_edges'][2], :] = 0
                    image_edges[self.analysis['metrics']['roi_edges'][3]+1:, :] = 0

            self.analysis['arrays']['edges'] = image_edges

            # ellipse fit (min_size: min of major axis, max_size: max of minor axis)
            if filtering.ellipse:
                h_ellipse = hough_ellipse(image_edges, min_size=10, max_size=60, accuracy=8)
                h_ellipse.sort(order='accumulator')
                best = list(h_ellipse[-1])
                yc, xc, a, b = (int(round(x)) for x in best[1:5])
                ellipse = (yc, xc, a, b, best[5])
                self.analysis['metrics']['ellipse'] = ellipse  # (i, j, major, minor, orientation)

                # cy, cx = ellipse_perimeter(*ellipse)
                pos_ellipse = np.array([yc, xc])

                self.analysis['positions']['ellipse_ij'] = tuple(pos_ellipse)  # i, j
                self.analysis['positions']['long_names'].append('ellipse')
                self.analysis['positions']['short_names'].append('ellipse')

        except Exception as ex:
            if isinstance(image, Path):
                image_name = f' {image.name}'
            else:
                image_name = ''
            api_error.error(str(ex), f'Failed to analyze image{image_name}')
            pass
