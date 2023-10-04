from __future__ import annotations

import unittest
from typing import NamedTuple

import numpy as np

from image_analysis.algorithms.qwlsi import QWLSIImageAnalyzer
from image_analysis import ureg, Q_

class QWLSIAnalysisTestCase(unittest.TestCase):

    def setUp(self):
        self.qia = QWLSIImageAnalyzer(reconstruction_method='velghe')

    def generate_test_interferogram(self):

        self.qia.CAMERA_RESOLUTION = Q_(4.0, 'um')
        self.qia.GRATING_CAMERA_DISTANCE = Q_(1.0, 'mm')

        x = np.arange(128) * self.qia.CAMERA_RESOLUTION
        y = np.arange(96) * self.qia.CAMERA_RESOLUTION
        x0 = 50 * self.qia.CAMERA_RESOLUTION
        y0 = 30 * self.qia.CAMERA_RESOLUTION
        x_sig = 20 * self.qia.CAMERA_RESOLUTION
        y_sig = 10 * self.qia.CAMERA_RESOLUTION
        wavefront_ampl = Q_(3000, 'nm')

        X, Y = np.meshgrid(x, y)
        self.distance_from_center_squared = np.square(X - x0) + np.square(Y - y0)
        wavefront = wavefront_ampl * np.exp(-np.square(X - x0) / (2 * x_sig**2) - np.square(Y - y0) / (2 * y_sig**2))

        grad_wavefront = NamedTuple('gradient_2d', x=np.ndarray, y=np.ndarray)(
            x = -2*(X - x0) / (2 * x_sig**2) * wavefront, 
            y = -2*(Y - y0) / (2 * y_sig**2) * wavefront, 
        )

        interferogram = sum([
            np.cos(2*np.pi * (dsc.nu_x * X + dsc.nu_y * Y)
                    - 2*np.pi * self.qia.GRATING_CAMERA_DISTANCE * (dsc.nu_x * grad_wavefront.x + dsc.nu_y * grad_wavefront.y)
                  )
            for dsc in self.qia.diffraction_spot_centers
        ]).m

        return interferogram

    def assertMaxWavefrontMinusBackgroundWithinRange(self):
        self.assertAlmostEqual(np.mean(self.reconstructed_wavefront[self.distance_from_center_squared < Q_(20.0, 'um')**2]) 
                                   - np.mean(self.reconstructed_wavefront[[0, -1],:]), 
                               Q_(2900, 'nm'),
                               delta=Q_(1000, 'nm') 
                              )

    def test_calculate_wavefront(self):

        self.reconstructed_wavefront = self.qia.calculate_wavefront(self.generate_test_interferogram())
        self.assertMaxWavefrontMinusBackgroundWithinRange()

    def test_calculate_wavefront_baffou(self):

        self.qia.reconstruction_method = 'baffou'
        self.reconstructed_wavefront = self.qia.calculate_wavefront(self.generate_test_interferogram())
        self.assertMaxWavefrontMinusBackgroundWithinRange()


if __name__ == "__main__":
    unittest.main()
