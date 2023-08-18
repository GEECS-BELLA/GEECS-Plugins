from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from image_analysis.analyzers.UC_BeamSpot import UC_BeamSpotImageAnalyzer
from image_analysis.utils import ROI

class TestUC_BeamSpot(unittest.TestCase):
    image_folder = Path(__file__).parent / 'data'
    
    def generate_gaussian_test_image(self, 
                                     shape: tuple[int, int], 
                                     ampl, 
                                     x0, sigma_x,
                                     y0, sigma_y,
                                    ):
        
        X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        return ampl * np.exp(-np.square(X - x0) / (2 * sigma_x**2) - np.square(Y - y0) / (2 * sigma_y**2))

    def test_analyze_image(self):
        image_analyzer = UC_BeamSpotImageAnalyzer(camera_roi=ROI(0, 12, 0, 16))
        image = self.generate_gaussian_test_image((12, 16), 1.0, 4.3, 2.3, 8.3, 0.5)
        image_analysis = image_analyzer.analyze_image(image)
        # at this point, outputs a dictionary, not consistent with current 
        # ImageAnalysis workflow. TODO: align these
        self.assertSetEqual(set(image_analysis.keys()), 
                            {'filter_pars', 'positions', 'arrays', 
                             'filters', 'metrics', 'flags',
                            }
                           )
        # y position of center of mass
        self.assertAlmostEqual(image_analysis['positions']['com_ij'][0], 8, delta=1)
        # x position of center of mass
        self.assertAlmostEqual(image_analysis['positions']['com_ij'][1], 4, delta=1)
        # ellipse major
        # self.assertAlmostEqual(image_analysis['metrics']['ellipse'][2], 2.3, delta=0.2)

if __name__ == '__main__':
    unittest.main()
