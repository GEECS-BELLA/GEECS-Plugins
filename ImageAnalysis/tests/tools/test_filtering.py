from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from image_analysis.tools import filtering

class TestFiltering(unittest.TestCase):
    image_folder = Path(__file__).parent / 'data'
    
    def test_clip_hot_pixels(self):
        test_image = np.array([
            [0.0,  0.1, 0.1],
            [0.1, 10.0, 0.2],
            [0.0,  0.3, 0.4],
        ])
        
        clipped_image = filtering.clip_hot_pixels(test_image)
        self.assertLess(clipped_image[1,1], 1.0)

    def test_denoise(self):
        # called in basic_filter()
        pass

    def test_basic_filter(self):
        # check that it outputs a dict, and that the blurred image is the right
        # size
        test_image = np.array([
            [0.0,   0.1, 0.1],
            [0.1,  -0.4, 0.2],
            [0.0,   0.3, 0.4],
        ])

        filter_result: dict[str, dict] = filtering.basic_filter(test_image)
        self.assertSetEqual(set(filter_result.keys()), {'filter_pars', 'positions', 'arrays'})
        self.assertEqual(filter_result['arrays']['blurred'].shape, (3, 3))
