from __future__ import annotations

import unittest
from pathlib import Path
from itertools import chain, cycle

import numpy as np

from image_analysis.utils import read_imaq_image

class TestReadIMAQImage(unittest.TestCase):
    image_folder = Path(__file__).parent / 'data'
    
    def test_tif_image(self):
        pass

    def test_regular_png_image(self):
        """ A basic png image.
        
        From http://www.schaik.com/pngsuite/
        """
        img = read_imaq_image(self.image_folder / 'basn0g08.png')
        def generate_ref_image():
            it = cycle(chain(range(0, 256), range(254, 0, -1)))
            for c in range(32*32):
                yield next(it)
        ref_img = np.reshape(np.fromiter(generate_ref_image(), int), (32, 32))
        self.assertTrue((img == ref_img).all())

    def test_NI_png_image(self):
        """ An image with a weird number of significant bits
        
        From http://www.schaik.com/pngsuite/
        """
        # TODO: create NI test png
        pass

if __name__ == "__main__":
    unittest.main()