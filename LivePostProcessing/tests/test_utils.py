import unittest
from image_analysis.utils import read_imaq_image
from livepostprocessing.utils import find_undulator_folder
import numpy as np

class TestReadIMAQImage(unittest.TestCase):
    def test_get_undulator_folder(self):
        undulator_folder = find_undulator_folder()
        self.assertEqual(undulator_folder.parts[-2:], ('data', 'Undulator'))
    
    def test_read_png_image(self):
        """ A NI IMAQ png image has a special encoding. test that read_imaq_image()
            handles this correctly.
        
        """
        image_filename = (find_undulator_folder() / 'Y2022' / '09-Sep' / '22_0929' / 'scans' / 'Scan027' / 
                          'UC_GhostUpstream' / 'Scan027_UC_GhostUpstream_178.png'
                         )

        image = read_imaq_image(image_filename)
        image_values = np.sort(np.unique(image))

        # we expect that the values are 0, 1, 2, ... 
        self.assertTrue(np.all(image_values[:10] == np.arange(10)))
        self.assertLess(image.max(), 256)

    def test_read_tif_image(self):
        pass

    def test_write_png_image(self):
        pass

if __name__ == "__main__":
    unittest.main()
