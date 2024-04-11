"""
Since this analyzer is functionally the same as UndulatorExitCam, here I am only testing that the labview adapters
module correctly initializes the right analyzer.  The default ROI settings for this camera makes it difficult to do
a typical small image.

Point being, if this test fails and ExitCam fails, it's a class initialization problem.  If this fails while ExitCam
passes, then it's a labview_adapters problem.  If this test passes and ExitCam fails, it's an analyzer problem.

-Chris
"""
from __future__ import annotations
import unittest
from image_analysis import labview_adapters


class TestUndulatorExitCamAnalyze(unittest.TestCase):
    def test_undulatorrad2cam_initialize(self):
        camera_name = "UC_UndulatorRad2"
        default_analyzer = labview_adapters.analyzer_from_device_type(camera_name)
        input_parameters = default_analyzer.build_input_parameter_dictionary()

        self.assertEqual(len(input_parameters), 9)
        self.assertListEqual(list(input_parameters['roi_bounds_pixel']), [600, 1100, 650, 2100])


if __name__ == '__main__':
    unittest.main()
