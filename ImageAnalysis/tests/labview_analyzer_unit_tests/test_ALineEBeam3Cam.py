"""
This tests the full functionality of the ALineEBeam analyzer using the settings for AlineEBeam3.  Later on, if 1 and 2
are implemented then

@Chris
"""

from __future__ import annotations
import unittest
import numpy as np
import matplotlib.pyplot as plt
import cv2
from image_analysis.analyzers.UC_ALineEBeamCam import UC_ALineEBeamCamAnalyzer
import image_analysis.labview_adapters as labview_function_caller


def generate_gaussian_image(image_size=(256, 256), spot_std=20.0, amp=200):
    rows, cols = image_size
    x = np.linspace(-int(.5 * cols), int(.5 * cols), cols)
    y = np.linspace(-int(.5 * rows), int(.5 * rows), rows)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(np.square(xx) + np.square(yy))
    gaussian = amp * np.exp(-0.5 * np.square(r / spot_std))
    return gaussian


class TestUndulatorExitCamAnalyze(unittest.TestCase):

    def test_undulatorexitcam_analyze(self):
        image_size = 60
        spot_size = 5
        amplitude = 200
        test_image = generate_gaussian_image(image_size=(image_size, image_size), spot_std=spot_size, amp=amplitude)

        # Make an analyzer using some set parameters
        analyzer = UC_ALineEBeamCamAnalyzer(
            noise_threshold=10,
            circular_crop_center_x=25,
            circular_crop_center_y=25,
            circular_crop_radius=4,
            saturation_value=190,
            spatial_calibration=24.4,
        )
        results = analyzer.analyze_image(test_image)

        return_dictionary = results['analyzer_return_dictionary']
        self.assertEqual(return_dictionary["camera_saturation_counts"], 4)
        self.assertAlmostEqual(return_dictionary["camera_total_intensity_counts"], 3598.974, delta=1e-2)
        self.assertAlmostEqual(return_dictionary["peak_intensity_counts"], 167.75, delta=1e-1)
        self.assertAlmostEqual(return_dictionary["centroid_x_um"], 628.34, delta=1e-1)
        self.assertAlmostEqual(return_dictionary["centroid_y_um"], 628.34, delta=1e-1)
        self.assertAlmostEqual(return_dictionary["fwhm_x_um"], 97.6, delta=1e0)
        self.assertAlmostEqual(return_dictionary["fwhm_y_um"], 97.6, delta=1e0)

        # Check that the default config file works
        camera_name = "UC_ALineEBeam3"
        test_default_analyzer = labview_function_caller.analyzer_from_device_type(camera_name)
        input_parameters = test_default_analyzer.build_input_parameter_dictionary()
        default_roi = input_parameters['roi_bounds_pixel']
        self.assertListEqual(list(default_roi), [0, 950, 50, 1000])

        # Check that the labview wrapper function works
        test_array_shape = np.shape(test_image[default_roi[0]:default_roi[1], default_roi[2]:default_roi[3]])
        returned_image_labview, analyze_dict_labview, lineouts_labview = labview_function_caller.analyze_labview_image(
            camera_name, test_image, background=None)
        np.testing.assert_array_equal(np.shape(returned_image_labview), test_array_shape)
        np.testing.assert_array_equal(np.shape(analyze_dict_labview),
                                      np.shape(labview_function_caller.read_keys_of_interest('ALineCam')))
        np.testing.assert_array_equal(np.array(lineouts_labview), np.array([0.]))


if __name__ == '__main__':
    unittest.main()
