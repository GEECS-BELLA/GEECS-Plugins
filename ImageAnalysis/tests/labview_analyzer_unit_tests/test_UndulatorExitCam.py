"""
Tues 8-15-2023



@Chris
"""

from __future__ import annotations
import unittest
import numpy as np
import cv2
from image_analysis.analyzers.UC_GenericLightSpecCam import UC_LightSpectrometerCamAnalyzer
import image_analysis.labview_adapters as labview_function_caller


def generate_mirrored_gaussian_image(image_size=(256, 256), spot_std=20.0, rot_angle=2, amp=200):
    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))

    # Calculate the center of the spots
    center_x_left = image_size[0] // 4
    center_x_right = 3 * (image_size[0] // 4)
    center_y = image_size[1] // 2

    # Generate Gaussian distributions for the left and right spots
    left_spot = np.exp(-((x - center_x_left) ** 2 + (y - center_y) ** 2) / (2 * spot_std ** 2))
    right_spot = np.exp(-((x - center_x_right) ** 2 + (y - center_y) ** 2) / (2 * spot_std ** 2))

    # Combine the left and right spots to create the mirrored image
    mirrored_image = (left_spot + right_spot) * amp

    height, width = mirrored_image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rot_angle, 1.0)
    rotated_image = cv2.warpAffine(mirrored_image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

    return rotated_image


class TestUndulatorExitCamAnalyze(unittest.TestCase):

    def test_undulatorexitcam_analyze(self):
        image_size = 50
        spot_size = 2
        rotation_angle = 2
        amplitude = 200
        test_image = generate_mirrored_gaussian_image(image_size=(image_size, image_size), spot_std=spot_size,
                                                      rot_angle=-rotation_angle, amp=amplitude)

        # Make an image_analyzer using some smaller calibration numbers
        analyzer = UC_LightSpectrometerCamAnalyzer(
            noise_threshold=50,
            roi=[None, None, None, None],  # ROI(top, bottom, left, right)
            saturation_value=amplitude*0.90,
            calibration_image_tilt=rotation_angle,
            calibration_wavelength_pixel=20.,
            calibration_0th_order_pixel=37.5,
            minimum_wavelength_analysis=150.0,
            optimization_central_wavelength=420.0,
            optimization_bandwidth_wavelength=10.0
        )
        results = analyzer.analyze_image(test_image)

        return_dictionary = results['analyzer_return_dictionary']
        self.assertEqual(return_dictionary["camera_saturation_counts"], 4)
        self.assertAlmostEqual(return_dictionary["camera_total_intensity_counts"], 3921.766, delta=1e-2)
        self.assertAlmostEqual(return_dictionary["peak_wavelength_nm"], 510.0, delta=1e0)
        self.assertAlmostEqual(return_dictionary["average_wavelength_nm"], 509.79484, delta=1e-4)
        self.assertAlmostEqual(return_dictionary["wavelength_spread_weighted_rms_nm"], 25.37977, delta=1e-4)
        self.assertAlmostEqual(return_dictionary["optimization_factor"], 0.37742, delta=1e-4)

        # Check that the default config file works
        camera_name = "UC_UndulatorExitCam"
        test_default_analyzer = labview_function_caller.analyzer_from_device_type(camera_name)

        input_parameters = test_default_analyzer.build_input_parameter_dictionary()
        default_roi = input_parameters['roi_bounds_pixel']
        np.testing.assert_array_equal(np.array(default_roi), np.array([0, 1025, 0, 1281]))

        # Check that the labview wrapper function works

        test_array_shape = np.shape(test_image[default_roi[0]:default_roi[1], default_roi[2]:default_roi[3]])
        returned_image_labview, analyze_dict_labview, lineouts_labview = labview_function_caller.analyze_labview_image(
            camera_name, test_image, background=None)
        np.testing.assert_array_equal(np.shape(returned_image_labview), test_array_shape)
        np.testing.assert_array_equal(np.shape(analyze_dict_labview),
                                      np.shape(labview_function_caller.read_keys_of_interest('LightSpecCam')))
        np.testing.assert_array_equal(np.shape(lineouts_labview), np.array([2, test_array_shape[1]]))


if __name__ == '__main__':
    unittest.main()
