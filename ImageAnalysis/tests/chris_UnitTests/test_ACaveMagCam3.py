"""
Tues 8-15-2023

Unit test for the output of UC_GenericMagSpecCam's analyze_image for the ACaveMagCam3 Camera.  Checks the output
dictionary against a simple elliptical Gaussian with a tilt and partially clipped on the window.

Much of this is the same as the HiResMagSpec's analyze test, but this makes sure that the energy axis is loaded correct
when 'acave3' is used, as well as the labview wrapper functions.

@Chris
"""

from __future__ import annotations
import unittest
import numpy as np

# import sys
# import os
# rootpath = os.path.abspath("../../")
# sys.path.insert(0, rootpath)

import image_analysis.analyzers.default_analyzer_initialization as default_analyzer
import image_analysis.analyzers.UC_GenericMagSpecCam as mag_spec_caller
import image_analysis.labview_adapters as labview_function_caller


def generate_elliptical_gaussian(amplitude, height, width, center_x, center_y, sigma_x, sigma_y, angle_deg):
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    x, y = np.meshgrid(x, y)

    # Apply rotation to the coordinates
    angle_rad = np.radians(angle_deg)
    x_rot = (x - center_x) * np.cos(angle_rad) - (y - center_y) * np.sin(angle_rad) + center_x
    y_rot = (x - center_x) * np.sin(angle_rad) + (y - center_y) * np.cos(angle_rad) + center_y

    # Calculate the Gaussian function
    exponent = -((x_rot - center_x) ** 2 / (2 * sigma_x ** 2) + (y_rot - center_y) ** 2 / (2 * sigma_y ** 2))
    gaussian = amplitude * np.exp(exponent)

    return gaussian


class TestACaveMagCam3Analyze(unittest.TestCase):

    def test_acavemagcam3_analyze(self):
        # Define parameters for the elliptical Gaussian
        amplitude = 6000
        height = 50
        width = 50
        center_x = 40
        center_y = 25
        sigma_x = 7
        sigma_y = 3
        angle_deg = 30

        # start = time.perf_counter()
        # Generate the elliptical Gaussian array
        elliptical_gaussian_array = generate_elliptical_gaussian(amplitude, height, width, center_x, center_y, sigma_x,
                                                                 sigma_y, angle_deg)
        # print("Elapsed Time: ", time.perf_counter() - start, "s")

        # start = time.perf_counter()
        test_analyzer = mag_spec_caller.UC_GenericMagSpecCamAnalyzer(
            mag_spec_name='acave3',
            roi=[1, -1, 1, -1],
            noise_threshold=100,  # CONFIRM IF THIS WORKS
            saturation_value=4095,
            normalization_factor=1,
            transverse_calibration=1,
            do_transverse_calculation=True,
            transverse_slice_threshold=0.02,
            transverse_slice_binsize=5,
            optimization_central_energy=93.0,
            optimization_bandwidth_energy=2.0)
        results = test_analyzer.analyze_image(elliptical_gaussian_array)
        # results = mag_spec_caller.return_default_hi_res_mag_cam_analyzer().analyze_image(elliptical_gaussian_array)
        # print("Elapsed Time: ", time.perf_counter() - start, "s")
        # print(analyze_dict)

        # plt.imshow(elliptical_gaussian_array)
        # plt.show()

        # Here I check that the mag spec analyzer is working properly using the constants set above with the sample data

        analyze_dict = results['analyzer_return_dictionary']
        self.assertAlmostEqual(analyze_dict["camera_clipping_factor"], 0.42678, delta=1e-4)
        self.assertEqual(analyze_dict["camera_saturation_counts"], 49)
        self.assertAlmostEqual(analyze_dict["total_charge_pC"], 671992.75, delta=1e-1)
        self.assertAlmostEqual(analyze_dict["peak_charge_pc/MeV"], 48442.74, delta=1e-1)
        self.assertAlmostEqual(analyze_dict["peak_charge_energy_MeV"], 82.8332077, delta=1e-4)
        self.assertAlmostEqual(analyze_dict["weighted_average_energy_MeV"], 82.90576452951896, delta=1e-4)
        self.assertAlmostEqual(analyze_dict["energy_spread_weighted_rms_MeV"], 0.4254417441866327, delta=1e-4)
        self.assertAlmostEqual(analyze_dict["energy_spread_percent"], 0.5131630431260932, delta=1e-4)
        self.assertAlmostEqual(analyze_dict["weighted_average_beam_size_um"], 3.21847, delta=1e-4)
        self.assertAlmostEqual(analyze_dict["projected_beam_size_um"], 4.07075, delta=1e-4)
        self.assertAlmostEqual(analyze_dict["beam_tilt_um/MeV"], 4.3162045059, delta=1e-4)
        self.assertAlmostEqual(analyze_dict["beam_tilt_intercept_um"], -333.19934, delta=1e-4)
        self.assertAlmostEqual(analyze_dict["beam_tilt_intercept_100MeV_um"], 98.42110, delta=1e-4)
        self.assertAlmostEqual(analyze_dict["optimization_factor"], 3.46646, delta=1e-4)

        # Here I am only checking that the labview wrapper function is working properly by checking the output shapes

        test_default_analyzer = default_analyzer.return_default_acave_mag_cam3_analyzer()
        input_parameters = test_default_analyzer.build_input_parameter_dictionary()
        default_roi = input_parameters['roi_bounds_pixel']
        # print(default_roi)
        np.testing.assert_array_equal(np.array(default_roi), np.array([112, 278, 23, 1072]))

        # I have to make a really strange shape in order for the large bounds above to work with the small image below

        camera_name = "UC_ACaveMagCam3"
        big_elliptical_gaussian = generate_elliptical_gaussian(amplitude, 113, 24, center_x, center_y, sigma_x,
                                                               sigma_y, angle_deg)
        test_array_shape = np.shape(big_elliptical_gaussian[default_roi[0]:default_roi[1],
                                    default_roi[2]:default_roi[3]])
        returned_image_labview, analyze_dict_labview, lineouts_labview = labview_function_caller.analyze_labview_image(
            camera_name, big_elliptical_gaussian, background=None)
        np.testing.assert_array_equal(np.shape(returned_image_labview), test_array_shape)
        np.testing.assert_array_equal(np.shape(analyze_dict_labview), np.array([14, ]))
        np.testing.assert_array_equal(np.shape(lineouts_labview), np.array([2, test_array_shape[1]]))


if __name__ == '__main__':
    unittest.main()
