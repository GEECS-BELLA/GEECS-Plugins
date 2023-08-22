"""
Tues 8-15-2023

Unit test for the output of U_HiResMagSpec's analyze_image.  Checks the output dictionary against a simple elliptical
Gaussian with a tilt and partially clipped on the window.

IMPORTANT:  Right now I assume a static energy axis given by assuming a 825 mT field in the dipole.  In the future it
would be nice to separate this unit test from the operation of building this energy axis.

@Chris
"""

from __future__ import annotations
import unittest
import sys
import os
# import time
import numpy as np
import matplotlib.pyplot as plt

rootpath = os.path.abspath("../../../")
sys.path.insert(0, rootpath)

import ImageAnalysis.image_analysis.analyzers.U_HiResMagCam.U_HiResMagSpec as MagSpecCaller


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


class TestHiResMagSpecAnalyze(unittest.TestCase):

    def test_hiresmagspec_analyze(self):
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
        returned_image, analyzeDict, inputParams = MagSpecCaller.U_HiResMagSpecImageAnalyzer(
            noise_threshold=100,
            edge_pixel_crop=1,
            saturation_value=4095,
            normalization_factor=1,
            transverse_calibration=1,
            do_transverse_calculation=True,
            transverse_slice_threshold=0.02,
            transverse_slice_binsize=5).analyze_image(elliptical_gaussian_array)
        # print("Elapsed Time: ", time.perf_counter() - start, "s")
        #print(analyzeDict)

        # plt.imshow(elliptical_gaussian_array)
        # plt.show()

        self.assertTrue((abs(analyzeDict["Clipped-Percentage"] - 0.42678) < 1e-4)
                        and (analyzeDict["Saturation-Counts"] == 49)
                        and (abs(analyzeDict["Charge-On-Camera"] - 671992.75) < 1e-1)
                        and (abs(analyzeDict["Peak-Charge"] - 48442.74) < 1e-1)
                        and (abs(analyzeDict["Peak-Charge-Energy"] - 89.51816) < 1e-4)
                        and (abs(analyzeDict["Average-Energy"] - 89.53399) < 1e-4)
                        and (abs(analyzeDict["Energy-Spread"] - 0.09497) < 1e-4)
                        and (abs(analyzeDict["Energy-Spread-Percent"] - 0.00106076) < 1e-6)
                        and (abs(analyzeDict["Average-Beam-Size"] - 3.21847) < 1e-4)
                        and (abs(analyzeDict["Projected-Beam-Size"] - 4.07075) < 1e-4)
                        and (abs(analyzeDict["Beam-Tilt"] - 19.30100) < 1e-4)
                        and (abs(analyzeDict["Beam-Intercept"] - (-1703.46109)) < 1e-4)
                        and (abs(analyzeDict["Beam-Intercept-100MeV"] - 226.63913) < 1e-4))


if __name__ == '__main__':
    unittest.main()
