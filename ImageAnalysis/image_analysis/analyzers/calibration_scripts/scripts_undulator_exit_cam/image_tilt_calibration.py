"""
Wednesday 9-6-2023

Takes a few images from the UndulatorExitCam with the blue beam diode aligned on it.  Then, with the 0th and 1st order
light visible on the image, find the locations of the two peaks, the tilt of the image, and the nm/pixel calibration
of the tilt-corrected image.

@Chris
"""

import sys
import os
import time
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

rootpath = os.path.abspath("../../../../")
sys.path.insert(0, rootpath)
import image_analysis.analyzers.calibration_scripts.modules_image_processing.pngTools as pngTools


def find_two_peaks(input_image):
    left_image = input_image[:, 0:int(np.shape(input_image)[1] / 2)]
    right_image = input_image[:, int(np.shape(input_image)[1] / 2) + 1:-1]

    left_ysize, left_xsize = np.shape(left_image)
    left_x_projection = np.sum(left_image, axis=0)
    left_y_projection = np.sum(left_image, axis=1)
    left_x_loc = np.average(np.arange(0, left_xsize), weights=left_x_projection)
    left_y_loc = np.average(np.arange(0, left_ysize), weights=left_y_projection)

    right_ysize, right_xsize = np.shape(right_image)
    right_x_projection = np.sum(right_image, axis=0)
    right_y_projection = np.sum(right_image, axis=1)
    right_x_loc = np.average(np.arange(0, right_xsize), weights=right_x_projection)
    right_y_loc = np.average(np.arange(0, right_ysize), weights=right_y_projection)

    right_x_loc = right_x_loc + int(np.shape(input_image)[1] / 2)
    return left_x_loc, left_y_loc, right_x_loc, right_y_loc


# Load the image

sample_data_path = "Z:/data/Undulator/Y2023/09-Sep/23_0906/auxiliary data/"
sample_filename = "UC_UndulatorExitCam_"
# sample_shot_number = 11  # 11 - 34
sample_extension = ".png"

threshold = 100
wavelength = 450  # nm
min_wavelength = 150

shot_array = np.arange(11, 34 + 1)
# shot_array = np.array([11])
num_shots = len(shot_array)
tilt_values = np.zeros(num_shots)
calibration_values = np.zeros(num_shots)
zeroth_values = np.zeros(num_shots)

for i in range(num_shots):
    sample_shot_number = shot_array[i]
    fullpath = sample_data_path + sample_filename + '{:03d}'.format(sample_shot_number) + sample_extension
    raw_image = np.array(pngTools.nBitPNG(fullpath))

    """
    print("Raw Image: ", np.shape(raw_image))
    plt.imshow(raw_image)
    plt.title("Raw")
    plt.show()
    """

    # Threshold so that we only see the peaks

    threshold_image = np.copy(raw_image) - threshold
    threshold_image[np.where(threshold_image < 0)] = 0

    # Find the two peaks

    raw_left_x, raw_left_y, raw_right_x, raw_right_y = find_two_peaks(threshold_image)

    # Calculate the tilt

    # print(raw_left_y-raw_right_y)
    # print("* Calibrated Tilt Angle:")
    tilt_radians = np.arctan((raw_left_y - raw_right_y)/(raw_left_x - raw_right_x))
    # print(tilt_radians, "radians")
    tilt_angle = tilt_radians*180/np.pi
    # print(tilt_angle, "degrees")

    # Tilt the image

    rotated_image = ndimage.rotate(threshold_image, tilt_angle, reshape=False)

    # With the tilt, find the calibration

    rot_left_x, rot_left_y, rot_right_x, rot_right_y = find_two_peaks(rotated_image)
    x_separation = np.abs(rot_left_x - rot_right_x)
    calibration = wavelength/x_separation
    # print("* Calibration Factor:", calibration, "nm/pixel")
    # print("* 0th Order Location:", rot_right_x, "pixel")

    tilt_values[i] = tilt_angle
    calibration_values[i] = calibration
    zeroth_values[i] = rot_right_x

"""
print("Rotated Image: ", np.shape(rotated_image))
plt.imshow(rotated_image)
plt.title("Rotated")
plt.show()
"""

print("* Calibrated Tilt Angle    :", np.average(tilt_values), "+/-", np.std(tilt_values))
print("* Calibration Factor       :", np.average(calibration_values), "+/-", np.std(calibration_values))
print("* Calibrated 0th Order Loc.:", np.average(zeroth_values), "+/-", np.std(zeroth_values))


# Perform a projection

spectra_projection = np.sum(rotated_image, axis=0)

# Plot out the sample spectrum

pixel_axis = np.arange(0, np.shape(rotated_image)[1])
wavelength_axis = (pixel_axis - rot_right_x) * -calibration

crop_bounds = np.where(wavelength_axis > min_wavelength)[0]
crop_wavelength_axis = wavelength_axis[crop_bounds]
crop_spectra_projection = spectra_projection[crop_bounds]

plt.plot(wavelength_axis, spectra_projection)
plt.plot(crop_wavelength_axis, crop_spectra_projection)
plt.title("UC_UndulatorExitCam Spectra")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Amplitude (arb.)")
plt.show()
