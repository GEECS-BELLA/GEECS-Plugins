"""
Wednesday 9-6-2023

Takes a few images from the UndulatorExitCam with the blue beam diode aligned on it.  Then, with the 0th and 1st order
light visible on the image, find the locations of the two peaks, the tilt of the image, and the nm/pixel calibration
of the tilt-corrected image.

@Chris
"""

import numpy as np
import sys
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path
from image_analysis.utils import read_imaq_png_image


def find_two_peaks(input_image):
    left_image = input_image[:, 0:int(np.shape(input_image)[1] / 2)]
    right_image = input_image[:, int(np.shape(input_image)[1] / 2) + 1:-1]

    left_y_size, left_x_size = np.shape(left_image)
    left_x_projection = np.sum(left_image, axis=0)
    left_y_projection = np.sum(left_image, axis=1)
    left_x_loc = np.average(np.arange(0, left_x_size), weights=left_x_projection)
    left_y_loc = np.average(np.arange(0, left_y_size), weights=left_y_projection)

    right_y_size, right_x_size = np.shape(right_image)
    right_x_projection = np.sum(right_image, axis=0)
    right_y_projection = np.sum(right_image, axis=1)
    right_x_loc = np.average(np.arange(0, right_x_size), weights=right_x_projection)
    right_y_loc = np.average(np.arange(0, right_y_size), weights=right_y_projection)

    right_x_loc = right_x_loc + int(np.shape(input_image)[1] / 2)
    return left_x_loc, left_y_loc, right_x_loc, right_y_loc


# Load the image
camera = "UVCam"  # "UVCam"  # "ExitCam"
if camera == "ExitCam":
    sample_data_path = "Z:/data/Undulator/Y2023/09-Sep/23_0906/auxiliary data/"
    sample_filename = "UC_UndulatorExitCam_"
    # sample_shot_number = 11  # 11 - 34
    sample_extension = ".png"

    threshold = 100
    wavelength = 450  # nm
    min_wavelength = 150

    shot_array = np.arange(11, 34 + 1)
    # shot_array = np.array([11])

    zero_side_left = False

    top = 0
    bot = 1025
    left = 0
    right = 1281

elif camera == "UCRad2":
    sample_data_path = "Z:/data/Undulator/Y2024/05-May/24_0502/scans/Scan002/UC_UndulatorRad2/"
    sample_filename = "Scan002_UC_UndulatorRad2_"
    # sample_shot_number = 11  # 11 - 34
    sample_extension = ".png"

    threshold = 60
    wavelength = 405  # 450  # nm
    min_wavelength = 150

    shot_array = np.arange(1, 20 + 1)
    #shot_array = np.array([1])

    zero_side_left = True

    top = 1000
    bot = 1600
    left = 0
    right = 1500

elif camera == "UVCam":
    sample_data_path = "Z:/data/Undulator/Y2024/05-May/24_0508/scans/Scan001/UC_PostUndulatorUVSpecCam/"
    sample_filename = "Scan001_UC_PostUndulatorUVSpecCam_"
    # sample_shot_number = 11  # 11 - 34
    sample_extension = ".png"

    threshold = 10
    wavelength = 405  # 450  # nm
    min_wavelength = 150

    shot_array = np.arange(1, 20 + 1)
    #shot_array = np.array([1])

    zero_side_left = True

    top = 1200
    bot = 2000
    left = 000
    right = 2800

elif camera == "UCRad2_OLD":
    sample_data_path = "Z:/data/Undulator/Y2024/03-Mar/24_0314/scans/Scan004/UC_UndulatorRad2/"
    sample_filename = "Scan004_UC_UndulatorRad2_"
    # sample_shot_number = 11  # 11 - 34
    sample_extension = ".png"

    threshold = 100
    wavelength = 450  # nm
    min_wavelength = 150

    shot_array = np.arange(1, 20 + 1)
    #shot_array = np.array([1])

    zero_side_left = True

    top = 600
    bot = 1100
    left = 650
    right = 2100

elif camera == "UVCam_OLD":
    sample_data_path = "Z:/data/Undulator/Y2024/04-Apr/24_0423/scans/Scan009/UC_PostUndulatorUVSpecCam/"
    sample_filename = "Scan009_UC_PostUndulatorUVSpecCam_"
    # sample_shot_number = 11  # 11 - 34
    sample_extension = ".png"

    threshold = 10
    wavelength = 450  # nm
    min_wavelength = 150

    shot_array = np.arange(1, 20 + 1)
    #shot_array = np.array([1])

    zero_side_left = True

    top = 800
    bot = 1100
    left = 0
    right = 2800
else:
    print("Need a valid camera")
    sys.exit()

num_shots = len(shot_array)
tilt_values = np.zeros(num_shots)
calibration_values = np.zeros(num_shots)
zeroth_values = np.zeros(num_shots)

do_plot = False
rotated_image = []
for i in range(num_shots):
    sample_shot_number = shot_array[i]
    fullpath = sample_data_path + sample_filename + '{:03d}'.format(sample_shot_number) + sample_extension

    raw_image = read_imaq_png_image(Path(fullpath))*1.0

    roi = [top, bot, left, right]

    cropped_image = raw_image[roi[0]:roi[1], roi[2]:roi[3]]

    if do_plot:
        print("Raw Image: ", np.shape(raw_image))
        plt.imshow(raw_image)
        plt.title("Raw")
        plt.show()

        print("Cropped Image: ", np.shape(cropped_image))
        plt.imshow(cropped_image)
        plt.title("Cropped")
        plt.show()

    # Threshold so that we only see the peaks

    threshold_image = np.copy(cropped_image) - threshold
    threshold_image[np.where(threshold_image < 0)] = 0

    # Find the two peaks

    raw_left_x, raw_left_y, raw_right_x, raw_right_y = find_two_peaks(threshold_image)

    if do_plot:
        print("Threshold Image: ", np.shape(raw_image))
        plt.imshow(threshold_image)
        plt.scatter([raw_left_x, raw_right_x], [raw_left_y, raw_right_y], c='r', alpha=0.5)
        plt.title("Raw")
        plt.show()

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

    if do_plot:
        print("Rotated Image: ", np.shape(rotated_image))
        plt.imshow(rotated_image)
        plt.scatter([rot_left_x, rot_right_x], [rot_left_y, rot_right_y], c='r', alpha=0.5)
        plt.title("Rotated")
        plt.show()

    x_separation = np.abs(rot_left_x - rot_right_x)
    calibration = wavelength/x_separation
    # print("* Calibration Factor:", calibration, "nm/pixel")
    # print("* 0th Order Location:", rot_right_x, "pixel")

    tilt_values[i] = tilt_angle
    if zero_side_left:
        zeroth_values[i] = rot_left_x
        calibration = calibration*-1
    else:
        zeroth_values[i] = rot_right_x
    calibration_values[i] = calibration

"""
print("Rotated Image: ", np.shape(rotated_image))
plt.imshow(rotated_image)
plt.title("Rotated")
plt.show()
"""

calibration_0th_order = np.average(zeroth_values)
calibration_um_per_pixel = np.average(calibration_values)
print("* Calibrated Tilt Angle    :", np.average(tilt_values), "+/-", np.std(tilt_values))
print("* Calibration Factor       :", calibration_um_per_pixel, "+/-", np.std(calibration_values))
print("* Calibrated 0th Order Loc.:", calibration_0th_order, "+/-", np.std(zeroth_values))


# Perform a projection

spectra_projection = np.sum(rotated_image, axis=0)

# Plot out the sample spectrum

pixel_axis = np.arange(0, np.shape(rotated_image)[1])
wavelength_axis = (pixel_axis - calibration_0th_order) * -calibration_um_per_pixel

crop_bounds = np.where(wavelength_axis > min_wavelength)[0]
crop_wavelength_axis = wavelength_axis[crop_bounds]
crop_spectra_projection = spectra_projection[crop_bounds]

plt.plot(wavelength_axis, spectra_projection)
plt.plot(crop_wavelength_axis, crop_spectra_projection)
plt.title(f"{camera} Spectra")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Amplitude (arb.)")
plt.show()
