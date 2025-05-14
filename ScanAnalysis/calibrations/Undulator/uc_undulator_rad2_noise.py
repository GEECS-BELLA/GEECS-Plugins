"""
Takes a noscan on an empty UC_Rad2 camera and finds a smooth background.  This is to account for the fact that the noise
level changes across the Rad2 image.  Saves a npy file that can be loaded by an analyzer.

-Chris
"""

from geecs_data_utils import ScanData
from image_analysis.utils import read_imaq_png_image
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter

do_plot = False

tag = ScanData.get_scan_tag(year=2025, month=3, day=12, number=2, experiment='Undulator')

folder = ScanData.get_device_shot_path(tag=tag, device_name="UC_UndulatorRad2", shot_number=1).parent
total_shots = len(list(folder.iterdir()))

found_shots = 0
average_image = None
for i in range(total_shots):
    print("Shot", i+1)
    image_file = ScanData.get_device_shot_path(tag=tag, device_name="UC_UndulatorRad2", shot_number=i+1)
    if image_file.exists():
        found_shots += 1
        loaded_image = read_imaq_png_image(image_file) * 1.0
        loaded_image = median_filter(loaded_image, size=3)
        if average_image is None:
            average_image = loaded_image
        else:
            average_image += loaded_image

average_image /= found_shots

if do_plot:
    plt.imshow(average_image, vmin=0, vmax=5)
    plt.show()

image = average_image
filters = [('M', 3), ('G', 3), ('G', 5)]
for kind, size in filters:
    if kind == 'G':
        image = gaussian_filter(image, sigma=size)
    elif kind == 'M':
        image = median_filter(image, size=3)

if do_plot:
    plt.imshow(image, vmin=0, vmax=5)
    plt.show()

current_directory = Path(__file__).parent
np.save(current_directory / 'rad2_background.npy', np.array(image))
print("Done!")
