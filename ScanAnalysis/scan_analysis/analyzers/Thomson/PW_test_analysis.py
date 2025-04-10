# -*- coding: utf-8 -*-
"""
PW Analysis for testing camera in control room
Child to ScanAnalysis (./scan_analysis/base.py)
"""
# %% imports

from __future__ import annotations
import sys
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
sys.path.insert(0, 'C:\\GEECS\\Developers Version\\source\\GEECS-Plugins\\ScanAnalysis') # Without this path the code outputs ModuleNotFoundError: No module named scan_analysis
print("PW_test_analysis.py line 18")
from scan_analysis.base import ScanAnalysis
print("PW_test_analysis.py line 20")
from image_analysis.utils import read_imaq_png_image
from image_analysis.analyzers.online_analysis_modules.image_processing_funcs import threshold_reduction
sys.path.insert(0, 'C:\\GEECS\\Developers Version\\source\\GEECS-Plugins\\GEECS-PythonAPI')

from geecs_python_api.analysis.scans.scan_data import ScanData

# %% classes
class PWTestAnalysis(ScanAnalysis):
    def __init__(self, scan_tag: ScanTag, device_name: Optional[str] = None, skip_plt_show: bool = True):
        super().__init__(scan_tag, device_name=None, skip_plt_show=skip_plt_show)

        # self.device_list = ['CAM-HPD-MultiPlane1']
        self.device_list = ['CAM-CR-Beampointing1']
        # self.background_tag = ScanData.get_scan_tag(year=2025, month=3, day=28, number=2, experiment='ControlRoom')
        self.backgrounds = {}

        # Check if data directory exists and is not empty
        for device in self.device_list:
            device_path = self.scan_directory / device
            print("PW_test_anlaysis.py line 41 device_path is", device_path)

            if not device_path.exists() or not any(device_path.iterdir()):
                msg = f"Data directory 'device_path' does not exist or is empty."
                logging.warning(msg)
                raise NotADirectoryError(msg)

        self.save_path = self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / "ScAnalyzer"
        # self.load_backgrounds()
        self.max_counts = 0
        self.mean_counts = 0
        self.sum_counts = 0

    # def load_backgrounds(self):
    #     """ From the background tag, loads 10 images from each device and averages them.  Saved into a dict """
    #     background_shots = list(range(1, 11))
    #     for device in self.device_list:
    #         average_image = None
    #         for shot_num in background_shots:
    #             image_file = ScanData.get_device_shot_path(tag=self.background_tag, device_name=device,
    #                                                        shot_number=shot_num)
    #             image = read_imaq_png_image(image_file) * 1.0
    #             average_image = image if average_image is None else average_image + image
    #         average_image /= 10
    #         self.backgrounds[device] = average_image

    # def run_analysis(self, config_options: Optional[str] = None):
    #     """ Main function to run the analysis and generate plots. """
    #     # For HTU we grab these from files
    #     # energy_values, charge_density_matrix

    #     # For each shot, stitch together the three images and project it onto the x-axis.  Save the final lineout
    #     all_projections = []
    #     for shot_num in self.auxiliary_data['Shotnumber'].values:
    #         stitched_projection = None
    #         for device in self.device_list:
    #             # For more complex analysis, the actual `image analysis` code within this `for` block could be moved
    #             #  to a dedicated ImageAnalysis class, and one could multithread the analysis so that each thread gets
    #             #  a single image to analyze.  But for this simple example I am opting to keep it all within this file
    #             image_file = ScanData.get_device_shot_path(tag=self.tag, device_name=device,
    #                                                        shot_number=int(float(shot_num)))
    #             image = read_imaq_png_image(image_file)*1.0
    #             # image -= self.backgrounds[device]
    #             image[np.where(image < 0)] = 0
    #             image = image[1:, :]
    #             # image = median_filter(image, size=3)
    #             # image = threshold_reduction(image=image, threshold=2)

    #             projection = np.sum(image, axis=0)
    #             if stitched_projection is None:
    #                 stitched_projection = projection
    #             else:
    #                 stitched_projection = np.concatenate((stitched_projection, projection))

    #             if device == 'CAM-CR-Beampointing1':  # Add on some zeros
    #                 stitched_projection = np.concatenate((stitched_projection, np.zeros(200)))

    #             #plt.imshow(image, aspect='auto', vmin = 0, vmax = 15)
    #             #plt.plot(projection)
    #             #plt.show()

    #         all_projections.append(stitched_projection)
    #         # Calculate the mean sum counts for the current shot
    #         mean_counts = np.mean(stitched_projection)
    #         max_counts = np.max(stitched_projection)
    #         sum_counts = np.sum(stitched_projection)

    #         #plt.plot(stitched_projection)
    #         #plt.show()

    #     all_projections = np.vstack(all_projections)

    #     plt.figure(figsize=(10, 6))
    #     plt.imshow(all_projections, aspect='auto')
    #     plt.xlabel('Stitched Image Horizontal Pixel')
    #     plt.ylabel('Shotnumber')
    #     plt.title(f"{self.tag.experiment}: {self.tag.month:02d}/{self.tag.day:02d}/{self.tag.year} Scan{self.tag.number:03d} Raw Magspec Waterfall")

    #     save_path = self.save_path / 'RawMagspecWaterfall'
    #     save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    #     plt.savefig(save_path, bbox_inches='tight')

    #     self.close_or_show_plot()

    #     # Could from here do additional analysis and append scalars to the sfile using:
    #     # self.append_to_sfile(self, dict_to_append={'key': list_of_data}):

    #     self.display_contents.append(str(save_path))
    #     return self.display_contents

    def run_analysis(self, config_options: Optional[str] = None):
        """ Main function to run the analysis and generate plots. """
        # For each shot, read, process, and display the image
        for shot_num in self.auxiliary_data['Shotnumber'].values:
            for device in self.device_list:
                image_file = ScanData.get_device_shot_path(tag=self.tag, device_name=device,
                                                        shot_number=int(float(shot_num)))
                image = read_imaq_png_image(image_file)*1.0
                # image -= self.backgrounds[device]
                image[np.where(image < 0)] = 0
                image = image[1:, :]
                image = median_filter(image, size=3)
                # image = threshold_reduction(image=image, threshold=2)
                max = np.max(image)
                mean = np.mean(image)
                sum = np.sum(image)
                projection = np.sum(image, axis=0)
                
                print(shot_num, device, "Max mean sum projection =", max, mean, sum, projection)
                
                # Display the image and its projection
                plt.figure(figsize=(10, 6))
                plt.subplot(2, 1, 1)
                plt.imshow(image, aspect='auto', vmin=0, vmax=15)
                plt.title(f"{device} Image for Shot {shot_num}")
                plt.subplot(2, 1, 2)
                plt.plot(projection)
                plt.xlabel('Horizontal Pixel')
                plt.ylabel('Intensity')
                plt.title(f"{device} Projection for Shot {shot_num}")
                plt.tight_layout()
                plt.show()

                # # Save the plot
                # save_path = self.save_path / f"{device}_Shot{shot_num}_ImageAndProjection.png"
                # save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
                # plt.savefig(save_path, bbox_inches='tight')

                # self.display_contents.append(str(save_path))
        return self.display_contents

if __name__ == "__main__":
    from geecs_python_api.analysis.scans.scan_data import ScanData
    tag = ScanData.get_scan_tag(year=2025, month=3, day=4, number=5, experiment='ControlRoom')
    analyzer = PWTestAnalysis(scan_tag=tag, skip_plt_show=False)
    analyzer.run_analysis()
