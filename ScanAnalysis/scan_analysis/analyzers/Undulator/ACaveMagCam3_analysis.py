"""
Visa E Beam Image Analysis

Visa YAG screen image analyzer.
Child to CameraImageAnalysis (./scan_analysis/analyzers/Undulator/CameraImageAnalysis.py)
"""
# %% imports
from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag
import numpy as np
import cv2
from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalysis
from image_analysis.analyzers.ACaveMagCam3 import ACaveMagCam3ImageAnalyzer

from geecs_python_api.analysis.scans.scan_data import ScanData, ScanTag
from image_analysis.utils import read_imaq_image

# %% classes
class ACaveMagCam3ScanAnalysis(CameraImageAnalysis):

    def __init__(self, scan_tag: ScanTag,
                 device_name: Optional[str] = None, skip_plt_show: bool = True,
                 flag_logging: bool = True, flag_save_images: bool = True,
                 scaling_type: str = 'phase') -> None:
        """
        Initialize the VisaEBeamAnalysis class.

        Args:
            scan_tag (ScanTag): Path to the scan directory containing data.
            device_name (str): Name of the Visa camera.  If not given, automatically detects which one
            skip_plt_show (bool): Flag that sets if matplotlib is tried to use for plotting
            flag_logging (bool): Flag that sets if error and warning messages are displayed
            flag_save_images (bool): Flag that sets if images are saved to disk
        """
        self.device_name = device_name
        self.scaling_type = scaling_type

        self.processor = ACaveMagCam3ImageAnalyzer()

        # enact parent init
        super().__init__(scan_tag, self.device_name, skip_plt_show=skip_plt_show,
                         flag_logging=flag_logging, flag_save_images=flag_save_images)

    def load_images_for_bin(self, bin_number: int) -> list[np.ndarray]:
        """
        Load all images corresponding to a specific bin number by matching the shot number.

        Args:
            bin_number (int): The bin number for which to load images.

        Returns:
            list of np.ndarray: List of images in the bin.
        """
        images = []
        shots_in_bin = self.auxiliary_data[self.auxiliary_data['Bin #'] == bin_number]['Shotnumber'].values
        # expected_image_size = None

        for shot_num in shots_in_bin:
            image_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}.png'), None)
            if image_file:
                try:
                    # Get the analysis dictionary, e.g., {"var1": value, "var2": value}
                    results_dict = self.processor.analyze_image(file_path = image_file)
                    analysis_dict = results_dict['analysis_results']
                    image = results_dict['processed_image']
                    images.append(image)

                    logging.info(f"Analysis for shot {shot_num}: {analysis_dict}")

                    # Update the row(s) in auxiliary_data that match the current shot number.
                    for key, value in analysis_dict.items():
                        # This creates a new column if key does not exist.
                        self.auxiliary_data.loc[self.auxiliary_data['Shotnumber'] == shot_num, key] = value
                except Exception as e:
                    logging.warning(f"Image for shot {shot_num} skipped due to error: {e}")
            else:
                if self.flag_logging:
                    logging.warning(f"Missing data for shot {shot_num}, adding zero array.")

        return images

    def run_noscan_analysis(self):
        """
        Image analysis in the case of a no scan.

        """
        # load images
        data = {'shot_num': [], 'images': []}
        for shot_num in self.auxiliary_data['Shotnumber'].values:
            image_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}.png'), None)
            logging.info(f'image file is {image_file}')
            if image_file:
                data['shot_num'].append(shot_num)
                results_dict = self.processor.analyze_image(file_path = image_file)
                analysis_dict = results_dict['analysis_results']
                image = results_dict['processed_image'].astype(np.uint16)
                data['images'].append(image)

                # Update the row(s) in auxiliary_data that match the current shot number.
                for key, value in analysis_dict.items():
                    # This creates a new column if key does not exist.
                    self.auxiliary_data.loc[self.auxiliary_data['Shotnumber'] == shot_num, key] = value
            else:
                if self.flag_logging:
                    logging.warning(f"Missing data for shot {shot_num}.")


        # get average image
        avg_image = self.average_images(data['images'])

        if self.flag_save_images:
            self.save_geecs_scaled_image(avg_image, save_dir=self.path_dict['save'],
                                         save_name=f'{self.device_name}_average_processed.png')

            save_name = f'{self.device_name}_average_processed_visual.png'
            self.save_normalized_image(avg_image, save_dir=self.path_dict['save'],
                                       save_name=save_name, label=save_name)
            display_content_path = Path(self.path_dict['save']) / save_name

            self.display_contents.append(str(display_content_path))

        # make gif
        if self.flag_save_images:
            filepath = self.path_dict['save'] / 'noscan.gif'
            self.create_gif(data['images'], filepath,
                            titles=[f"Shot {num}" for num in data['shot_num']])

            self.display_contents.append(str(filepath))

if __name__ == "__main__":
    tag = ScanData.get_scan_tag(year=2025, month=3, day=6, number=39, experiment_name='Undulator')
    analyzer = ACaveMagCam3ScanAnalysis(scan_tag=tag, device_name='UC_ACaveMagCam3', skip_plt_show=True)
    analyzer.run_analysis()
