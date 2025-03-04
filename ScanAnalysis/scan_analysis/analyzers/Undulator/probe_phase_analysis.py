"""
Visa E Beam Image Analysis

Visa YAG screen image analyzer.
Child to CameraImageAnalysis (./scan_analysis/analyzers/Undulator/CameraImageAnalysis.py)
"""
# %% imports
from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag
import numpy as np
import cv2
from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalysis
from geecs_python_api.analysis.scans.scan_data import ScanData

from image_analysis.utils import read_imaq_image


# %% classes
class ProbePhaseAnalysis(CameraImageAnalysis):

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

        # enact parent init
        super().__init__(scan_tag, self.device_name, skip_plt_show=skip_plt_show,
                         flag_logging=flag_logging, flag_save_images=flag_save_images)

        # redefine save path for this specific analysis
        # Organize paths: one for image data and one for analysis output.
        if self.scaling_type == 'phase':
            save_dir = self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / device_name / 'PhaseAnalysis'
        elif self.scaling_type == 'intensity':
            save_dir = self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / device_name / 'IntensityAnalysis'

        self.path_dict = {
            'data_img': self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / device_name / 'HasoAnalysis',
            'save': save_dir
        }

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
        logging.info(f'shots in bin:{shots_in_bin}')

        # expected_image_size = None

        for shot_num in shots_in_bin:

            if self.scaling_type == 'phase':
                image_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}_postprocessed.tsv'), None)
            elif self.scaling_type == 'intensity':
                image_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}_intensity.tsv'), None)
            logging.info(f'image file: {image_file}')

            if image_file:
                image = read_imaq_image(image_file)
                images.append(image)
                # if expected_image_size is None:
                # expected_image_size = image.shape  # Determine image size from the first valid image
            else:
                if self.flag_logging:
                    logging.warning(f"Missing data for shot {shot_num}, adding zero array.")
                # kj comment: not sure the below makes sense.
                # Will keep code for now but comment out.
                # if expected_image_size:
                #     images.append(np.zeros(expected_image_size, dtype=np.uint16))

        return images

    def average_images(self, images: list[np.ndarray]) -> Optional[np.ndarray]:
        """
        Average a list of images.

        Args:
            images (list of np.ndarray): List of images to average.

        Returns:
            np.ndarray: The averaged image.
        """
        if len(images) == 0:
            return None
        avg_image = np.mean(images, axis=0)

        avg_image = np.abs(avg_image - np.nanmin(avg_image))
        if self.scaling_type == 'phase':
            avg_image = avg_image*2**16
        elif self.scaling_type == 'intensity':
            avg_image = avg_image

        return avg_image.astype(np.uint16)  # Keep 16-bit format for the averaged image

if __name__ == "__main__":
    tag = ScanData.get_scan_tag(year=2025, month=2, day=26, number=17, experiment_name='Undulator')
    analyzer = ProbePhaseAnalysis(scan_tag=tag, device_name='U_HasoLift', skip_plt_show=True, scaling_type='intensity')
    analyzer.run_analysis()
