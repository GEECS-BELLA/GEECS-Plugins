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
from image_analysis.analyzers.density_from_phase_analysis import PhaseDownrampProcessor, PhaseAnalysisConfig

from geecs_python_api.analysis.scans.scan_data import ScanData, ScanTag
from image_analysis.utils import read_imaq_image

# %% classes
class DownrampPhaseAnalysis(CameraImageAnalysis):

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

        bkg_st = ScanTag(2025, 2, 19, 2, experiment='Undulator')
        bkg_s_data = ScanData(tag=bkg_st)
        path_to_bkg = Path(bkg_s_data.get_analysis_folder() / "U_HasoLift" / "HasoAnalysis" / 'average_phase.tsv')

        config: PhaseAnalysisConfig = PhaseAnalysisConfig(
            pixel_scale=10.1,  # um per pixel (vertical)
            wavelength_nm=800,  # Probe laser wavelength in nm
            threshold_fraction=0.2,  # Threshold fraction for pre-processing
            roi=(10, -10, 10, -100),  # Example ROI: (x_min, x_max, y_min, y_max)
            background=path_to_bkg  # Background is now a Path
        )

        self.processor = PhaseDownrampProcessor(config)

        # enact parent init
        super().__init__(scan_tag, self.device_name, skip_plt_show=skip_plt_show,
                         flag_logging=flag_logging, flag_save_images=flag_save_images)

        # redefine save path for this specific analysis
        # Organize paths: one for image data and one for analysis output.
        if self.scaling_type == 'phase':
            save_dir = self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / device_name / 'PlasmaPhaseAnalysis'

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

        for shot_num in shots_in_bin:

            if self.scaling_type == 'phase':
                image_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}_postprocessed.tsv'), None)

            # if image_file:
            #     try:
            #         image = self.processor.analyze_image(image_file)
            #         images.append(image)
            #     except:
            #         logging.warning('image skipped')
            if image_file:
                try:
                    # Get the analysis dictionary, e.g., {"var1": value, "var2": value}
                    results_dict = self.processor.analyze_image(image_file)
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
            avg = avg_image - np.min(avg_image)
            avg_image = avg_image*2**16
        elif self.scaling_type == 'intensity':
            avg_image = avg_image

        return avg_image.astype(np.uint16)  # Keep 16-bit format for the averaged image

if __name__ == "__main__":
    tag = ScanData.get_scan_tag(year=2025, month=2, day=19, number=3, experiment_name='Undulator')
    analyzer = DownrampPhaseAnalysis(scan_tag=tag, device_name='U_HasoLift', skip_plt_show=True, scaling_type='phase')
    analyzer.run_analysis()
