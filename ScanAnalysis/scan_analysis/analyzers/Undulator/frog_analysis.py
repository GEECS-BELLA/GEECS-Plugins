"""
Frog Analysis

Quick FROG analysis.

authors:
Kyle Jensen, kjensen@lbl.gov
Finn Kohrell, 
"""
# %% imports
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional, Dict, List
from numpy.typing import NDArray
if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag

from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

from scan_analysis.base import ScanAnalysis
from image_analysis.utils import read_imaq_image

# %% classes

class FrogAnalysis(ScanAnalysis):

    def __init__(self,
                 scan_tag: ScanTag,
                 device_name: str,
                 skip_plt_show: bool = True,
                 flag_logging: bool = True,
                 flag_save_images: bool = True,
                 image_analyzer=None) -> None:
        """
        Initialize FrogAnalysis class for analyzing FROG images.

        Parameters
        ----------
        scan_tag : ScanTag
            Path to scan directory containing data
        device_name : str
            Name of device to construct the subdirectory path
        skip_plt_show : bool, optional
            Whether to skip matplotlib plotting, by default True
        flag_logging : bool, optional
            Whether to enable error and warning messages, by default True
        flag_save_images : bool, optional
            Whether to save images to disk, by default True

        Raises
        ------
        ValueError
            If device_name is empty
        """
        if not device_name:
            raise ValueError("FrogAnalysis requires a device name.")

        super().__init__(scan_tag=scan_tag, device_name=device_name,
                         skip_plt_show=skip_plt_show)

        # store flags
        self.flag = {'logging': flag_logging,
                     'save_images': flag_save_images}

        # organize various paths
        self.path_dict = {'data_img': Path(self.scan_directory) / f"{device_name}",
                          'save': (self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name
                                   / f"{device_name}" / f"{self.__class__.__name__}")}

        # check if data directory exists and is not empty
        if not self.path_dict['data_img'].exists() or not any(self.path_dict['data_img'].iterdir()):
            if self.flag['logging']:
                logging.warning(f"Data directory '{self.path_dict['data_img']}' does not exist or is empty.")

    def run_analysis(self, config_options: Optional[Union[Path, str]] = None) -> Optional[list[Union[Path, str]]]:
        """
        Run the appropriate analysis based on scan type.

        Returns
        -------
        List or None
            List of display contents if successful, None otherwise
        """
        # run initial checks
        if self.path_dict['data_img'] is None or self.auxiliary_data is None:
            if self.flag['logging']:
                logging.info(f"Warning: Skipping {self.__class__.__name__} for {self.device_name} due to missing data or auxiliary file.")
            return

        # if saving, make sure save location exists
        if self.flag['save_images'] and not self.path_dict['save'].exists():
            self.path_dict['save'].mkdir(parents=True)

        # delegate analysis type
        try:
            if self.noscan:
                self.run_noscan_analysis()
            else:
                self.run_scan_analysis()

            return self.display_contents

        except Exception as e:
            if self.flag['logging']:
                logging.warning(f"Warning: {self.__class__.__name__} for {self.device_name} failed due to: {e}")

    def run_noscan_analysis(self):

        # run base analysis
        shot_dict = self.base_analysis()


            # append scalars to sfile

        # plot second moment vs shotnumber
            # save, append to display contents list

        pass

    def run_scan_analysis(self):

        # run base analysis
        shot_dict = self.base_analysis()

        # bin scan parameters

        # plot second moment vs scan parameter
            # save, append to display contents

        pass

    def base_analysis(self) -> Dict[str, Union[int, NDArray, float]]:
        """
        Perform base analysis common to both scan types.

        Returns
        -------
        dict
            Dictionary containing analysis results including images and measurements
        """
        # load images
        img_dict = self.load_images()
        num_shots = len(img_dict['images'])

        # initialize containters upfront, analyze first shot
        result_dict = {}
        first_img = self.single_shot_analysis(img_dict['images'][0])
        for key, val in first_img.items():
            if key not in result_dict:
                result_dict[key] = [[] for _ in range(num_shots)]
                result_dict[key][0] = val

        # iterate remaining shots
        for ind in range(1, num_shots):
            result = self.single_shot_analysis(img_dict['images'][ind])
            for key, val in result.items():
                result_dict[key][ind] = val

        # append to sfile
        append_dict = {f"{self.device_name}: {key}": val for key, val in result_dict.items()}
        self.append_to_sfile(append_dict)

        # organize return dict
        return_dict = img_dict | result_dict

        return return_dict

    def load_images(self) -> Dict[str: List[Union[int, NDArray[np.float64]]]]:
        """
        Load images for all shots in the scan.

        Returns
        -------
        dict
            Dictionary containing:
                'shot_number' : list of int
                    Shot numbers for each image
                'images' : list of ndarray
                    Corresponding image data
        """
        # initialize storage
        shot_numbers = self.auxiliary_data['Shotnumber'].values
        images = [[] for _ in shot_numbers]

        # iterate shot numbers, load and store image
        for ind, shot_num in enumerate(shot_numbers):
            try:
                file = self.scan_data.get_device_shot_path(self.tag, self.device_name,
                                                           shot_num, file_extension='png')

                images[ind] = read_imaq_image(file)

            except Exception as e:
                if self.flag['logging']:
                    logging.error(f"Warning: Error reading data for {self.device_name}, shot {shot_num}: {e}")
                images[ind] = None

        # return as dict
        img_dict = {'shot_number': shot_numbers, 'images': images}

        return img_dict

    def single_shot_analysis(self, img: NDArray[np.float64]) -> Dict[str, float]:
        """
        Analyze a single FROG image.

        Parameters
        ----------
        img : ndarray
            2D numpy array containing the image data

        Returns
        -------
        dict
            Dictionary containing:
                'second_moment' : float
                    Second moment of the temporal lineout
                'peak_value' : float
                    Peak value of the temporal lineout
        """
        try:
            # integrate wrt to each axis (horizontal = temporal, vertical = spectral)
            spectral = img.sum(axis=0)
            temporal = img.sum(axis=1)

            # calculate second moment of temporal lineouts
            spectral_second_moment = self.calculate_second_moment(spectral)
            temporal_second_moment = self.calculate_second_moment(temporal)
    
            # get peak value of lineout
            spectral_peak = spectral.max()
            temporal_peak = temporal.max()
        
        except Exception:
            spectral_second_moment, spectral_peak = None, None
            temporal_second_moment, temporal_peak = None, None

        # organize outputs
        outputs = {'temporal_second_moment': temporal_second_moment,
                   'temporal_peak': temporal_peak,
                   'spectral_second_moment': spectral_second_moment,
                   'spectral_peak': spectral_peak}

        return outputs

    @staticmethod
    def calculate_second_moment(data: NDArray[np.float64]) -> float:
        """
        Calculate the second moment of the input data.

        Parameters
        ----------
        data : ndarray
            1D numpy array of intensity values

        Returns
        -------
        float
            Second moment of the distribution
        """
        indices = np.arange(len(data))
        mean = np.sum(indices * data) / np.sum(data)
        second_moment = np.sqrt(((indices - mean)**2 * data).sum() / data.sum())
        return second_moment
# %% routine


def testing():

    from geecs_python_api.analysis.scans.scan_data import ScanData

    kwargs = {'year': 2025, 'month': 3, 'day': 6, 'number': 15, 'experiment': 'Undulator'}
    tag = ScanData.get_scan_tag(**kwargs)

    analyzer = FrogAnalysis(scan_tag=tag, device_name="U_FROG_Grenouille-Temporal")

    analyzer.run_analysis()

    pass

# %% execute
if __name__ == "__main__":
    testing()
