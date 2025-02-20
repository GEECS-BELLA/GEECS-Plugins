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
                 flag_save_images: bool = True) -> None:
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

    def run_analysis(self):
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
            # integrate wrt y-axis (horizontal/temporal lineouts)
            lineout = img.sum(axis=1)
    
            # calculate second moment of temporal lineouts
            second_moment = self.calculate_second_moment(lineout)
    
            # get peak value of lineout
            peak_value = lineout.max()
        
        except Exception as e:
            second_moment, peak_value = None, None

        # organize outputs
        outputs = {'second_moment': second_moment,
                   'peak_value': peak_value}

        return outputs

    def append_to_sfile(self,
                        dict_to_append: Dict[str, Union[List, NDArray[np.float64]]]) -> None:
        """
        Append new data to the auxiliary file.
        
        Args:
            dict_to_append: Dictionary containing column names and their values to append
            
        Raises:
            DataLengthError: If the length of array values doesn't match existing data
        """
        try:
            # copy auxiliary dataframe
            df_copy = self.auxiliary_data.copy()
    
            # check column lengths match existing dataframe
            lengths = {len(vals) for vals in dict_to_append.values() if isinstance(vals, (list, np.ndarray))}
            if lengths and lengths.pop() != len(df_copy):
                if self.flag['logging']:
                    raise DataLengthError()

            # check if columns exist within dataframe
            existing_cols = set(df_copy) & set(dict_to_append.keys())
            if existing_cols:
                if self.flag['logging']:
                    logging.warning(f"Warning: Columns already exist in sfile: {existing_cols}. Overwriting existing columns.")
    
            # append new fields to df_copy
            df_new = df_copy.assign(**dict_to_append)
    
            # save updated dataframe to sfile
            df_new.to_csv(self.auxiliary_file_path,
                          index=False, sep='\t', header=True)
    
            # copy updated dataframe to class attribute
            self.auxiliary_data = df_new.copy()

        except DataLengthError:
            logging.error(f"Error: Error appending {self.device_name} field to sfile due to inconsistent array lengths. Scan file not updated.")

        except Exception as e:
            logging.error(f"Error: Unexpected error in {self.append_to_sfile.__name__}: {e}")


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

# error classes
class DataLengthError(ValueError):
    """Raised when data arrays have inconsistent lengths."""
    pass


# %% routine

def testing():

    from geecs_python_api.analysis.scans.scan_data import ScanData

    kwargs = {'year': 2024, 'month': 12, 'day': 10, 'number': 9, 'experiment': 'Undulator'}
    tag = ScanData.get_scan_tag(**kwargs)

    analyzer = FrogAnalysis(scan_tag=tag, device_name="U_FROG_Grenouille-Temporal")

    analyzer.run_analysis()

    pass

# %% execute
if __name__ == "__main__":
    testing()


# Finns analysis

# import os
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt


# # folderpath = "12-09-2024 (low power)"
# # subfolder = "U_FROG_Grenouille-Temporal"
# # scan_number = "Scan008"

# folderpath = "12-10-2024 (High Power)"
# subfolder = "U_FROG_Grenouille-Temporal"
# scan_number = "Scan009"


# def load_images(scan_number, base_folder=folderpath):
#     """
#     Load images from the specified scan folder and map them to shot numbers.

#     Parameters:
#         scan_number (str): The scan number to look for.
#         base_folder (str): The base directory containing all scans.

#     Returns:
#         dict: A dictionary mapping shot numbers to (filename, PIL.Image) tuples.
#     """
#     scan_folder = os.path.join(base_folder, scan_number, subfolder)
#     if not os.path.exists(scan_folder):
#         print(f"Error: The folder '{scan_folder}' does not exist.")
#         return {}

#     images = {}
#     for filename in sorted(os.listdir(scan_folder)):
#         if filename.endswith(".png") and scan_number in filename:
#             try:
#                 # Extract shot number from the filename
#                 shot_number = int(filename.split('_')[-1].split('.')[0])
#                 image_path = os.path.join(scan_folder, filename)
#                 img = Image.open(image_path)
#                 images[shot_number] = (filename, img)
#             except Exception as e:
#                 print(f"Error processing file {filename}: {e}")

#     if not images:
#         print(f"No images found in '{scan_folder}' for scan '{scan_number}'.")
#     else:
#         print(f"Loaded {len(images)} images from '{scan_folder}'.")
    
#     return images

# def integrate_images(images):
#     """
#     Integrate each image over the y-axis.

#     Parameters:
#         images (dict): A dictionary of images mapped to shot numbers.

#     Returns:
#         dict: A dictionary of integrated y-axis arrays mapped to shot numbers.
#     """
#     integrated_data = {}
#     for shot_number, (filename, img) in images.items():
#         img_array = np.array(img)
#         y_integrated = np.sum(img_array, axis=1)
#         integrated_data[shot_number] = y_integrated
#     return integrated_data

# def calculate_second_moments(integrated_data):
#     """
#     Calculate the second moment and peak value of each integrated trace.

#     Parameters:
#         integrated_data (dict): A dictionary of integrated y-axis arrays.

#     Returns:
#         dict: A dictionary of second moments and peak values mapped to shot numbers.
#     """
#     second_moments = {}
#     for shot_number, y_integrated in integrated_data.items():
#         y_indices = np.arange(len(y_integrated))
        
#         # Calculate the second moment
#         mean_y = np.sum(y_indices * y_integrated) / np.sum(y_integrated)
#         second_moment = np.sqrt(np.sum(((y_indices - mean_y) ** 2) * y_integrated) / np.sum(y_integrated))
        
#         # Calculate the peak value
#         peak_value = np.max(y_integrated)
        
#         # Store both second moment and peak value
#         second_moments[shot_number] = (second_moment, peak_value)
    
#     return second_moments


# def save_second_moments(second_moments, output_file):
#     """
#     Save the second moments and peak values to a text file.

#     Parameters:
#         second_moments (dict): A dictionary of second moments and peak values mapped to shot numbers.
#         output_file (str): The output file name.
#     """
#     with open(output_file, "w") as f:
#         f.write("Shot_Number\tSecond_Moment\tPeak_Value\n")
#         for shot_number, (second_moment, peak_value) in sorted(second_moments.items()):
#             f.write(f"{shot_number}\t{second_moment}\t{peak_value}\n")
#     print(f"Second moments and peak values saved to '{output_file}'.")

# # Example usage
# images = load_images(scan_number)
# if images:
#     integrated_data = integrate_images(images)
#     second_moments = calculate_second_moments(integrated_data)
#     #save_second_moments(second_moments, output_file=f"second_moments-{folderpath}-{scan_number}.txt")

# def plot_image_with_integration(images, integrated_data, N):
#     if N not in images:
#         print(f"Shot number {N} not found.")
#         return

#     filename, img = images[N]
#     y_integrated = integrated_data[N]
#     img_array = np.array(img)

#     fig, ax1 = plt.subplots(figsize=(5, 4),dpi=300)

#     # Plot the image
#     ax1.imshow(img_array, cmap='gray', aspect='auto')
#     ax1.set_xlabel("X-axis (Pixel)")
#     ax1.set_ylabel("Y-axis (Pixel)")
#     #ax1.set_title(f"Image: {filename}")

#     # Create a twin x-axis for the integrated trace
#     ax2 = ax1.twiny()
#     ax2.plot(y_integrated, range(len(y_integrated)), color='red', label=f"{filename}")
#     ax2.invert_yaxis()  # Match the orientation of the image
#     ax2.set_xlabel("Integrated Value")
#     ax2.legend(loc='upper right')

#     plt.tight_layout()
#     plt.show()


# images = load_images(scan_number)
# if images:
#     integrated_data = integrate_images(images)
#     N = 1  # Set N to the desired shot number
#     plot_image_with_integration(images, integrated_data, N)
