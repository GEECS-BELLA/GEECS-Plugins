"""
Classes containing common functionality and requirements available for all scan analyzers.  ScanAnalysis is the parent
class for all implementing analyzers and the main requirements are to satisfy the initialization and `run_analysis()`.

------------------------------------------------------------------------------------------------------------------------

For the "requirements" block of AnalyzerInfo to be compatible with `scan_evaluator.py`, follow these guidelines and see
`map_Undulator` as an example.

# Either a dictionary element of 'AND' or 'OR' followed by a list of devices.
# Alternatively, just a set/list of devices will suffice.

# AND blocks are evaluated true if all the devices exist in a given scan folder
# OR blocks are evaluated true if at least one of the devices exist
# AND/OR dict blocks can be written as recursive elements.
"""
# %% imports
from __future__ import annotations
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Optional, Union, Type, NamedTuple, Dict, List
if TYPE_CHECKING:
    from geecs_data_utils import ScanTag
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geecs_data_utils import ScanData

from image_analysis.base import ImageAnalyzer  # explicit import for the base image analyzer

# %% classes
class AnalyzerInfo(NamedTuple):
    analyzer_class: Type[ScanAnalysis]
    requirements: Union[dict[str, list], set, str]
    device_name: Optional[str] = None
    config_file: Optional[str] = None
    is_active: bool = True
    # New optional field for explicit image analyzer import
    image_analyzer_class: Optional[Type[ImageAnalyzer]] = None
    file_tail: Optional[str] = ".png"  # required file extension
    image_analysis_config: Optional[any] = None

# error classes
class DataLengthError(ValueError):
    """Raised when data arrays have inconsistent lengths."""
    pass

class ScanAnalysis:
    """
    Base class for performing analysis on scan data. Handles loading auxiliary data and extracting
    scan parameters from .ini files.

    Attributes:
        scan_directory (Path): Path to the scan directory containing data.
        auxiliary_file_path (Path): Path to the auxiliary ScanData file.
        ini_file_path (Path): Path to the .ini file containing scan parameters.
        scan_parameter (str): The scan parameter extracted from the .ini file.
        bins (np.ndarray): Bin numbers for the data, extracted from the auxiliary file.
        auxiliary_data (pd.DataFrame): DataFrame containing auxiliary scan data.
    """

    def __init__(self, scan_tag: ScanTag, device_name: Optional[str] = None, skip_plt_show: bool = True, image_analyzer=None):
        """
        Initialize the ScanAnalysis class.

        Args:
            scan_tag (ScanTag): NamedTuple containing the scan's experiment, date, and scan number
            device_name (Optional[str]): An optional string to specify which device the analyzer should analyze
            skip_plt_show (bool): Flag to ultimately try plt.show() or not.
        """
        self.tag = scan_tag
        self.scan_data = ScanData(tag=self.tag, load_scalars=True, read_mode=True)
        self.scan_directory = self.scan_data.get_folder()  # ScanData.get_scan_folder_path(tag=scan_tag)
        self.experiment_dir = scan_tag.experiment
        self.auxiliary_file_path = self.scan_data.get_analysis_folder().parent / f"s{self.tag.number}.txt"
        self.ini_file_path = self.scan_directory / f"ScanInfo{self.scan_directory.name}.ini"
        self.noscan = False

        self.device_name = device_name

        self.skip_plt_show = skip_plt_show

        self.bins = None
        self.auxiliary_data: Optional[pd.DataFrame] = None
        self.binned_param_values = None
        # scan_path = self.scan_directory.parent.parent / 'analysis' / self.scan_directory.name
        #
        # self.scan_path: Path = Path(scan_path)
        self.scan_path: Path = self.scan_data.get_analysis_folder()
        logging.info(f'analysis path is : {self.scan_path}')
        self.display_contents = []

        try:
            # Extract the scan parameter
            self.scan_parameter = self.extract_scan_parameter_from_ini()

            logging.info(f"Scan parameter is: {self.scan_parameter}.")
            s_param = self.scan_parameter.lower()

            if s_param == 'noscan' or s_param == 'shotnumber':
                logging.warning("No parameter varied during the scan, setting noscan flag.")
                self.noscan = True

            self.load_auxiliary_data()

            if self.auxiliary_data is None:
                logging.warning("Scan parameter not found in auxiliary data. Possible aborted scan. Skipping analysis.")
                return  # Stop further execution cleanly

            self.total_shots = len(self.auxiliary_data)

        except FileNotFoundError as e:
            logging.warning(f"{e}. Could not find auxiliary or .ini file in {self.scan_directory}. Skipping analysis.")
            return


    def run_analysis(self, config_options: Optional[Union[Path, str]] = None) -> Optional[list[Union[Path, str]]]:
        """
        Analysis routine called by execute_scan_analysis for a given analyzer.  Needs to be implemented for each class

        :param config_options: Optional input to specify a filepath for configuration settings
        :return: Optional return for a list of key images/files generated by the analysis for use with experiment log
        """
        raise NotImplementedError

    def extract_scan_parameter_from_ini(self) -> str:
        """
        Extract the scan parameter from the .ini file.

        Returns:
            str: The scan parameter with colons replaced by spaces.
        """

        ini_contents = self.scan_data.load_scan_info()
        cleaned_scan_parameter = ini_contents['Scan Parameter'].strip().replace(':', ' ').replace('"', '')
        return cleaned_scan_parameter

    def load_auxiliary_data(self):
        """ Uses the data frame in the ScanData instance to find the bins and the binned parameter values 
            Note: Auxiliary data loaded from tdms file, not scan file or sfile.
        """
        try:
            self.auxiliary_data = pd.read_csv(self.auxiliary_file_path, delimiter='\t')
            self.bins = self.auxiliary_data['Bin #'].values

            if not self.noscan:
                # Find the scan parameter column and calculate the binned values
                scan_param_column = self.find_scan_param_column()[0]
                self.binned_param_values = self.auxiliary_data.groupby('Bin #')[scan_param_column].mean().values

        except (KeyError, FileNotFoundError) as e:
            logging.warning(f"{e}. Scan parameter not found in auxiliary data. Possible aborted scan. Skipping")

    def close_or_show_plot(self):
        """Decide whether to display or close plots based on the skip_plt_show setting."""
        if not self.skip_plt_show:
            plt.show()  # Display for interactive use
        else:
            plt.close('all')  # Ensure plots close when not using the GUI

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
                raise DataLengthError()

            # check if columns exist within dataframe
            existing_cols = set(df_copy) & set(dict_to_append.keys())
            if existing_cols:
                # if self.flag['logging']:
                logging.warning(f"Warning: Columns already exist in sfile: "
                                f"{existing_cols}. Overwriting existing columns.")

            # append new fields to df_copy
            df_new = df_copy.assign(**dict_to_append)

            # save updated dataframe to sfile
            df_new.to_csv(self.auxiliary_file_path,
                          index=False, sep='\t', header=True)

            # copy updated dataframe to class attribute
            self.auxiliary_data = df_new.copy()

        except DataLengthError:
            # if self.flag['logging']:
            logging.error(f"Error: Error appending {self.device_name} field to sfile due to "
                          f"inconsistent array lengths. Scan file not updated.")

        except Exception as e:
            logging.error(f"Error: Unexpected error in {self.append_to_sfile.__name__}: {e}")

    def generate_limited_shotnumber_labels(self, max_labels: int = 20) -> np.ndarray:
        """
        Generate a list of shot number labels with a maximum of `max_labels`.

        Args:
            max_labels (int): Maximum number of labels to display.

        Returns:
            np.ndarray: Array of shot numbers, spaced out if necessary.
        """
        if self.total_shots <= max_labels:
            # If the number of shots is less than or equal to max_labels, return the full range
            return np.arange(1, self.total_shots + 1)
        else:
            # Otherwise, return a spaced-out array with at most max_labels
            step = self.total_shots // max_labels
            return np.arange(1, self.total_shots + 1, step)

    def find_scan_param_column(self) -> tuple[Optional[str], Optional[str]]:
        """
        Find the column in the auxiliary data corresponding to the scan parameter.
        The method strips unnecessary characters (e.g., quotes) and handles cases where an alias is present.

        Returns:
            tuple: A tuple containing the column name and alias (if any) for the scan parameter.
        """
        # Clean the scan parameter by stripping any quotes or extra spaces
        # cleaned_scan_parameter = self.scan_parameter

        if not self.noscan:
            # Search for the first column that contains the cleaned scan parameter string
            for column in self.auxiliary_data.columns:
                # Match the part of the column before 'Alias:'
                if self.scan_parameter in column.split(' Alias:')[0]:
                    # Return the column and the alias if present
                    return column, column.split('Alias:')[-1].strip() if 'Alias:' in column else column

            logging.warning(f"Warning: Could not find column containing scan parameter: {self.scan_parameter}")
            return None, None
        else:
            return None, None


# %% executable
def testing_routine():
    print(ScanData)
    pass


if __name__ == "__main__":
    testing_routine()
