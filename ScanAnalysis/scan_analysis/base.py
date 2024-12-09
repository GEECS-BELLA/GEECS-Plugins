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
from typing import TYPE_CHECKING, List, Dict, Optional, Union, Type, NamedTuple
if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag
from pathlib import Path
import logging
import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geecs_python_api.analysis.scans.scan_data import ScanData
import yaml


# %% classes
class AnalyzerInfo(NamedTuple):
    analyzer_class: Type['ScanAnalysis']
    requirements: Union[dict[str, list], set, str]
    device_name: Optional[str] = None
    config_file: Optional[str] = None


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
    def __init__(self, scan_tag: 'ScanTag', device_name: Optional[str] = None, skip_plt_show: bool = True, clear_display_contents_file: bool = True):
        """
        Initialize the ScanAnalysis class.

        Args:
            scan_tag (ScanTag): NamedTuple containing the scan's experiment, date, and scan number
            device_name (Optional[str]): An optional string to specify which device the analyzer should analyze
            skip_plt_show (bool): Flag to ultimately try plt.show() or not.
            clear_display_contents_file (bool): Flag to clear all previous results in the analysis/ScanXXX folder
        """
        self.tag = scan_tag
        self.scan_directory = ScanData.build_scan_folder_path(tag=scan_tag)
        self.experiment_dir = scan_tag.experiment
        self.auxiliary_file_path = self.scan_directory / f"ScanData{self.scan_directory.name}.txt"
        self.ini_file_path = self.scan_directory / f"ScanInfo{self.scan_directory.name}.ini"
        self.noscan = False

        self.device_name = device_name

        self.skip_plt_show = skip_plt_show

        self.bins = None
        self.auxiliary_data: Optional[pd.DataFrame] = None
        self.binned_param_values = None
        
        scan_path = self.scan_directory.parent.parent / 'analysis' / self.scan_directory.name
        self.scan_path: Path = Path(scan_path)
        logging.info(f'analysis path is : {scan_path}')
        if clear_display_contents_file:
            self.remove_yaml_files(self.scan_path)


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

    @staticmethod
    def _load_display_content(yaml_path: Path) -> List[Dict[str, str]]:
        """
        Load existing display content from the YAML file. If the file does not exist, create an empty file.

        Args:
            yaml_path (Path): Path to the YAML file.
        Returns:
            List[Dict[str, str]]: Existing display content or an empty list if the file doesn't exist.
        """
        print(f"Attempting to load display content from: {yaml_path}")

        if not yaml_path.exists():
            print(f"File does not exist. Creating an empty YAML file at: {yaml_path}")
            yaml_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directories exist
            with yaml_path.open("w") as yaml_file:
                yaml.dump([], yaml_file)  # Write an empty list to the file

        print(f"File exists or has been created: {yaml_path}")

        with yaml_path.open("r") as yaml_file:
            existing_content = yaml.safe_load(yaml_file) or []
            if not isinstance(existing_content, list):
                existing_content = []
            return existing_content


    @staticmethod
    def _write_display_content(yaml_path: Path, content: List[Dict[str, str]]) -> None:
        """
        Write the updated display content to the YAML file.
        Args:
            yaml_path (Path): Path to the YAML file.
            content (List[Dict[str, str]]): List of content dictionaries to write.
        """
        print(f"Attempting to write display content {content} to: {yaml_path}")
        with yaml_path.open("w") as yaml_file:
            yaml.dump(content, yaml_file)
    
    def remove_yaml_files(self, directory_path: Path) -> None:
        """
        Remove all .yaml files in the specified directory.

        Args:
            directory_path (Path): Path object representing the directory.
        """
        if not directory_path.is_dir():
            raise ValueError(f"The provided path {directory_path} is not a valid directory.")

        for file_path in directory_path.glob("*.yaml"):  # Match only .yaml files
            if file_path.is_file():
                file_path.unlink()  # Remove the file
                print(f"Deleted: {file_path}")
        print(f"All .yaml files in {directory_path} have been removed.")
    
    def append_display_content(
        self, 
        content_path: str, 
        content_type: str = "file", 
        description: str = ""
    ) -> None:
        """
        Append a display content dictionary to the centralized YAML file.
        Args:
            content_path (str): Path to the content to append.
            content_type (str): Type of the content (e.g., "image", "data").
            description (str): Description of the content.
        """
        yaml_path: Path = self.scan_path / "display_content.yaml"
        logging.info(f"Appending display content to {yaml_path}")

        # Load the current display content from the file
        if yaml_path.exists():
            with yaml_path.open("r") as yaml_file:
                current_content = yaml.safe_load(yaml_file) or {}
        else:
            current_content = {}

        # Ensure the top-level key exists
        if "display_content" not in current_content:
            current_content["display_content"] = {}

        # Autogenerate a unique entry key
        entry_count = len(current_content["display_content"]) + 1
        entry_key = f"entry {entry_count}"

        # Create the new content dictionary
        new_entry = {
            "path": content_path, 
            "type": content_type, 
            "description": description
        }

        logging.info(f"New entry to append under key '{entry_key}': {new_entry}")

        # Append the new entry under the autogenerated key
        current_content["display_content"][entry_key] = new_entry

        # Write the updated content back to the file
        with yaml_path.open("w") as yaml_file:
            yaml.dump(current_content, yaml_file)
        logging.info(f"Updated content written to {yaml_path}")


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
        config = configparser.ConfigParser()
        config.read(self.ini_file_path)
        cleaned_scan_parameter = config['Scan Info']['Scan Parameter'].strip().replace(':', ' ').replace('"', '')
        return cleaned_scan_parameter

    def load_auxiliary_data(self):
        """
        Load auxiliary binning data from the ScanData file and retrieve the binning structure.
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
    pass


if __name__ == "__main__":
    testing_routine()
