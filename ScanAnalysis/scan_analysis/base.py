"""
Class containing common functionality and requirements available for all scan analyzers
"""
# %% imports
from pathlib import Path
import logging
import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% classes
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
    def __init__(self, scan_directory, use_gui = True, experiment_dir = "Undulator"):

        """
        Initialize the ScanAnalysis class.

        Args:
            scan_directory (str or Path): Path to the scan directory containing data.
        """
        self.scan_directory = Path(scan_directory)
        self.experiment_dir = experiment_dir
        self.auxiliary_file_path = self.scan_directory / f"ScanData{self.scan_directory.name}.txt"
        self.ini_file_path = self.scan_directory / f"ScanInfo{self.scan_directory.name}.ini"
        self.noscan = False
        self.use_gui = use_gui

        try:
            # Extract the scan parameter
            self.scan_parameter = self.extract_scan_parameter_from_ini(self.ini_file_path)

            logging.info(f"Scan parameter is: {self.scan_parameter}.")
            s_param = self.scan_parameter.lower()

            if s_param == 'noscan' or s_param == 'shotnumber':
                logging.warning("No parameter varied during the scan, setting noscan flag.")
                self.noscan = True

            self.bins, self.auxiliary_data, self.binned_param_values = self.load_auxiliary_data()

            if self.auxiliary_data is None:
                logging.warning("Scan parameter not found in auxiliary data. Possible aborted scan. Skipping analysis.")
                return  # Stop further execution cleanly

            self.total_shots = len(self.auxiliary_data)

        except FileNotFoundError as e:
            logging.warining(f"Warning: {e}. Could not find auxiliary or .ini file in {self.scan_directory}. Skipping analysis.")
            return

    def analyze_scan(self):  #TODO add Scantag as argument (maybe config for options??)
        raise NotImplementedError

    def extract_scan_parameter_from_ini(self, ini_file_path):
        """
        Extract the scan parameter from the .ini file.

        Args:
            ini_file_path (Path): Path to the .ini file.

        Returns:
            str: The scan parameter with colons replaced by spaces.
        """
        config = configparser.ConfigParser()
        config.read(ini_file_path)
        cleaned_scan_parameter = config['Scan Info']['Scan Parameter'].strip().replace(':', ' ').replace('"', '')
        return cleaned_scan_parameter

    def load_auxiliary_data(self):
        """
        Load auxiliary binning data from the ScanData file and retrieve the binning structure.

        Returns:
            tuple: A tuple containing the bin numbers (np.ndarray) and the auxiliary data (pd.DataFrame).
        """

        try:
            auxiliary_data = pd.read_csv(self.auxiliary_file_path, delimiter='\t')
            bins = auxiliary_data['Bin #'].values

            if not self.noscan:
                # Find the scan parameter column and calculate the binned values
                scan_param_column = self.find_scan_param_column(auxiliary_data)[0]
                binned_param_values = auxiliary_data.groupby('Bin #')[scan_param_column].mean().values

                return bins, auxiliary_data, binned_param_values

            else:
                return bins, auxiliary_data, None

        except (KeyError, FileNotFoundError) as e:
            logging.warning(f"Warning: {e}. Scan parameter not found in auxiliary data. Possible aborted scan. Skipping analysis")
            return None, None, None

    def close_or_show_plot(self):
        """Decide whether to display or close plots based on the use_gui setting."""
        if not self.use_gui:
            plt.show()  # Display for interactive use
        else:
            plt.close('all')  # Ensure plots close when not using the GUI

    def generate_limited_shotnumber_labels(self, total_shots, max_labels=20):
        """
        Generate a list of shot number labels with a maximum of `max_labels`.

        Args:
            total_shots (int): Total number of shots.
            max_labels (int): Maximum number of labels to display.

        Returns:
            np.ndarray: Array of shot numbers, spaced out if necessary.
        """
        if total_shots <= max_labels:
            # If the number of shots is less than or equal to max_labels, return the full range
            return np.arange(1, total_shots + 1)
        else:
            # Otherwise, return a spaced-out array with at most max_labels
            step = total_shots // max_labels
            return np.arange(1, total_shots + 1, step)

    def find_scan_param_column(self, auxiliary_data):
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
            for column in auxiliary_data.columns:
                # Match the part of the column before 'Alias:'
                if self.scan_parameter in column.split(' Alias:')[0]:
                    # Return the column and the alias if present
                    return column, column.split('Alias:')[-1].strip() if 'Alias:' in column else column

            logging.warning(f"Warning: Could not find column containing scan parameter: {self.scan_parameter}")
            return None, None

# %% executable
def testing_routine():
    pass

if __name__=="__main__":
    testing_routine()
