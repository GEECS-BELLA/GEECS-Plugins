import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import configparser
import logging

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
    def __init__(self, scan_directory):
        """
        Initialize the ScanAnalysis class.

        Args:
            scan_directory (str or Path): Path to the scan directory containing data.
        """
        self.scan_directory = Path(scan_directory)
        self.auxiliary_file_path = self.scan_directory / f"ScanData{self.scan_directory.name}.txt"
        self.ini_file_path = self.scan_directory / f"ScanInfo{self.scan_directory.name}.ini"
        self.noscan = False
        
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

class MagSpecStitcherAnalysis(ScanAnalysis):
    def __init__(self, scan_directory, device_name):
        super().__init__(scan_directory)
        # self.data_subdirectory = Path(scan_directory) / data_subdirectory
        self.data_subdirectory = Path(scan_directory) / f"{device_name}-interpSpec"

        # Check if data directory exists and is not empty
        if not self.data_subdirectory.exists() or not any(self.data_subdirectory.iterdir()):
            logging.warning(f"Warning: Data directory '{self.data_subdirectory}' does not exist or is empty. Skipping analysis.")
            self.data_subdirectory = None

    def load_charge_data(self):
        """
        Load charge-per-energy data from files.
        """
        charge_data_files = sorted(self.data_subdirectory.glob('*.txt'))
        charge_density_matrix = []
        energy_values = None

        for file in charge_data_files:
            data = pd.read_csv(file, delimiter='\t')
            if energy_values is None:
                energy_values = data.iloc[:, 0].values
            charge_density_matrix.append(data.iloc[:, 1].values)

        return energy_values, np.array(charge_density_matrix)

    def interpolate_data(self, energy_values, charge_density_matrix, min_energy=0.06, max_energy=0.2, num_points=1000):
        """
        Linearly interpolate the data for plotting.
        """
        linear_energy_axis = np.linspace(min_energy, max_energy, num_points)
        interpolated_matrix = np.empty((charge_density_matrix.shape[0], num_points))

        for i, row in enumerate(charge_density_matrix):
            interpolated_matrix[i] = np.interp(linear_energy_axis, energy_values, row)

        return linear_energy_axis, interpolated_matrix

    def bin_data(self, charge_density_matrix):
        """
        Bin the data using the bin numbers from the auxiliary data.
        """
        binned_matrix = []
        unique_bins = np.unique(self.bins)

        for bin_num in unique_bins:
            binned_matrix.append(np.mean(charge_density_matrix[self.bins == bin_num], axis=0))

        return np.array(binned_matrix)

    def plot_waterfall_with_labels(self, energy_axis, charge_density_matrix, title, ylabel, vertical_values, save_dir=None, save_name=None):
        """
        Plot the waterfall chart with centered labels on the vertical axis.
        """
        plt.figure(figsize=(10, 6))

        # Set y-axis ticks to correspond to the vertical_values
        y_ticks = np.arange(1, len(vertical_values) + 1)

        # Adjust the extent to shift y-values by 0.5 to center the labels
        plt.imshow(charge_density_matrix, aspect='auto', cmap='plasma',
                   extent=[energy_axis[0], energy_axis[-1], y_ticks[-1] + 0.5, y_ticks[0] - 0.5], interpolation='none')

        plt.colorbar(label='Charge Density (pC/GeV)')
        plt.xlabel('Energy (GeV/c)')

        # Set y-axis labels to be the vertical values, centering them within the rows
        plt.yticks(y_ticks, labels=[f"{v:.2f}" for v in vertical_values])
        plt.ylabel(ylabel)
        plt.title(title)

        # If save_dir and save_name are provided, save the plot
        if save_dir and save_name:
            save_path = Path(save_dir) / save_name
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

    def run_analysis(self):
        """
        Main function to run the analysis and generate plots.
        """
        if self.data_subdirectory is None or self.auxiliary_data is None:
            logging.info(f"Skipping analysis due to missing data or auxiliary file.")
            return

        try:
            energy_values, charge_density_matrix = self.load_charge_data()

            # Interpolate for unbinned plot
            linear_energy_axis, interpolated_charge_density_matrix = self.interpolate_data(energy_values, charge_density_matrix)

            shot_num_labels = self.generate_limited_shotnumber_labels(self.total_shots, max_labels = 20)
            print(shot_num_labels)
            # Plot unbinned data
            self.plot_waterfall_with_labels(linear_energy_axis, interpolated_charge_density_matrix,
                                            f'{str(self.scan_directory)}', 'Shotnumber',
                                            # vertical_values = np.arange(1, len(interpolated_charge_density_matrix) + 1),
                                            vertical_values = shot_num_labels,
                                            save_dir=self.scan_directory, save_name='charge_density_vs_shotnumber.png')

            # Skip binning if noscan
            if self.noscan:
                logging.info("No scan performed, skipping binning and binned plots.")
                return  # Skip the rest of the analysis

            # Bin the data and plot binned data
            binned_matrix = self.bin_data(charge_density_matrix)
            linear_energy_axis_binned, interpolated_binned_matrix = self.interpolate_data(energy_values, binned_matrix)

            # Plot the binned waterfall plot using the average scan parameter values for the vertical axis
            self.plot_waterfall_with_labels(linear_energy_axis_binned, interpolated_binned_matrix,
                                            f'{str(self.scan_directory)}', self.find_scan_param_column(self.auxiliary_data)[1],
                                            vertical_values=self.binned_param_values,
                                            save_dir=self.scan_directory, save_name='charge_density_vs_scan_parameter.png')

        except Exception as e:
            logging.warning(f"Warning: Analysis failed due to: {e}")
            return