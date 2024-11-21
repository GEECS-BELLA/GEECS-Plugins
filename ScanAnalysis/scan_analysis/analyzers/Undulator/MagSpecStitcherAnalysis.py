# -*- coding: utf-8 -*-
"""
B-cave Magnetic Spectrometer Stitcher Analysis

Child to ScanAnalysis (./scan_analysis/base.py)
"""
# %% imports
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scan_analysis.base import ScanAnalysis

# %% classes
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
        Load charge-per-energy data from files, using shot numbers from auxiliary data.
        If data for a shot is missing, replace with properly sized arrays of zeros.
        """
        charge_density_matrix = []
        energy_values = None
        missing_shots_placeholder = []  # To keep track of shots missing data

        # Get the shot numbers from the auxiliary data
        shot_numbers = self.auxiliary_data['Shotnumber'].values

        for shot_num in shot_numbers:
            try:
                # Check if a file exists for this shot
                shot_file = next(self.data_subdirectory.glob(f'*_{shot_num:03d}.txt'), None)

                if shot_file:
                    data = pd.read_csv(shot_file, delimiter='\t')
                    if energy_values is None:
                        energy_values = data.iloc[:, 0].values  # Initialize energy values from first valid file

                    charge_density_matrix.append(data.iloc[:, 1].values)  # Second column is charge density
                else:
                    # If no file found for this shot, append a placeholder (None) to handle later
                    logging.warning(f"Missing data for shot {shot_num}, adding placeholder.")
                    charge_density_matrix.append(None)
                    missing_shots_placeholder.append(len(charge_density_matrix) - 1)  # Track the index of missing shots

            except Exception as e:
                logging.error(f"Error reading data for shot {shot_num}: {e}")
                charge_density_matrix.append(None)  # Append None in case of error
                missing_shots_placeholder.append(len(charge_density_matrix) - 1)

        # After the loop, if energy_values is still None, log an error and exit
        if energy_values is None:
            logging.error("No valid shot data found. Cannot proceed with analysis.")
            return None, None

        # Replace all placeholders with zero arrays of the correct size
        for idx in missing_shots_placeholder:
            charge_density_matrix[idx] = np.zeros_like(energy_values)

        return energy_values, np.array(charge_density_matrix)


    def interpolate_data(self, energy_values, charge_density_matrix, min_energy=0.06, max_energy=.3, num_points=1500):
        """
        Linearly interpolate the data for plotting.
        """
        linear_energy_axis = np.linspace(min_energy, max_energy, num_points)
        interpolated_matrix = np.empty((charge_density_matrix.shape[0], num_points))

        for i, row in enumerate(charge_density_matrix):
            try:
                interpolated_matrix[i] = np.interp(linear_energy_axis, energy_values, row)
            except Exception as e:
                logging.warning(f"Interpolation failed for shot {i}. Using zeros. Error: {e}")
                interpolated_matrix[i] = np.zeros(num_points)  # Handle any interpolation failure

            # interpolated_matrix[i] = np.interp(linear_energy_axis, energy_values, row)

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
            logging.info(f"Plot saved to {save_path}")

    def run_analysis(self):
        """
        Main function to run the analysis and generate plots.
        """
        if self.data_subdirectory is None or self.auxiliary_data is None:
            logging.info(f"Skipping analysis due to missing data or auxiliary file.")
            return

        try:
            energy_values, charge_density_matrix = self.load_charge_data()

            if energy_values is None or len(charge_density_matrix) == 0:
                logging.error("No valid charge data found. Skipping analysis.")
                return

            # Interpolate for unbinned plot
            linear_energy_axis, interpolated_charge_density_matrix = self.interpolate_data(energy_values, charge_density_matrix)

            shot_num_labels = self.generate_limited_shotnumber_labels(self.total_shots, max_labels = 20)
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

# %% executable

def testing_routine():

    from geecs_python_api.controls.data_acquisition.data_acquisition import DataInterface

    # define scan information
    scan = {'year': '2024',
            'month': 'Nov',
            'day': '21',
            'num': 3}
    # device_name = "UC_VisaEBeam1"
    device_name = "U_BCaveMagSpec"

    # initialize data interface and analysis class
    data_interface = DataInterface()
    data_interface.year = scan['year']
    data_interface.month = scan['month']
    data_interface.day = scan['day']
    (raw_data_path,
     analysis_data_path) = data_interface.create_data_path(scan['num'])

    scan_directory = raw_data_path / f"Scan{scan['num']:03d}"
    analysis_class = MagSpecStitcherAnalysis(scan_directory, device_name)

    analysis_class.run_analysis()

if __name__=="__main__":
    testing_routine()
