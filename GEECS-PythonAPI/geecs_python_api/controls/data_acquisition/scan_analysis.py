import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import configparser
import logging
import yaml

import cv2

from scipy.ndimage import median_filter

from .utils import get_full_config_path  # Import the utility function

import re

from image_analysis.utils import read_imaq_png_image

# %% base scan analysis class
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

# %% mag spec analysis classes
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

# %% camera analysis classes
class CameraImageAnalysis(ScanAnalysis):

    def __init__(self, scan_directory, device_name, use_gui=True, experiment_dir = 'Undulator'):
        """
        Initialize the CameraImageAnalysis class.

        Args:
            scan_directory (str or Path): Path to the scan directory containing data.
            device_name (str): Name of the device to construct the subdirectory path.
        """
        super().__init__(scan_directory, use_gui=use_gui)  # Pass use_gui to the parent class
        # Construct the subdirectory for images based on device name
        self.device_name = device_name
        self.data_subdirectory = Path(scan_directory) / f"{device_name}"
        self.experiment_dir = experiment_dir

        self.camera_analysis_config_path = get_full_config_path(self.experiment_dir, 'aux_configs', 'camera_analysis_settings.yaml')
        self.camera_analysis_configs = None



        if self.camera_analysis_config_path.exists():
            self.load_camera_analysis_config()

        # Check if data directory exists and is not empty
        if not self.data_subdirectory.exists() or not any(self.data_subdirectory.iterdir()):
            logging.warning(f"Warning: Data directory '{self.data_subdirectory}' does not exist or is empty. Skipping analysis.")
            self.data_subdirectory = None

    def load_camera_analysis_config(self):

        """
        Load the master camera configs from the given YAML file.

        Returns:
            dict: A dictionary of analysis configs loaded from the YAML file.
        """

        camaera_analysis_configs_file = str(self.camera_analysis_config_path)  # Convert Path object to string
        with open(camaera_analysis_configs_file, 'r') as file:
            self.camera_analysis_configs = yaml.safe_load(file)

    def extract_shot_number(self, filename):
        """
        Extract the shot number from the filename.

        Args:
            filename (str): The filename from which to extract the shot number.

        Returns:
            int: The extracted shot number, or None if the format is incorrect.
        """
        # Match the last number before the .png extension
        match = re.search(r'_(\d+)\.png$', filename)
        if match:
            return int(match.group(1))
        return None

    def load_images_for_bin(self, bin_number):
        """
        Load all images corresponding to a specific bin number by matching the shot number.

        Args:
            bin_number (int): The bin number for which to load images.

        Returns:
            list of np.ndarray: List of images in the bin.
        """
        images = []
        shots_in_bin = self.auxiliary_data[self.auxiliary_data['Bin #'] == bin_number]['Shotnumber'].values
        expected_image_size = None

        for shot_num in shots_in_bin:
            image_file = next(self.data_subdirectory.glob(f'*_{shot_num:03d}.png'), None)
            if image_file:
                image = read_imaq_png_image(image_file)
                images.append(image)
                if expected_image_size is None:
                    expected_image_size = image.shape  # Determine image size from the first valid image
            else:
                logging.warning(f"Missing data for shot {shot_num}, adding zero array.")
                if expected_image_size:
                    images.append(np.zeros(expected_image_size, dtype=np.uint16))

        return images

    def average_images(self, images):
        """
        Average a list of images.

        Args:
            images (list of np.ndarray): List of images to average.

        Returns:
            np.ndarray: The averaged image.
        """
        if len(images) == 0:
            return None

        return np.mean(images, axis=0).astype(np.uint16)  # Keep 16-bit format for the averaged image

    def save_image(self, image, save_dir, save_name, bit_depth=16):
        """
        Save an image as a .png file.

        Args:
            image (np.ndarray): The image to save.
            save_dir (str or Path): Directory where the image will be saved.
            save_name (str): The name of the saved image file.
            bit_depth (int): The bit depth of the saved image (default is 16-bit).
        """
        save_path = Path(save_dir) / save_name

        # Save in 16-bit for the averaged image
        if bit_depth == 16:
            plt.imsave(save_path, image, cmap='gray', format='png', vmin=0, vmax=65535)  # 16-bit depth
        else:
            plt.imsave(save_path, image, cmap='gray')  # For 8-bit visualization

        logging.info(f"Image saved at {save_path}")

    def normalize_for_visualization(self, image):
        """
        Normalize a 16-bit image to 8-bit for visualization purposes.

        Args:
            image (np.ndarray): The 16-bit image to normalize.

        Returns:
            np.ndarray: The normalized 8-bit image.
        """
        # Normalize 16-bit image to 8-bit by scaling the pixel values
        normalized_image = (image/(2**16-1)*255).astype(np.uint8)

        return normalized_image

    def create_image_array(self, avg_images):
        """
        Arrange the averaged images into a sensibly sized grid and display them with scan parameter labels.
        For visualization purposes, images will be normalized to 8-bit.

        Args:
            avg_images (list of np.ndarray): List of averaged images.
        """
        if len(avg_images) == 0:
            logging.warning("No averaged images to arrange into an array.")
            return

        # Calculate grid size for arranging images in a square-like layout
        num_images = len(avg_images)
        grid_cols = int(np.ceil(np.sqrt(num_images)))
        grid_rows = int(np.ceil(num_images / grid_cols))

        # Create a figure with the appropriate number of subplots
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))

        # Flatten axes array for easy indexing (if there's only one row/col, axs won't be a 2D array)
        axs = axs.flatten()

        # # Flatten all images in the batch to find global percentiles
        all_pixels = np.concatenate([img.ravel() for img in avg_images if img is not None])

        max_val = np.max(all_pixels)
        low = 0
        high = max_val

        for i, (img, param_value) in enumerate(zip(avg_images, self.binned_param_values)):

            # Display with adjusted scale
            axs[i].imshow(img, cmap='plasma', vmin=low, vmax=high)
            axs[i].set_title(f'{self.scan_parameter}: {param_value:.2f}', fontsize=10)  # Use scan parameter for label
            axs[i].axis('off')  # Turn off axes for cleaner display

        # Hide any unused subplots
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        # Save the final image grid for visualization
        save_name = f'{self.device_name}_averaged_image_grid.png'
        plt.tight_layout()
        plt.savefig(Path(self.scan_directory) / save_name, bbox_inches='tight')
        logging.info(f"Saved final image grid as {save_name}.")
        self.close_or_show_plot()

    def crop_image(self, image, analysis_settings):
        """
        This function loads uses predefined analysis_settings to crop an image.

        Args:
        - image: loaded image to be processed.

        """

        cropped_image = image[analysis_settings['Top ROI']:analysis_settings['Top ROI'] + analysis_settings['Size_Y'],
                              analysis_settings['Left ROI']:analysis_settings['Left ROI'] + analysis_settings['Size_X']]

        return cropped_image

    def run_analysis(self):
        """
        Main function to run the image analysis.

        MAY BE DEPRECATED. DEDICATED RUN ANALYSIS FOR VISA IMAGES EXISTS BELOW.
        SHOULD REDUCE FOR BASE IMAGE ANALYSIS.
        """

        if self.data_subdirectory is None or self.auxiliary_data is None:
            logging.info(f"Skipping analysis due to missing data or auxiliary file.")
            return

        avg_images = []
        try:
            unique_bins = np.unique(self.bins)
            logging.info(f"unique_bins: {unique_bins}")

            for bin_number in unique_bins:
                # Load all images for this bin
                images = self.load_images_for_bin(bin_number)

                if len(images) == 0:
                    logging.warning(f"No images found for bin {bin_number}.")
                    continue

                # Average the images
                avg_image = self.average_images(images)


                if self.device_name in self.camera_analysis_configs.keys():
                    analysis_settings = self.camera_analysis_configs[self.device_name]
                    avg_image_processed = self.crop_image(avg_image, analysis_settings)
                    avg_images.append(avg_image_processed)

                    # Save the averaged image
                    save_name = f'{self.device_name}_{bin_number}_processed.png'
                    self.save_image(avg_image_processed, save_dir=self.scan_directory, save_name=save_name)

                else:
                    avg_images.append(avg_image)


                # Save the averaged image
                save_name = f'{self.device_name}_{bin_number}.png'
                self.save_image(avg_image, save_dir=self.scan_directory, save_name=save_name)

                logging.info(f"Averaged images for bin {bin_number} and saved as {save_name}.")

            # Once all bins are processed, create an array of the averaged images
            self.create_image_array(avg_images)

        except Exception as e:
            logging.warning(f"Warning: Image analysis failed due to: {e}")
            return

class VisaEBeamAnalysis(CameraImageAnalysis):

    def __init__(self, scan_directory, device_name, use_gui=True, experiment_dir = 'Undulator'):
        """
        Initialize the CameraImageAnalysis class.

        Args:
            scan_directory (str or Path): Path to the scan directory containing data.
            device_name (str): Name of the device to construct the subdirectory path.
        """
        super().__init__(scan_directory, device_name, use_gui=use_gui)

    def create_cross_mask(self, image, cross_center, angle, cross_height=54,
                          cross_width=54, thickness=8):
        """
        Creates a mask with a cross centered at `cross_center` with the cross being zeros and the rest ones.

        Args:
        - image (np.array): The image on which to base the mask size.
        - cross_center (tuple): The (x, y) center coordinates of the cross.
        - cross_height (int): The height of the cross extending vertically from the center.
        - cross_width (int): The width of the cross extending horizontally from the center.
        - thickness (int): The thickness of the lines of the cross.

        Returns:
        - np.array: The mask with the cross.
        """
        mask = np.ones_like(image, dtype=np.uint16)
        x_center, y_center = cross_center
        vertical_start, vertical_end = max(y_center - cross_height, 0), min(y_center + cross_height, image.shape[0])
        horizontal_start, horizontal_end = max(x_center - cross_width, 0), min(x_center + cross_width, image.shape[1])
        mask[vertical_start:vertical_end, x_center - thickness // 2:x_center + thickness // 2] = 0
        mask[y_center - thickness // 2:y_center + thickness // 2, horizontal_start:horizontal_end] = 0

        M = cv2.getRotationMatrix2D(cross_center, angle, 1.0)
        rotated_mask = cv2.warpAffine(mask, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
        return rotated_mask

    def preprocess_visa_image(self, image, analysis_settings):
        """
        This function loads uses predefined analysis_settings to crop an image and mask out "crosses" from the images.

        Args:
        - image: loaded image to be processed.

        """
        mask1 = self.create_cross_mask(image, analysis_settings['Cross1'], analysis_settings['Rotate'])
        mask2 = self.create_cross_mask(image, analysis_settings['Cross2'], analysis_settings['Rotate'])

        masked_image = image * mask1 * mask2
        cropped_image = masked_image[analysis_settings['Top ROI']:analysis_settings['Top ROI'] + analysis_settings['Size_Y'],
                              analysis_settings['Left ROI']:analysis_settings['Left ROI'] + analysis_settings['Size_X']]

        processed_image = cropped_image

        return processed_image

    def run_analysis(self):
        """
        Main function to UC_VisaEBeam image analysis.
        """

        # skip analysis if data is missing
        if self.data_subdirectory is None or self.auxiliary_data is None:
            logging.info(f"Skipping analysis due to missing data or auxiliary file.")
            return

        # preallocate storage
        avg_images = []

        # attempt analysis
        try:

            # identify unique parameter bins
            unique_bins = np.unique(self.bins)
            logging.info(f"unique_bins: {unique_bins}")

            # iterate parameter bins
            for bin_number in unique_bins:

                # load all images for this bin
                images = self.load_images_for_bin(bin_number)
                if len(images) == 0:
                    logging.warning(f"No images found for bin {bin_number}.")
                    continue

                # average the images
                avg_image = self.average_images(images)

                # check if camera analysis settings exist in config
                if self.device_name in self.camera_analysis_configs.keys():

                    # extract analysis settings, apply preprocessing routine
                    analysis_settings = self.camera_analysis_configs[self.device_name]
                    avg_image_processed = self.preprocess_visa_image(avg_image, analysis_settings)
                    avg_images.append(avg_image_processed)

                    # Save the averaged image
                    save_name = f'{self.device_name}_{bin_number}_processed.png'
                    self.save_image(avg_image_processed, save_dir=self.scan_directory, save_name=save_name)

                else:
                    avg_images.append(avg_image)

                # save the averaged image
                save_name = f'{self.device_name}_{bin_number}.png'
                self.save_image(avg_image, save_dir=self.scan_directory, save_name=save_name)

                logging.info(f"Averaged images for bin {bin_number} and saved as {save_name}.")

            # Once all bins are processed, create an array of the averaged images
            self.create_image_array(avg_images)

        # throw warning if analysis fails
        except Exception as e:
            logging.warning(f"Warning: Image analysis failed due to: {e}")
            return
