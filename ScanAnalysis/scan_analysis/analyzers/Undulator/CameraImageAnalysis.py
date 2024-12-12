"""
Camera Image Analysis

General camera image analyzer.
Child to ScanAnalysis (./scan_analysis/base.py)
"""
# %% imports
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional

if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag
from pathlib import Path
import logging
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter
import imageio as io

from scan_analysis.base import ScanAnalysis
from geecs_python_api.controls.data_acquisition.utils import get_full_config_path
from image_analysis.utils import read_imaq_png_image


# %% classes
class CameraImageAnalysis(ScanAnalysis):

    def __init__(self, scan_tag: ScanTag, device_name: str, skip_plt_show: bool = True,
                 flag_logging: bool = True, flag_save_images: bool = True):
        """
        Initialize the CameraImageAnalysis class.

        Args:
            scan_tag (ScanTag): Path to the scan directory containing data.
            device_name (str): Name of the device to construct the subdirectory path.
            skip_plt_show (bool): Flag that sets if matplotlib is tried to use for plotting
            flag_logging (bool): Flag that sets if error and warning messages are displayed
            flag_save_images (bool): Flag that sets if images are saved to disk
        """
        if device_name is None:
            raise ValueError(f"CameraImageAnalysis requires a device_name to be specified.")

        super().__init__(scan_tag, device_name=device_name, skip_plt_show=skip_plt_show)

        # define flags
        self.flag_logging = flag_logging
        self.flag_save_images = flag_save_images

        # organize various paths
        self.path_dict = {'data_img': Path(self.scan_directory) / f"{device_name}",
                          'save': (self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name
                                   / f"{device_name}" / "CameraImageAnalysis"),
                          'cam_configs': get_full_config_path(self.experiment_dir, 'aux_configs',
                                                              'camera_analysis_settings.yaml')
                          }

        # if self.camera_analysis_config_path.exists():
        self.camera_analysis_configs = None
        self.camera_analysis_settings = None
        if self.path_dict['cam_configs'].exists():
            self.camera_analysis_configs = self.load_camera_analysis_config()
            if self.device_name in self.camera_analysis_configs:
                self.camera_analysis_settings = self.camera_analysis_configs[self.device_name]

        # Check if data directory exists and is not empty
        if not self.path_dict['data_img'].exists() or not any(self.path_dict['data_img'].iterdir()):
            if self.flag_logging:
                logging.warning(f"Data directory '{self.path_dict['data_img']}' does not exist or is empty. Skipping")

    def run_analysis(self, config_options: Optional[str] = None):
        # initialize analysis
        if self.path_dict['data_img'] is None or self.auxiliary_data is None:
            if self.flag_logging:
                logging.info(f"Skipping analysis due to missing data or auxiliary file.")
            return

        # initialize various analysis parameters
        if config_options is not None:
            # TODO read the given file to `self.camera_analysis_settings`
            raise NotImplementedError

        # if saving, make sure save location exists
        if self.flag_save_images and not self.path_dict['save'].exists():
            self.path_dict['save'].mkdir(parents=True)

        # delegate analysis type
        try:
            if self.noscan:
                self.run_noscan_analysis()
            else:
                self.run_scan_analysis()

        except Exception as e:
            if self.flag_logging:
                logging.warning(f"Warning: Image analysis failed due to: {e}")
            return

    def load_camera_analysis_config(self) -> dict:

        """
        Load the master camera configs from the given YAML file.

        Returns:
            dict: A dictionary of analysis configs loaded from the YAML file.
        """

        camera_analysis_configs_file = str(self.path_dict['cam_configs'])  # convert Path object to string
        with open(camera_analysis_configs_file, 'r') as file:
            camera_analysis_configs = yaml.safe_load(file)

        return camera_analysis_configs

    def save_fig(self, save_path: Path,
                 bbox_inches: str = 'tight', pad_inches: float = 0.) -> None:

        # ensure save directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # save image
        plt.savefig(save_path, bbox_inches=bbox_inches, pad_inches=pad_inches)


    def save_geecs_scaled_image(self, image: np.ndarray, save_dir: Union[str, Path],
                                save_name: str, bit_depth: int = 16):
        """
        Images saved through GEECS typically are saved as 16bit, but the hardware saves
        12 bit. In other words, the last 4 bits are unused. This method will save as
        16 bit image or, if 8bit representation is desired for visualization, it will
        scale to 8 bits properly

        Args:
            image (np.ndarray): The image to save.
            save_dir (str or Path): Directory where the image will be saved.
            save_name (str): The name of the saved image file.
            bit_depth (int): The bit depth of the saved image (default is 16-bit).
        """
        save_path = Path(save_dir) / save_name

        # Ensure the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to 16-bit if required
        if bit_depth == 16:
            if image.dtype == np.uint8:
                image = (image.astype(np.uint16)) * 256  # Scale 8-bit to 16-bit
            elif image.dtype != np.uint16:
                raise ValueError("Image must be either 8-bit or 16-bit format.")
        elif bit_depth == 8:
            image = (image * 2 ** 4).astype(np.uint8)
        else:
            raise ValueError("Unsupported bit depth. Only 8 or 16 bits are supported.")

        # Save the image using cv2
        cv2.imwrite(str(save_path), image)
        if self.flag_logging:
            logging.info(f"Image saved at {save_path}")

    def save_normalized_image(self, image: np.ndarray, save_dir: Union[str, Path], save_name: str):
        """Display and optionally save a 16-bit image with specified min/max values for visualization."""

        max_val = np.max(image)

        # Display the image
        plt.clf()

        plt.imshow(image, cmap='gray', vmin=0, vmax=max_val)
        plt.colorbar()  # Adds a color scale bar
        plt.axis('off')  # Hide axis for a cleaner look

        # Save the image if a path is provided
        if save_dir and save_name:
            save_path = Path(save_dir) / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            if self.flag_logging:
                logging.info(f"Image saved at {save_path}")

    def create_image_array(self, binned_data: dict[dict], ref_coords: Optional[tuple] = None,
                           plot_scale: Optional[float] = None):
        """
        Arrange the averaged images into a sensibly sized grid and display them with scan parameter labels.
        For visualization purposes, images will be normalized to 8-bit.

        Args:
            binned_data (dict[dict]): List of averaged images. TODO for faster speed consider making a numpy array
            ref_coords (tuple): The x and y data to be plotted as a reference, as element 0 and 1, respectively
            plot_scale (float): A float value for the maximum color
        """
        if len(binned_data) == 0:
            if self.flag_logging:
                logging.warning("No averaged images to arrange into an array.")
            return

        # Calculate grid size for arranging images in a square-like layout
        num_images = len(binned_data)
        grid_cols = int(np.ceil(np.sqrt(num_images)))
        grid_rows = int(np.ceil(num_images / grid_cols))

        # get global color scale
        all_pixels = np.concatenate([binned_data[bnum]['image'].ravel()
                                     for bnum in list(binned_data.keys())
                                     if binned_data[bnum]['image'] is not None])
        low, high = 0, all_pixels.max()
        if plot_scale is not None:
            high = plot_scale

        # Create a figure with the appropriate number of subplots
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))

        # Flatten axes array for easy indexing (if there's only one row/col, axs won't be a 2D array)
        axs = axs.flatten()

        bin_ind = None
        for bin_num, bin_item in binned_data.items():
            bin_ind = bin_num - 1
            img = bin_item['image']
            param_value = bin_item['value']

            # display with adjusted scale
            axs[bin_ind].imshow(img, cmap='plasma', vmin=low, vmax=high)
            axs[bin_ind].set_title(f'{self.scan_parameter}: {param_value:.2f}',
                                   fontsize=10)  # Use scan parameter for label
            axs[bin_ind].axis('off')  # Turn off axes for cleaner display

            if ref_coords is not None:
                axs[bin_ind].plot(ref_coords[0], ref_coords[1], color='g', marker='o')

        for j in range(bin_ind + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()

        # Save the final image grid for visualization
        save_name = f'{self.device_name}_averaged_image_grid.png'
        plt.savefig(Path(self.path_dict['save']) / save_name, bbox_inches='tight')
        if self.flag_logging:
            logging.info(f"Saved final image grid as {save_name}.")
        self.close_or_show_plot()

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
        # expected_image_size = None

        for shot_num in shots_in_bin:
            image_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}.png'), None)
            if image_file:
                image = read_imaq_png_image(image_file)
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

    @staticmethod
    def average_images(images: list[np.ndarray]) -> Optional[np.ndarray]:
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

    def bin_images(self, flag_save=None):
        # TODO should we keep flag_save as an argument here or just use the class variable?
        # set default
        if flag_save is None:
            flag_save = self.flag_save_images

        # identify unique parameter bins
        unique_bins = np.unique(self.bins)
        if self.flag_logging:
            logging.info(f"unique_bins: {unique_bins}")

        # preallocate storage
        binned_data = {bins: {} for bins in unique_bins}

        # iterate parameter bins
        for bin_ind, bin_val in enumerate(unique_bins):

            # load all images for this bin
            images = self.load_images_for_bin(bin_val)  # TODO couldn't figure out the type hinting warning here...
            if len(images) == 0:
                if self.flag_logging:
                    logging.warning(f"No images found for bin {bin_val}.")
                continue

            # average the images
            # need to confirm this is foolproof, are binned_param_values completely correlated?
            avg_image = self.average_images(images)
            binned_data[bin_val] = {'value': self.binned_param_values[bin_ind],
                                    'image': avg_image}

            if flag_save:
                save_name = f"{self.device_name}_{bin_val}.png"
                self.save_geecs_scaled_image(avg_image,
                                             save_dir=self.path_dict['save'],
                                             save_name=save_name)

                if self.flag_logging:
                    logging.info(f"Averaged images for bin {bin_val} and saved as {save_name}.")

            else:
                if self.flag_logging:
                    logging.info(f"Averaged images for bin {bin_val} but not saved.")

        # check for empty bins and remove
        binned_data = {key: value for key, value in binned_data.items() if value}

        return binned_data  # TODO figure out type hinting for this.  Tried dict[dict] but Pycharm wasn't happy

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """
        This function loads uses predefined analysis_settings to crop an image.

        Args:
        - image: loaded image to be processed.

        """
        settings = self.camera_analysis_settings

        cropped_image = image[settings['Top ROI']:settings['Top ROI'] + settings['Size_Y'],
                              settings['Left ROI']:settings['Left ROI'] + settings['Size_X']]

        return cropped_image

    def filter_image(self, image: np.ndarray) -> np.ndarray:
        """
        This function uses predefined analysis_settings to filter an image.

        Args:
        - image: loaded image to be processed.
        - analysis_settings: settings for filtering.

        Returns:
        - Processed image with specified filters applied.
        """

        settings = self.camera_analysis_settings

        processed_image = image.astype(np.float32)  # Ensure a consistent numeric type

        for _ in range(settings.get('Median Filter Cycles', 0)):
            processed_image = median_filter(processed_image, size=settings['Median Filter Size'])

        for _ in range(settings.get('Gaussian Filter Cycles', 0)):
            processed_image = gaussian_filter(processed_image, sigma=settings['Gaussian Filter Size'])

        # Optionally cast back to a 16-bit integer if needed
        return processed_image.astype(np.uint16) if image.dtype == np.uint16 else processed_image

    def image_processing(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # crop image, save image
        cropped_image = self.crop_image(image)
        processed_image = cropped_image.copy()

        # filter image, save image
        if self.camera_analysis_settings.get('Median Filter Size', None):
            processed_image = self.filter_image(processed_image)

        return cropped_image, processed_image

    @staticmethod
    def create_gif(image_arrays: list[np.ndarray], output_file: str,
                   titles: Optional[list[str]] = None, duration: float = 100):

        # make default list of titles
        if titles is None:
            titles = [f"Shot {num + 1}" for num in range(len(image_arrays))]

        # define font parameters
        # font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1
        thickness = 2
        color = (255, 255, 255)

        images = []
        for img, title in zip(image_arrays, titles):

            # make a copy of the array
            img_copy = img.copy()

            # get image dimensions
            height, width = img_copy.shape[:2]

            # get title size
            (text_width, text_height), _ = cv2.getTextSize(title, font, font_scale, thickness)

            # calculate text position for center alignment
            x = (width - text_width) // 2
            y = 50  # distance from top

            # add text to image
            cv2.putText(img_copy, title,
                        (x, y), font, font_scale, color, thickness)

            # convert BGR to RGB if necessary
            if len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

            images.append(img_copy)

        # create gif
        io.mimsave(output_file, images, duration=duration, loop=0)

    def run_noscan_analysis(self):
        """
        Image analysis in the case of a no scan.

        """

        # load images
        data = {'shot_num': [], 'images': []}
        for shot_num in self.auxiliary_data['Shotnumber'].values:
            image_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}.png'), None)
            if image_file:
                data['shot_num'].append(shot_num)
                data['images'].append(read_imaq_png_image(image_file))
            else:
                if self.flag_logging:
                    logging.warning(f"Missing data for shot {shot_num}.")

        # perform image analysis
        for i in range(len(data['images'])):
            _, data['images'][i] = self.image_processing(data['images'][i])

        # get average image
        avg_image = self.average_images(data['images'])
        if self.flag_save_images:
            self.save_geecs_scaled_image(avg_image, save_dir=self.path_dict['save'],
                                         save_name=f'{self.device_name}_average_processed.png')
            self.save_normalized_image(avg_image, save_dir=self.path_dict['save'],
                                       save_name=f'{self.device_name}_average_processed_visual.png')

        # make gif
        if self.flag_save_images:
            filepath = self.path_dict['save'] / 'noscan.gif'
            self.create_gif(data['images'], filepath,
                            titles=[f"Shot {num}" for num in data['shot_num']])

    def run_scan_analysis(self):
        """
        Image analysis in the case of a scanned variable.

        """
        # bin data
        binned_data = self.bin_images(flag_save=self.flag_save_images)

        for bin_key, bin_item in binned_data.items():

            # perform basic image processing
            (cropped_image,
             processed_image) = self.image_processing(bin_item['image'])

            # overwrite stored image
            binned_data[bin_key]['image'] = processed_image

            # save figures
            if self.flag_save_images:
                self.save_geecs_scaled_image(cropped_image, save_dir=self.path_dict['save'],
                                             save_name=f"{self.device_name}_{bin_key}_cropped.png")
                self.save_geecs_scaled_image(processed_image, save_dir=self.path_dict['save'],
                                             save_name=f'{self.device_name}_{bin_key}_processed.png')
                self.save_normalized_image(processed_image, save_dir=self.path_dict['save'],
                                           save_name=f'{self.device_name}_{bin_key}_processed_visual.png')

        # Once all bins are processed, create an array of the averaged images
        if len(binned_data) > 1 and self.flag_save_images:
            plot_scale = self.camera_analysis_settings.get('Plot Scale', None)
            self.create_image_array(binned_data, plot_scale=plot_scale)  # TODO more to do with binned_data type hints


# %% executable
def testing_routine():
    from geecs_python_api.controls.data_acquisition.data_acquisition import DataInterface

    # define scan information
    scan = {'year': '2024',
            'month': 'Nov',
            'day': '26',
            'num': 13}
    device_name = "UC_ALineEBeam3"
    # device_name = "UC_UndulatorRad2"

    # initialize data interface and analysis class
    data_interface = DataInterface()
    data_interface.year = scan['year']
    data_interface.month = scan['month']
    data_interface.day = scan['day']
    (raw_data_path,
     analysis_data_path) = data_interface.create_data_path(scan['num'])

    scan_directory = raw_data_path / f"Scan{scan['num']:03d}"
    analysis_class = CameraImageAnalysis(scan_directory, device_name)

    analysis_class.run_analysis()


if __name__ == "__main__":
    testing_routine()
