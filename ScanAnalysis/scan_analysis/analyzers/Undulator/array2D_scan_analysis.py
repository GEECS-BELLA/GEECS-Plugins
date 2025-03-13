"""
Camera Image Analysis

General camera image analyzer.
Child to ScanAnalysis (./scan_analysis/base.py)
"""
# %% imports
from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional, List

if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag
    from numpy.typing import NDArray

from pathlib import Path
import logging
import numpy as np
import numbers
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio as io

from scan_analysis.base import ScanAnalysis

from image_analysis.analyzers.basic_image_analysis import BasicImageAnalyzer
from image_analysis.base import ImageAnalyzer

import traceback
PRINT_TRACEBACK = False

# %% classes
class Array2DScanAnalysis(ScanAnalysis):

    def __init__(self, scan_tag: ScanTag,
                 device_name: str,
                 image_analyzer: Optional[ImageAnalyzer] = None,
                 skip_plt_show: bool = True,
                 flag_logging: bool = True,
                 flag_save_images: bool = True):
        """
        Initialize the CameraImageAnalysis class.

        Args:
            scan_tag (ScanTag): Path to the scan directory containing data.
            device_name (str): Name of the device to construct the subdirectory path.
            skip_plt_show (bool): Flag that sets if matplotlib is tried to use for plotting
            flag_logging (bool): Flag that sets if error and warning messages are displayed
            flag_save_images (bool): Flag that sets if images are saved to disk
        """
        if not device_name:
            raise ValueError("CameraImageAnalysis requires a device_name.")

        super().__init__(scan_tag, device_name=device_name,
                         skip_plt_show=skip_plt_show)

        self.image_analyzer = image_analyzer or BasicImageAnalyzer()
        # define flags
        self.flag_logging = flag_logging
        self.flag_save_images = flag_save_images

        # organize various paths
        self.path_dict = {'data_img': Path(self.scan_directory) / f"{device_name}",
                          'save': (self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name
                                   / f"{device_name}" / "Array2DScanAnalysis")
                          }

        self.data = {'shot_num': [], 'images': []}

        # Check if data directory exists and is not empty
        if not self.path_dict['data_img'].exists() or not any(self.path_dict['data_img'].iterdir()):
            if self.flag_logging:
                logging.warning(f"Data directory '{self.path_dict['data_img']}' does not exist or is empty. Skipping")

    def run_analysis(self, config_options: Optional[str] = None):
        # initialize analysis
        if self.path_dict['data_img'] is None or self.auxiliary_data is None:
            if self.flag_logging:
                logging.info("Skipping analysis due to missing data or auxiliary file.")
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


            self.auxiliary_data.to_csv(self.auxiliary_file_path, sep='\t', index=False)

            return self.display_contents

        except Exception as e:
            if PRINT_TRACEBACK:
                print(traceback.format_exc())
            if self.flag_logging:
                logging.warning(f"Warning: Image analysis failed due to: {e}")
            return

    def save_fig(self, save_path: Path,
                 bbox_inches: str = 'tight', pad_inches: float = 0.) -> None:

        # ensure save directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # save image
        plt.savefig(save_path, bbox_inches=bbox_inches, pad_inches=pad_inches)

    def save_geecs_scaled_image(self, image: NDArray, save_dir: Union[str, Path],
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

    def save_normalized_image(self, image: np.ndarray, save_dir: Union[str, Path], save_name: str, label: Optional[str] = None):
        """
        Display and optionally save a 16-bit image with specified min/max values for visualization.
        Optionally, add a label to the image before saving.

        Args:
            image (np.ndarray): The image to save.
            save_dir (Union[str, Path]): The directory where the image will be saved.
            save_name (str): The name of the saved image file.
            label (Optional[str]): A label to add to the image. Defaults to None.
        """
        max_val = np.max(image)

        # Display the image
        plt.clf()

        plt.imshow(image, cmap='plasma', vmin=0, vmax=max_val)
        plt.colorbar()  # Adds a color scale bar
        plt.axis('off')  # Hide axis for a cleaner look
        
        # Add a label if provided
        if label:
            plt.title(label, fontsize=12, pad=10)  # Add the label at the top of the plot


        # Save the image if a path is provided
        if save_dir and save_name:
            save_path = Path(save_dir) / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            if self.flag_logging:
                logging.info(f"Image saved at {save_path}")

    def create_image_array(self, binned_data: dict[dict], ref_coords: Optional[tuple] = None,
                           plot_scale: Optional[float] = None, save_path: Optional[Path] = None):
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
        if save_path:
            filename = save_path.name
        else:
            filename = f'{self.device_name}_averaged_image_grid.png'
            save_path = Path(self.path_dict['save']) / filename
        plt.savefig(save_path, bbox_inches='tight')
        if self.flag_logging:
            logging.info(f"Saved final image grid as {filename}.")
        self.close_or_show_plot()

    def get_image_analysis_result(self, shot_num: int, file_path: Optional[Path]) -> Optional[np.ndarray]:
        if file_path is None:
            logging.warning(f"No file path provided for shot {shot_num}.")
            return None

        try:
            # Use the injected image analyzer
            logging.info(f'attempting to process {file_path}')
            results_dict = self.image_analyzer.analyze_image(file_path=file_path)
            # Expecting results_dict to have keys: 'processed_image' and 'analysis_results'
            image = results_dict.get("processed_image")
            if image is not None:
                image = image.astype(np.uint16)
                self.data['shot_num'].append(shot_num)
                self.data['images'].append(image)

                # Update auxiliary data if analysis results are provided
                analysis_dict = results_dict.get("analysis_results", {})
                if analysis_dict:
                    for key, value in analysis_dict.items():
                        if not isinstance(value, numbers.Number):
                            logging.warning(
                                f"[{self.__class__.__name__} using {self.image_analyzer.__class__.__name__}] "
                                f"analysis result for shot {shot_num} key '{key}' is not a number (got {type(value).__name__}). Skipping."
                            )
                        else:
                            self.auxiliary_data.loc[self.auxiliary_data['Shotnumber'] == shot_num, key] = value

            return image

        except FileNotFoundError:
            logging.info(f"Image file not found for shot {shot_num}: {file_path}")
            return None
        except Exception as e:
            logging.error(f"Error processing shot {shot_num}: {e}")
            return None

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

    def load_images_for_bin(self, bin_number: int) -> list[NDArray]:
        images = []
        shots_in_bin = self.auxiliary_data[self.auxiliary_data['Bin #'] == bin_number]['Shotnumber'].values

        for shot_num in shots_in_bin:
            image_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}.png'), None)
            image = self.get_image_analysis_result(shot_num, file_path=image_file)
            if image is not None:
                images.append(image)
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

    @staticmethod
    def create_gif(image_arrays: List[np.ndarray], output_file: str,
                   titles: Optional[List[str]] = None, duration: float = 100, dpi: int = 72):
        """
        Create a GIF from a list of images with titles, scaled to a fixed width,
        and using the 'plasma' colormap.

        Args:
            image_arrays (List[np.ndarray]): List of images to include in the GIF.
            output_file (str): Path to save the resulting GIF.
            titles (Optional[List[str]]): List of titles for each image.
            duration (float): Duration for each frame in the GIF in milliseconds.
            dpi (int): The DPI for the images (default is 72 DPI).
        """
        # Desired width in pixels based on DPI
        target_width_inches = 5  # Width in inches
        target_width_pixels = int(target_width_inches * dpi)

        # Create default titles if not provided
        if titles is None:
            titles = [f"Shot {num + 1}" for num in range(len(image_arrays))]


        # Initialize the colormap and normalization
        cmap = plt.get_cmap('plasma')
        norm = Normalize(vmin=np.min(image_arrays), vmax=np.mean([img.max() for img in image_arrays]))

        # Font parameters for adding titles
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)

        images = []
        for img, title in zip(image_arrays, titles):
            # Normalize the image and apply the colormap
            normalized_img = norm(img)
            colored_img = (cmap(normalized_img)[:, :, :3] * 255).astype(np.uint8)

            # Resize the image while maintaining the aspect ratio
            height, width, _ = colored_img.shape
            scale_factor = target_width_pixels / width
            target_height_pixels = int(height * scale_factor)
            resized_image = cv2.resize(colored_img, (target_width_pixels, target_height_pixels), interpolation=cv2.INTER_AREA)

            # Add title text
            (text_width, text_height), _ = cv2.getTextSize(title, font, font_scale, thickness)
            title_position = (max((target_width_pixels - text_width) // 2, 0), max(25, text_height + 10))

            # Add space for the title
            title_bar_height = 30
            title_image = np.zeros((title_bar_height + resized_image.shape[0], target_width_pixels, 3), dtype=np.uint8)
            title_image[title_bar_height:, :, :] = resized_image

            cv2.putText(title_image, title, title_position, font, font_scale, color, thickness)

            images.append(title_image)

        # Create GIF
        io.mimsave(output_file, images, duration=duration, loop=0)

    def run_noscan_analysis(self):
        """
        Image analysis in the case of a no scan.

        """
        # load images
        for shot_num in self.auxiliary_data['Shotnumber'].values:
            image_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}.png'), None)
            self.get_image_analysis_result(shot_num=shot_num, file_path = image_file)

        # get average image
        avg_image = self.average_images(self.data['images'])
        if self.flag_save_images:
            self.save_geecs_scaled_image(avg_image, save_dir=self.path_dict['save'],
                                         save_name=f'{self.device_name}_average_processed.png')
                                         
            save_name = f'{self.device_name}_average_processed_visual.png'
            self.save_normalized_image(avg_image, save_dir=self.path_dict['save'],
                                       save_name = save_name, label = save_name)
            display_content_path = Path(self.path_dict['save']) / save_name

            self.display_contents.append(str(display_content_path))

        # make gif
        if self.flag_save_images:
            filepath = self.path_dict['save'] / 'noscan.gif'
            self.create_gif(self.data['images'], filepath,
                            titles=[f"Shot {num}" for num in self.data['shot_num']])
                            
            self.display_contents.append(str(filepath))

    def run_scan_analysis(self):
        """
        Image analysis in the case of a scanned variable.

        """
        # bin data
        binned_data = self.bin_images(flag_save=self.flag_save_images)

        for bin_key, bin_item in binned_data.items():

            # overwrite stored image
            processed_image = binned_data[bin_key]['image']

            # save figures
            if self.flag_save_images:
                self.save_geecs_scaled_image(processed_image, save_dir=self.path_dict['save'],
                                             save_name=f'{self.device_name}_{bin_key}_processed.png')
                self.save_normalized_image(processed_image, save_dir=self.path_dict['save'],
                                           save_name=f'{self.device_name}_{bin_key}_processed_visual.png')

        # Once all bins are processed, create an array of the averaged images
        if len(binned_data) > 1 and self.flag_save_images:
            plot_scale = (getattr(self, 'camera_analysis_settings', {}) or {}).get('Plot Scale', None)

            # generate image array
            save_path = Path(self.path_dict['save']) /  f'{self.device_name}_averaged_image_grid.png'
            self.create_image_array(binned_data, plot_scale=plot_scale, save_path=save_path)

            # append save_path to display content list
            self.display_contents.append(str(save_path))

        # save binned data to class variable
        self.binned_data = binned_data

if __name__ == "__main__":
    from scan_analysis.base import AnalyzerInfo as Info
    from scan_analysis.execute_scan_analysis import analyze_scan
    from geecs_python_api.controls.api_defs import ScanTag
    from image_analysis.analyzers.ACaveMagCam3 import ACaveMagCam3ImageAnalyzer

    perform_analysis = True
    analyzer_info = Info(analyzer_class=Array2DScanAnalysis,
                         requirements={'UC_ACaveMagCam3'},
                         device_name='UC_ACaveMagCam3',
                         image_analyzer_class=ACaveMagCam3ImageAnalyzer)
    test_tag = ScanTag(year=2025, month=3, day=6, number=40, experiment='Undulator')
    test_analyzer = analyzer_info
    analyze_scan(test_tag, [analyzer_info], debug_mode=not perform_analysis)
