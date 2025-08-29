"""
Camera Image Analysis class.

General camera image analyzer.
Child to ScanAnalyzer (./scan_analysis/base.py)
"""

# %% imports
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional, List

if TYPE_CHECKING:
    from geecs_data_utils import ScanTag

from pathlib import Path
import logging
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter, gaussian_filter
import imageio as io

from scan_analysis.base import ScanAnalyzer
from image_analysis.utils import read_imaq_png_image

import traceback

PRINT_TRACEBACK = True


# %% classes
class CameraImageAnalyzer(ScanAnalyzer):
    """
    Analyzer for camera image data within GEECS experiments.

    This class provides methods for loading, processing, analyzing, and saving
    images from camera diagnostics. It supports both no-scan and scan-based
    analyses, including filtering, cropping, binning, averaging, and visualization.
    Images can be saved in both raw and normalized formats, and GIFs or image
    arrays can be generated for visualization.
    """

    def __init__(
        self,
        device_name: str,
        skip_plt_show: bool = True,
        flag_logging: bool = True,
        flag_save_images: bool = True,
    ):
        """
        Initialize a CameraImageAnalyzer.

        Parameters
        ----------
        device_name : str
            Name of the device to construct the subdirectory path.
        skip_plt_show : bool, default=True
            If True, suppresses interactive plotting with matplotlib.
        flag_logging : bool, default=True
            If True, enable logging of warnings and errors.
        flag_save_images : bool, default=True
            If True, save images to disk during analysis.

        Raises
        ------
        ValueError
            If `device_name` is not provided.
        """
        if not device_name:
            raise ValueError("CameraImageAnalyzer requires a device_name.")

        super().__init__(device_name=device_name, skip_plt_show=skip_plt_show)

        # define flags
        self.flag_logging = flag_logging
        self.flag_save_images = flag_save_images

    def _establish_additional_paths(self):
        # organize various paths
        config_folder = Path(__file__).parents[2] / "config" / self.experiment_dir
        self.path_dict = {
            "data_img": Path(self.scan_directory) / f"{self.device_name}",
            "save": (
                self.scan_directory.parents[1]
                / "analysis"
                / self.scan_directory.name
                / f"{self.device_name}"
                / "CameraImageAnalyzer"
            ),
            "cam_configs": config_folder / "camera_analysis_settings.yaml",
        }

        # set up camera configs and settings
        self.camera_analysis_configs = None
        self.camera_analysis_settings = None
        if self.path_dict["cam_configs"].exists():
            self.camera_analysis_configs = self.load_camera_analysis_config()
            if self.device_name in self.camera_analysis_configs:
                self.camera_analysis_settings = self.camera_analysis_configs[
                    self.device_name
                ]

        # Check if data directory exists and is not empty
        if not self.path_dict["data_img"].exists() or not any(
            self.path_dict["data_img"].iterdir()
        ):
            if self.flag_logging:
                logging.warning(
                    f"Data directory '{self.path_dict['data_img']}' does not exist or is empty. Skipping"
                )

    def _run_analysis_core(self):
        self._establish_additional_paths()

        # initialize analysis
        if self.path_dict["data_img"] is None or self.auxiliary_data is None:
            if self.flag_logging:
                logging.info("Skipping analysis due to missing data or auxiliary file.")
            return

        # if saving, make sure save location exists
        if self.flag_save_images and not self.path_dict["save"].exists():
            self.path_dict["save"].mkdir(parents=True)

        # delegate analysis type
        try:
            if self.noscan:
                self.run_noscan_analysis()
            else:
                self.run_scan_analysis()

            return self.display_contents

        except Exception as e:
            if PRINT_TRACEBACK:
                print(traceback.format_exc())
            if self.flag_logging:
                logging.warning(f"Warning: Image analysis failed due to: {e}")
            return

    def load_camera_analysis_config(self) -> dict:
        """
        Load the master camera analysis configuration.

        Reads a YAML configuration file containing device-specific
        camera analysis settings.

        Returns
        -------
        dict
            Dictionary of analysis configurations loaded from the YAML file.
        """
        camera_analysis_configs_file = str(
            self.path_dict["cam_configs"]
        )  # convert Path object to string
        with open(camera_analysis_configs_file, "r") as file:
            camera_analysis_configs = yaml.safe_load(file)

        return camera_analysis_configs

    def save_fig(
        self, save_path: Path, bbox_inches: str = "tight", pad_inches: float = 0.0
    ) -> None:
        """
        Save the current matplotlib figure to a file.

        Parameters
        ----------
        save_path : Path
            Destination file path for the saved figure.
        bbox_inches : str, default="tight"
            Bounding box option for saving the figure.
        pad_inches : float, default=0.0
            Padding in inches around the figure.
        """
        # ensure save directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # save image
        plt.savefig(save_path, bbox_inches=bbox_inches, pad_inches=pad_inches)

    def save_geecs_scaled_image(
        self,
        image: np.ndarray,
        save_dir: Union[str, Path],
        save_name: str,
        bit_depth: int = 16,
    ):
        """
        Save an image to disk in GEECS-compatible format.

        Images are typically stored as 16-bit. Hardware produces 12-bit,
        so the last 4 bits are unused. Optionally, images can be scaled
        to 8-bit for visualization.

        Parameters
        ----------
        image : np.ndarray
            The image array to save.
        save_dir : str or Path
            Directory where the image will be saved.
        save_name : str
            Filename for the saved image.
        bit_depth : {8, 16}, default=16
            Bit depth for saving. Only 8-bit and 16-bit are supported.

        Raises
        ------
        ValueError
            If `bit_depth` is not 8 or 16, or if `image` dtype is unsupported.
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
            image = (image * 2**4).astype(np.uint8)
        else:
            raise ValueError("Unsupported bit depth. Only 8 or 16 bits are supported.")

        # Save the image using cv2
        cv2.imwrite(str(save_path), image)
        if self.flag_logging:
            logging.info(f"Image saved at {save_path}")

    def save_normalized_image(
        self,
        image: np.ndarray,
        save_dir: Union[str, Path],
        save_name: str,
        label: Optional[str] = None,
    ):
        """
        Save a normalized image for visualization.

        Images are normalized to their maximum pixel value, displayed
        using a plasma colormap, and optionally annotated with a label.

        Parameters
        ----------
        image : np.ndarray
            The image array to save.
        save_dir : str or Path
            Directory where the image will be saved.
        save_name : str
            Filename for the saved image.
        label : str, optional
            Label to display as a title on the image.
        """
        max_val = np.max(image)

        # Display the image
        plt.clf()

        plt.imshow(image, cmap="plasma", vmin=0, vmax=max_val)
        plt.colorbar()  # Adds a color scale bar
        plt.axis("off")  # Hide axis for a cleaner look

        # Add a label if provided
        if label:
            plt.title(
                label, fontsize=12, pad=10
            )  # Add the label at the top of the plot

        # Save the image if a path is provided
        if save_dir and save_name:
            save_path = Path(save_dir) / save_name
            save_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure the directory exists
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            if self.flag_logging:
                logging.info(f"Image saved at {save_path}")

    def create_image_array(
        self,
        binned_data: dict[dict],
        ref_coords: Optional[tuple] = None,
        plot_scale: Optional[float] = None,
        save_path: Optional[Path] = None,
    ):
        """
        Create and save a grid of averaged images.

        Images from parameter bins are arranged in a grid layout with
        consistent scaling. Each subplot is labeled with the scan parameter
        value, and optional reference coordinates may be overlaid.

        Parameters
        ----------
        binned_data : dict of dict
            Dictionary mapping bin numbers to averaged images and parameter values.
        ref_coords : tuple, optional
            Reference x and y coordinates to overlay on each image.
        plot_scale : float, optional
            Maximum color scale value. Defaults to the global maximum pixel value.
        save_path : Path, optional
            Path to save the resulting grid image. If not provided,
            a default filename is used.
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
        all_pixels = np.concatenate(
            [
                binned_data[bnum]["image"].ravel()
                for bnum in list(binned_data.keys())
                if binned_data[bnum]["image"] is not None
            ]
        )
        low, high = 0, all_pixels.max()
        if plot_scale is not None:
            high = plot_scale

        # Create a figure with the appropriate number of subplots
        fig, axs = plt.subplots(
            grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3)
        )

        # Flatten axes array for easy indexing (if there's only one row/col, axs won't be a 2D array)
        axs = axs.flatten()

        bin_ind = None
        for bin_num, bin_item in binned_data.items():
            bin_ind = bin_num - 1
            img = bin_item["image"]
            param_value = bin_item["value"]

            # display with adjusted scale
            axs[bin_ind].imshow(img, cmap="plasma", vmin=low, vmax=high)
            axs[bin_ind].set_title(
                f"{self.scan_parameter}: {param_value:.2f}", fontsize=10
            )  # Use scan parameter for label
            axs[bin_ind].axis("off")  # Turn off axes for cleaner display

            if ref_coords is not None:
                axs[bin_ind].plot(ref_coords[0], ref_coords[1], color="g", marker="o")

        for j in range(bin_ind + 1, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()

        # Save the final image grid for visualization
        if save_path:
            filename = save_path.name
        else:
            filename = f"{self.device_name}_averaged_image_grid.png"
            save_path = Path(self.path_dict["save"]) / filename
        plt.savefig(save_path, bbox_inches="tight")
        if self.flag_logging:
            logging.info(f"Saved final image grid as {filename}.")
        self.close_or_show_plot()

    def load_images_for_bin(self, bin_number: int) -> list[np.ndarray]:
        """
        Load all images corresponding to a specific bin.

        Parameters
        ----------
        bin_number : int
            The bin number for which to load images.

        Returns
        -------
        list of np.ndarray
            List of images belonging to the specified bin.
        """
        images = []
        shots_in_bin = self.auxiliary_data[self.auxiliary_data["Bin #"] == bin_number][
            "Shotnumber"
        ].values
        # expected_image_size = None

        for shot_num in shots_in_bin:
            image_file = next(
                self.path_dict["data_img"].glob(f"*_{shot_num:03d}.png"), None
            )
            if image_file:
                image = read_imaq_png_image(image_file)
                images.append(image)
                # if expected_image_size is None:
                # expected_image_size = image.shape  # Determine image size from the first valid image
            else:
                if self.flag_logging:
                    logging.warning(
                        f"Missing data for shot {shot_num}, adding zero array."
                    )
                # kj comment: not sure the below makes sense.
                # Will keep code for now but comment out.
                # if expected_image_size:
                #     images.append(np.zeros(expected_image_size, dtype=np.uint16))

        return images

    @staticmethod
    def average_images(images: list[np.ndarray]) -> Optional[np.ndarray]:
        """
        Compute the average of a list of images.

        Parameters
        ----------
        images : list of np.ndarray
            List of image arrays to average.

        Returns
        -------
        np.ndarray or None
            Averaged image in 16-bit format, or None if input is empty.
        """
        if len(images) == 0:
            return None

        return np.mean(images, axis=0).astype(
            np.uint16
        )  # Keep 16-bit format for the averaged image

    def bin_images(self, flag_save=None):
        """
        Bin images according to scan parameter bins.

        Loads, averages, and optionally saves images for each unique bin.

        Parameters
        ----------
        flag_save : bool, optional
            If True, save averaged images. Defaults to class attribute `flag_save_images`.

        Returns
        -------
        dict
            Dictionary mapping bin numbers to averaged images and parameter values.
        """
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
            images = self.load_images_for_bin(
                bin_val
            )  # TODO couldn't figure out the type hinting warning here...
            if len(images) == 0:
                if self.flag_logging:
                    logging.warning(f"No images found for bin {bin_val}.")
                continue

            # average the images
            # need to confirm this is foolproof, are binned_param_values completely correlated?
            avg_image = self.average_images(images)
            binned_data[bin_val] = {
                "value": self.binned_param_values[bin_ind],
                "image": avg_image,
            }

            if flag_save:
                save_name = f"{self.device_name}_{bin_val}.png"
                self.save_geecs_scaled_image(
                    avg_image, save_dir=self.path_dict["save"], save_name=save_name
                )

                if self.flag_logging:
                    logging.info(
                        f"Averaged images for bin {bin_val} and saved as {save_name}."
                    )

            else:
                if self.flag_logging:
                    logging.info(f"Averaged images for bin {bin_val} but not saved.")

        # check for empty bins and remove
        binned_data = {key: value for key, value in binned_data.items() if value}

        return binned_data  # TODO figure out type hinting for this.  Tried dict[dict] but Pycharm wasn't happy

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """
        Crop an image using predefined ROI settings.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Cropped image.
        """
        settings = self.camera_analysis_settings
        if settings is None:
            return image

        else:
            cropped_image = image[
                settings["Top ROI"] : settings["Top ROI"] + settings["Size_Y"],
                settings["Left ROI"] : settings["Left ROI"] + settings["Size_X"],
            ]

            return cropped_image

    def filter_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply filtering operations to an image.

        Uses median and Gaussian filters based on predefined settings.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Filtered image with specified preprocessing applied.
        """
        settings = self.camera_analysis_settings

        processed_image = image.astype(np.float32)  # Ensure a consistent numeric type

        for _ in range(settings.get("Median Filter Cycles", 0)):
            processed_image = median_filter(
                processed_image, size=settings["Median Filter Size"]
            )

        for _ in range(settings.get("Gaussian Filter Cycles", 0)):
            processed_image = gaussian_filter(
                processed_image, sigma=settings["Gaussian Filter Size"]
            )

        # Optionally cast back to a 16-bit integer if needed
        return (
            processed_image.astype(np.uint16)
            if image.dtype == np.uint16
            else processed_image
        )

    def image_processing(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform cropping and filtering on an image.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        tuple of np.ndarray
            Cropped image and processed image.
        """
        # crop image, save image
        cropped_image = self.crop_image(image)
        processed_image = cropped_image.copy()

        # filter image, save image
        if self.camera_analysis_settings and self.camera_analysis_settings.get(
            "Median Filter Size", None
        ):
            processed_image = self.filter_image(processed_image)

        return cropped_image, processed_image

    @staticmethod
    def create_gif(
        image_arrays: List[np.ndarray],
        output_file: str,
        titles: Optional[List[str]] = None,
        duration: float = 100,
        dpi: int = 72,
    ):
        """
        Create a GIF from a list of images.

        Parameters
        ----------
        image_arrays : list of np.ndarray
            Images to include in the GIF.
        output_file : str
            Path to save the resulting GIF.
        titles : list of str, optional
            Titles for each frame. Defaults to "Shot <n>".
        duration : float, default=100
            Duration of each frame in milliseconds.
        dpi : int, default=72
            Resolution in dots per inch for scaling images.
        """
        # Desired width in pixels based on DPI
        target_width_inches = 5  # Width in inches
        target_width_pixels = int(target_width_inches * dpi)

        # Create default titles if not provided
        if titles is None:
            titles = [f"Shot {num + 1}" for num in range(len(image_arrays))]

        # Initialize the colormap and normalization
        cmap = plt.get_cmap("plasma")
        norm = Normalize(
            vmin=np.min(image_arrays), vmax=np.mean([img.max() for img in image_arrays])
        )

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
            resized_image = cv2.resize(
                colored_img,
                (target_width_pixels, target_height_pixels),
                interpolation=cv2.INTER_AREA,
            )

            # Add title text
            (text_width, text_height), _ = cv2.getTextSize(
                title, font, font_scale, thickness
            )
            title_position = (
                max((target_width_pixels - text_width) // 2, 0),
                max(25, text_height + 10),
            )

            # Add space for the title
            title_bar_height = 30
            title_image = np.zeros(
                (title_bar_height + resized_image.shape[0], target_width_pixels, 3),
                dtype=np.uint8,
            )
            title_image[title_bar_height:, :, :] = resized_image

            cv2.putText(
                title_image, title, title_position, font, font_scale, color, thickness
            )

            images.append(title_image)

        # Create GIF
        io.mimsave(output_file, images, duration=duration, loop=0)

    def run_noscan_analysis(self):
        """
        Run analysis for the no-scan case.

        Loads all images, applies preprocessing, generates averages,
        saves outputs, and creates a GIF of all shots.
        """
        # load images
        data = {"shot_num": [], "images": []}
        for shot_num in self.auxiliary_data["Shotnumber"].values:
            image_file = next(
                self.path_dict["data_img"].glob(f"*_{shot_num:03d}.png"), None
            )
            if image_file:
                data["shot_num"].append(shot_num)
                data["images"].append(read_imaq_png_image(image_file))
            else:
                if self.flag_logging:
                    logging.warning(f"Missing data for shot {shot_num}.")

        # perform image analysis
        for i in range(len(data["images"])):
            _, data["images"][i] = self.image_processing(data["images"][i])

        # get average image
        avg_image = self.average_images(data["images"])
        if self.flag_save_images:
            self.save_geecs_scaled_image(
                avg_image,
                save_dir=self.path_dict["save"],
                save_name=f"{self.device_name}_average_processed.png",
            )

            save_name = f"{self.device_name}_average_processed_visual.png"
            self.save_normalized_image(
                avg_image,
                save_dir=self.path_dict["save"],
                save_name=save_name,
                label=save_name,
            )
            display_content_path = Path(self.path_dict["save"]) / save_name

            self.display_contents.append(str(display_content_path))

        # make gif
        if self.flag_save_images:
            filepath = self.path_dict["save"] / "noscan.gif"
            self.create_gif(
                data["images"],
                filepath,
                titles=[f"Shot {num}" for num in data["shot_num"]],
            )

            self.display_contents.append(str(filepath))

    def run_scan_analysis(self):
        """
        Run analysis for the scan case.

        Bins images by parameter values, applies preprocessing,
        saves cropped and processed images, and generates a grid
        of averaged images.
        """
        # bin data
        binned_data = self.bin_images(flag_save=self.flag_save_images)

        for bin_key, bin_item in binned_data.items():
            # perform basic image processing
            (cropped_image, processed_image) = self.image_processing(bin_item["image"])

            # overwrite stored image
            binned_data[bin_key]["image"] = processed_image

            # save figures
            if self.flag_save_images:
                self.save_geecs_scaled_image(
                    cropped_image,
                    save_dir=self.path_dict["save"],
                    save_name=f"{self.device_name}_{bin_key}_cropped.png",
                )
                self.save_geecs_scaled_image(
                    processed_image,
                    save_dir=self.path_dict["save"],
                    save_name=f"{self.device_name}_{bin_key}_processed.png",
                )
                self.save_normalized_image(
                    processed_image,
                    save_dir=self.path_dict["save"],
                    save_name=f"{self.device_name}_{bin_key}_processed_visual.png",
                )

        # Once all bins are processed, create an array of the averaged images
        if len(binned_data) > 1 and self.flag_save_images:
            plot_scale = (getattr(self, "camera_analysis_settings", {}) or {}).get(
                "Plot Scale", None
            )

            # generate image array
            save_path = (
                Path(self.path_dict["save"])
                / f"{self.device_name}_averaged_image_grid.png"
            )
            self.create_image_array(
                binned_data, plot_scale=plot_scale, save_path=save_path
            )

            # append save_path to display content list
            self.display_contents.append(str(save_path))

        # save binned data to class variable
        self.binned_data = binned_data


if __name__ == "__main__":
    from geecs_data_utils import ScanTag

    tag = ScanTag(year=2025, month=2, day=13, number=29, experiment="Undulator")
    analyzer = CameraImageAnalyzer(device_name="UC_ACaveMagCam3", skip_plt_show=True)
    analyzer.run_analysis(scan_tag=tag)
