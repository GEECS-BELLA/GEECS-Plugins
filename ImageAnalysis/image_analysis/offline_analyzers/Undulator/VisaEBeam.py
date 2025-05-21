from __future__ import annotations

from typing import Union, Optional
from pathlib import Path

import numpy as np
from scipy.ndimage import label, gaussian_filter
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import Normalize
from matplotlib.patches import Ellipse

from image_analysis.base import ImageAnalyzer
import logging

visa_ebeam_configs = {
    "UC_VisaEBeam1": {
        "Background Level": 15,
        "Left ROI": 558,
        "Top ROI": 290,
        "Size_X": 333,
        "Size_Y": 333,
        "Median Filter Size": 5,
        "Median Filter Cycles": 1,
        "Gaussian Filter Size": 3,
        "Gaussian Filter Cycles": 0,
        "Threshold": 0,
        "Rotate": 0,
        "Calibration": 0.0000075,
        "Apply Mask": True,
        "Cross1": [722, 285],
        "Cross2": [728, 636],
        "Cross Shift": -4,
        "Blue Centroid X": 764,
        "Blue Centroid Y": 449
    },
    "UC_VisaEBeam2": {
        "Background Level": 20,
        "Left ROI": 110,
        "Top ROI": 75,
        "Size_X": 333,
        "Size_Y": 333,
        "Median Filter Size": 5,
        "Median Filter Cycles": 1,
        "Gaussian Filter Size": 3,
        "Gaussian Filter Cycles": 0,
        "Threshold": 1000,
        "Rotate": 0,
        "Calibration": 0.0000075,
        "Apply Mask": True,
        "Cross1": [193, 97],
        "Cross2": [193, 465],
        "Cross Shift": 0,
        "Blue Centroid X": 216,
        "Blue Centroid Y": 252
    },
    "UC_VisaEBeam3": {
        "Background Level": 15,
        "Left ROI": 160,
        "Top ROI": 140,
        "Size_X": 333,
        "Size_Y": 333,
        "Median Filter Size": 5,
        "Median Filter Cycles": 0,
        "Gaussian Filter Size": 3,
        "Gaussian Filter Cycles": 0,
        "Threshold": 0,
        "Rotate": -3.5,
        "Calibration": 0.0000075,
        "Apply Mask": True,
        "Cross1": [374, 127],
        "Cross2": [338, 485],
        "Cross Shift": 0,
        "Blue Centroid X": 315,
        "Blue Centroid Y": 264
    },
    "UC_VisaEBeam4": {
        "Background Level": 15,
        "Left ROI": 180,
        "Top ROI": 200,
        "Size_X": 333,
        "Size_Y": 333,
        "Median Filter Size": 5,
        "Median Filter Cycles": 1,
        "Gaussian Filter Size": 3,
        "Gaussian Filter Cycles": 0,
        "Threshold": 0,
        "Rotate": 2.5,
        "Calibration": 0.0000075,
        "Apply Mask": True,
        "Cross1": [268, 175],
        "Cross2": [287, 533],
        "Cross Shift": -4,
        "Blue Centroid X": 318,
        "Blue Centroid Y": 336
    },
    "UC_VisaEBeam5": {
        "Background Level": 25,
        "Left ROI": 135,
        "Top ROI": 90,
        "Size_X": 333,
        "Size_Y": 333,
        "Median Filter Size": 0,
        "Median Filter Cycles": 0,
        "Gaussian Filter Size": 0,
        "Gaussian Filter Cycles": 0,
        "Threshold": 0,
        "Rotate": 0,
        "Calibration": 0.0000075,
        "Apply Mask": True,
        "Cross1": [267, 95],
        "Cross2": [262, 440],
        "Cross Shift": 0,
        "Blue Centroid X": 270,
        "Blue Centroid Y": 232
    },
    "UC_VisaEBeam6": {
        "Background Level": 25,
        "Left ROI": 195,
        "Top ROI": 85,
        "Size_X": 333,
        "Size_Y": 333,
        "Median Filter Size": 5,
        "Median Filter Cycles": 1,
        "Gaussian Filter Size": 3,
        "Gaussian Filter Cycles": 0,
        "Threshold": 0,
        "Rotate": 0,
        "Calibration": 0.0000075,
        "Apply Mask": True,
        "Cross1": [295, 85],
        "Cross2": [290, 425],
        "Cross Shift": 0,
        "Blue Centroid X": 289,
        "Blue Centroid Y": 204
    },
    "UC_VisaEBeam7": {
        "Background Level": 15,
        "Left ROI": 120,
        "Top ROI": 100,
        "Size_X": 333,
        "Size_Y": 333,
        "Median Filter Size": 5,
        "Median Filter Cycles": 1,
        "Gaussian Filter Size": 3,
        "Gaussian Filter Cycles": 0,
        "Threshold": 0,
        "Rotate": -8,
        "Calibration": 0.0000075,
        "Apply Mask": True,
        "Cross1": [275, 328],
        "Cross2": [275, 550],
        "Cross Shift": 0,
        "Blue Centroid X": 201,
        "Blue Centroid Y": 289
    },
    "UC_VisaEBeam8": {
        "Background Level": 30,
        "Left ROI": 105,
        "Top ROI": 100,
        "Size_X": 333,
        "Size_Y": 333,
        "Median Filter Size": 5,
        "Median Filter Cycles": 1,
        "Gaussian Filter Size": 3,
        "Gaussian Filter Cycles": 0,
        "Threshold": 0
    }
}


class VisaEBeam(ImageAnalyzer):
    def __init__(
        self,
        camera_name: str = 'UC_VisaEBeam1'
    ):
        """
        Initialize VisaEBeam analyzer for a specific camera.

        Parameters
        ----------
        camera_name : str
            The key for the desired camera config (e.g. 'UC_VisaEBeam3')
        config : Optional[dict]
            Optional dictionary to override values in the default config
        kwargs : dict
            Passed to the ImageAnalyzer base class (e.g., background_obj)
        """
        if camera_name not in visa_ebeam_configs:
            raise ValueError(f"Unknown camera name: {camera_name}")

        self.camera_name = camera_name
        self.config = visa_ebeam_configs[camera_name].copy()
        self.config["device_name"] = camera_name

        # Store raw ROI values for reference
        self.left = self.config["Left ROI"]
        self.top = self.config["Top ROI"]
        self.width = self.config["Size_X"]
        self.height = self.config["Size_Y"]

        # Convenience: derived ROI bounds as a tuple (top, bottom, left, right)
        self.roi = (self.top, self.top + self.height, self.left, self.left + self.width)

        self.preprocessed = False

        self.run_analyze_image_asynchronously = True
        self.flag_logging = True
        self.min_val = 0

        self.use_interactive = False

        super().__init__(config=self.config)

    def apply_roi(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the configured ROI to an image or a stack of images.

        Parameters
        ----------
        data : np.ndarray
            Either a 2D (H, W) image or a 3D (N, H, W) image stack.

        Returns
        -------
        np.ndarray
            ROI-cropped image or image stack.
        """
        top, bottom, left, right = self.roi

        if data.ndim == 3:
            return data[:, top:bottom, left:right]
        elif data.ndim == 2:
            return data[top:bottom, left:right]
        else:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    def analyze_image_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Subtracts a dynamically determined background and returns the processed images.

        Args:
            images (list[np.ndarray]): A list of input images.

        Returns:
            list[np.ndarray]: A list of processed images.
        """

        # Stack images for batch-level processing
        stack = np.stack(images, axis=0)
        stack = self.image_preprocess(stack)
        self.preprocessed = True

        # Step 1: Learn background from percentile projection
        self.background_obj.set_percentile_background_from_stack(stack=stack, percentile=2.5)
        stack = self.background_obj.subtract(data=stack)

        # Step 2: Subtract per-image medians
        stack = self.background_obj.subtract_imagewise_median(data=stack)

        # # Step 3: Generate and apply apodization
        # self.background_obj.generate_apodization_mask(stack=stack, percentile=91, sigma=5)
        # stack = self.background_obj.apply_apodization(data=stack)

        # Optionally record min value for later use
        self.min_val = np.min(stack)
        logging.info(f'min value from the stack: {self.min_val}')

        return list(stack)  # maintain list format for downstream compatibility

    @staticmethod
    def create_cross_mask(image: np.ndarray,
                          cross_center: tuple[int, int],
                          angle: Union[int, float],
                          cross_height: int = 54,
                          cross_width: int = 54,
                          thickness: int = 10) -> np.ndarray:
        """
        Creates a mask with a cross centered at `cross_center` with the cross being zeros and the rest ones.

        Args:
        image: The image on which to base the mask size.
        cross_center: The (x, y) center coordinates of the cross.
        angle: Rotation angle of the cross in degrees.
        cross_height: The height of the cross extending vertically from the center.
        cross_width: The width of the cross extending horizontally from the center.
        thickness: The thickness of the lines of the cross.

        Returns:
        - np.array: The mask with the cross.
        """
        mask = np.ones_like(image, dtype=np.uint16)
        x_center, y_center = cross_center
        vertical_start, vertical_end = max(y_center - cross_height, 0), min(y_center + cross_height, image.shape[0])
        horizontal_start, horizontal_end = max(x_center - cross_width, 0), min(x_center + cross_width, image.shape[1])
        mask[vertical_start:vertical_end, x_center - thickness // 2:x_center + thickness // 2] = 0
        mask[y_center - thickness // 2:y_center + thickness // 2, horizontal_start:horizontal_end] = 0

        m = cv2.getRotationMatrix2D(cross_center, angle, 1.0)
        rotated_mask = cv2.warpAffine(mask, m, (image.shape[1], image.shape[0]),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=1)
        return rotated_mask

    def image_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to an image or stack of images.

        This includes:
          - Applying crosshair masks defined in the config
          - Cropping the image to the configured region of interest (ROI)

        Parameters
        ----------
        image : np.ndarray
            A 2D array representing a single image (H, W),
            or a 3D array representing a stack of images (N, H, W)

        Returns
        -------
        np.ndarray
            The preprocessed image or image stack with masks and ROI applied.
        """

        image = self.apply_cross_mask(image)
        image = self.apply_roi(image)
        return image

    def apply_cross_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply cross masks to either a single image or a stack of images.

        Parameters
        ----------
        image : np.ndarray
            A single 2D image (H, W) or a stack of images (N, H, W)

        Returns
        -------
        np.ndarray
            Masked image or image stack
        """

        def generate_combined_mask(img: np.ndarray) -> np.ndarray:
            mask1 = self.create_cross_mask(img, tuple(self.config['Cross1']), self.config['Rotate'])
            mask2 = self.create_cross_mask(img, tuple(self.config['Cross2']), self.config['Rotate'])
            return mask1 * mask2

        if image.ndim == 3:
            mask = generate_combined_mask(image[0])  # Assume all images same shape
            return image * mask[None, :, :]  # Broadcast over all images
        elif image.ndim == 2:
            mask = generate_combined_mask(image)
            return image * mask
        else:
            raise ValueError(f"Expected 2D or 3D image array, got shape {image.shape}")

    def analyze_image(self, image: np.ndarray, auxiliary_data: Optional[dict] = None) -> dict[
        str, Union[float, int, str, np.ndarray]]:

        """
        Analyze an image from acave mag cam3.

        Parameters
        ----------
        image : np.array,
            the image.
        auxiliary_data: dict, containing any additional information needed for analysis

        Returns
        -------
        dict
            A dictionary with the processed image and placeholder for analysis results.
        """

        if not self.preprocessed:
            image = self.image_preprocess(image)

        analyzed_image = (image-self.min_val).astype(np.uint16)
        # references = {'Blue Centroid X':self.config["Blue Centroid X"]-self.left,'Blue Centroid Y':self.config["Blue Centroid Y"]-self.top}

        return_dictionary = self.build_return_dictionary(return_image = analyzed_image,
                                                         input_parameters = self.config)
        if self.use_interactive:
            self.render_image(image = analyzed_image, input_dict = self.config)

        return return_dictionary

    @staticmethod
    def render_image(
            image: np.ndarray,
            analysis_results_dict: Optional[dict[str, Union[float, int]]] = None,
            input_dict: Optional[dict[str, Union[float, int]]] = None,
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            cmap: str = 'plasma',
            ax: Optional[plt.Axes] = None
    ) -> None:
        if ax is None:
            use_color_bar = True
            fig, ax = plt.subplots()
        else:
            use_color_bar = False

        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)

        references = {'Blue Centroid X':input_dict["Blue Centroid X"]-input_dict['Left ROI'],
                      'Blue Centroid Y':input_dict["Blue Centroid Y"]-input_dict['Top ROI']}

        # Example below of how to use parameters generated from the analysis to add overlays
        use_overlay = True
        if use_overlay:
            cent_x = references['Blue Centroid X']
            cent_y = references['Blue Centroid Y']
            ax.plot(cent_x, cent_y, 'bo', markersize=5)

        ax.set_xlabel('X Pixels')
        ax.set_ylabel('Y Pixels')

        if use_color_bar:  # Only add colorbar if we're in standalone mode
            plt.colorbar(im, ax=ax)
            plt.show()


if __name__ == "__main__":
    image_analyzer  = VisaEBeam()
    image_analyzer.use_interactive = True
    # Resolve path relative to this file
    script_dir = Path(__file__).resolve()
    for _ in range(4):
        script_dir = script_dir.parent
    file_path = script_dir / "tests" / "data" / "VisaEBeam_test_data" / "UC_VisaEBeam1_001.png"
    results = image_analyzer.analyze_image_file(image_filepath=file_path)
