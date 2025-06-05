from __future__ import annotations

from typing import Union, Optional
from pathlib import Path

import numpy as np
from scipy.ndimage import label, gaussian_filter
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import Normalize
from matplotlib.patches import Ellipse

from dataclasses import dataclass
from dataclasses import asdict

from image_analysis.base import ImageAnalyzer
from image_analysis.tools.rendering import base_render_image

import logging

VISA_EBEAM_PRESETS = {
    "U_BCaveMagSpec": {
        "bkg_level": 60,
        "left_ROI": 200,
        "top_ROI": 1,
        "roi_width": 500,
        "roi_height": 300,
        "rotate": 0,
        "spatial_calibration": 0.00002217
    },
}

@dataclass
class VisaEBeamConfig:
    camera_name: str = 'undefined'
    bkg_level: float = 0.0
    left_ROI: int = 1
    top_ROI: int = 1
    roi_width: int = 333
    roi_height: int = 333
    rotate: int = 0
    spatial_calibration: float = 0.0000075
    fiducial_cross1_location: Optional[tuple[int, int]] = None
    fiducial_cross2_location: Optional[tuple[int, int]] = None
    blue_cent_x: Optional[int] = None
    blue_cent_y: Optional[int] =None

    @staticmethod
    def from_camera_name(name: str) -> "VisaEBeamConfig":
        overrides = VISA_EBEAM_PRESETS.get(name, {})
        return VisaEBeamConfig(camera_name=name, **overrides)

class VisaEBeam(ImageAnalyzer):
    def __init__(self, background: Background = None, camera_name: str = None, preprocessed: bool = False,):
        """
        Initialize VisaEBeam analyzer for a specific camera.

        Parameters
        ----------
        camera_name : str
            The key for the desired camera config (e.g. 'UC_VisaEBeam3')
        config : Optional[dict]
            Optional dictionary to override values in the default config
        kwargs : dict
            Passed to the ImageAnalyzer base class (e.g., background)
        """
        if camera_name not in VISA_EBEAM_PRESETS:
            raise ValueError(f"Unknown camera name: {camera_name}")

        config = VisaEBeamConfig.from_camera_name(camera_name)
        self.camera_name = camera_name
        self.config = config
        self.kwargs_dict = asdict(self.config)

        # Store raw ROI values for reference
        self.left = self.config.left_ROI
        self.top = self.config.top_ROI
        self.width = self.config.roi_width
        self.height = self.config.roi_height

        # Convenience: derived ROI bounds as a tuple (top, bottom, left, right)
        self.roi = (self.top, self.top + self.height, self.left, self.left + self.width)

        self.preprocessed = preprocessed

        self.run_analyze_image_asynchronously = False
        self.flag_logging = True
        self.min_val = 0

        self.use_interactive = False

        super().__init__(background=background)


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

    def analyze_image_batch(self, images: list[np.ndarray]
                            ) -> Tuple[list[Array2D], dict[str, Union[int, float, bool, str]]]:
        """
        Subtracts a dynamically determined background and returns the processed images.

        Args:
            images (list[np.ndarray]): A list of input images.

        Returns:
            Tuple[list[Array2D], dict[str, Union[int, float, bool, str]]]:
                the return is a tuple that contains the list of processed images
                and a dict containing any relevant information that might be needed
                by analyze_image. Note, the state attirbutes can be updated directly,
                but that is not sufficient if intending to use multi-processing functionality
                available in Array2DAnalysis.
        """

        # Stack images for batch-level processing
        stack = np.stack(images, axis=0)
        stack = self.image_preprocess(stack)
        self.preprocessed = True
        logging.info(f'batch: {self.roi}')

        # Step 1: Learn background from percentile projection
        self.background.set_percentile_background_from_stack(stack=stack, percentile=2.5)
        stack = self.background.subtract(data=stack)

        # Step 2: Subtract per-image medians
        stack = self.background.subtract_imagewise_median(data=stack)

        # # Step 3: Generate and apply apodization
        # self.background.generate_apodization_mask(stack=stack, percentile=91, sigma=5)
        # stack = self.background.apply_apodization(data=stack)

        return list(stack), {'preprocessed':self.preprocessed}

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

        vertical_start = max(y_center - cross_height, 0)
        vertical_end = min(y_center + cross_height, image.shape[0])
        horizontal_start = max(x_center - cross_width, 0)
        horizontal_end = min(x_center + cross_width, image.shape[1])

        mask[vertical_start:vertical_end, x_center - thickness // 2:x_center + thickness // 2] = 0
        mask[y_center - thickness // 2:y_center + thickness // 2, horizontal_start:horizontal_end] = 0

        center = (float(x_center), float(y_center))
        m = cv2.getRotationMatrix2D(center, float(angle), 1.0).astype(np.float32)

        rotated_mask = cv2.warpAffine(
            mask,
            m,
            (image.shape[1], image.shape[0]),  # width, height
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT
        )

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
        if self.config.fiducial_cross1_location and self.config.fiducial_cross2_location:
            image = self.apply_cross_mask(image)
        self.background.set_constant_background(0)
        image = self.background.subtract(image)
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

            mask1 = self.create_cross_mask(img,
                                           tuple(self.config.fiducial_cross1_location),
                                           self.config.rotate)
            mask2 = self.create_cross_mask(img,
                                           tuple(self.config.fiducial_cross2_location),
                                           self.config.rotate)
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
        processed_image = image

        # if not self.preprocessed:
        #     processed_image = self.image_preprocess(image)

        # Check 'processed' flag from auxiliary_data (if provided), else fall back to self.preprocessed
        # doing this is essential to enable the multi processing in Array2DAnalysis
        processed_flag = auxiliary_data.get('preprocessed', self.preprocessed) if auxiliary_data else self.preprocessed
        fp = auxiliary_data.get('file_path','Unknown') if auxiliary_data else 'Unknown'
        logging.info(f'file path for this image was: {fp}')

        if not processed_flag:
            processed_image = self.image_preprocess(image)

        # → New block: compute vertical sum, apply Gaussian weighting, and sum result ←
        # 1. Sum over rows (axis=0) → produces a 1D array of length = number of columns
        vertical_lineout = np.sum(processed_image, axis=0)

        # 2. Build Gaussian (center at 400, sigma=20) matching the length of the lineout
        x = np.arange(vertical_lineout.shape[0])
        sigma = 20.0
        center = 250.0
        gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)

        # 3. Multiply lineout by Gaussian, then sum to get the optimization target
        weighted_lineout = vertical_lineout * gaussian
        optimization_target = np.sum(weighted_lineout)

        # Build the usual return dictionary (contains 'return_image', etc.)
        return_dictionary = self.build_return_dictionary(return_scalars={'optimization_target':optimization_target},
            return_image=processed_image,
            input_parameters=self.kwargs_dict
        )

        # return_dictionary = self.build_return_dictionary(return_image = processed_image,
        #                                                  input_parameters = self.kwargs_dict)

        if self.use_interactive:
            fig, ax = self.render_image(image=processed_image, input_params_dict=self.kwargs_dict)
            plt.show()
            plt.close(fig)

        return return_dictionary

    @staticmethod
    def render_image(
        image: np.ndarray,
        analysis_results_dict: Optional[dict[str, Union[float, int]]] = None,
        input_params_dict: Optional[dict[str, Union[float, int]]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = 'plasma',
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Overlay-enhanced version of the base renderer for VisaEBeam or similar.
        """
        fig, ax = base_render_image(
            image=image,
            analysis_results_dict=analysis_results_dict,
            input_params_dict=input_params_dict,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            ax=ax
        )

        # Optional overlays (Visa-style centroid)
        if input_params_dict and input_params_dict.get("blue_cent_x",None):
            cx = input_params_dict["blue_cent_x"] - input_params_dict.get("left_ROI", 0)
            cy = input_params_dict["blue_cent_y"] - input_params_dict.get("top_ROI", 0)
            ax.plot(cx, cy, 'bo', markersize=5)

        return fig, ax

if __name__ == "__main__":
    dev_name = 'U_BCaveMagSpec'
    test_dict = {'camera_name':dev_name}
    image_analyzer  = VisaEBeam(**test_dict)

    image_analyzer.use_interactive = True

    file_path = Path('/Volumes/hdna2/data/Undulator/Y2025/06-Jun/25_0605/scans/Scan018/U_BCaveMagSpec/Scan018_U_BCaveMagSpec_001.png')


    results = image_analyzer.analyze_image_file(image_filepath=file_path)
    print(results)
