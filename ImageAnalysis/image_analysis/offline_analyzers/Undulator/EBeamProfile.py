from __future__ import annotations

from typing import Union, Optional, Tuple, List
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataclasses import dataclass,asdict

from image_analysis.base import ImageAnalyzer
from image_analysis.tools.rendering import base_render_image
from image_analysis.tools.basic_beam_stats import beam_profile_stats, flatten_beam_stats
from image_analysis.tools.background import Background
from image_analysis.tools.lcls_tools_gauss_fit import gauss_fit

import logging

e_beam_camera_configs = {
    "UC_HiResMagCam": {
        "bkg_level": 60,
        "left_ROI": 1,
        "top_ROI": 1,
        "roi_width": 1287,
        "roi_height": 325,
        "rotate": 0,
        "spatial_calibration": 0.00002217
    },
    "U_BCaveMagSpec": {
        "bkg_level": 60,
        "left_ROI": 200,
        "top_ROI": 1,
        "roi_width": 500,
        "roi_height": 300,
        "rotate": 0,
        "spatial_calibration": 0.00002217
    },
    "UC_ALineEbeam1": {
        "bkg_level": 60,
        "left_ROI": 1,
        "top_ROI": 1,
        "roi_width": 1288,
        "roi_height": 963,
        "rotate": 0,
        "spatial_calibration": 0.00002217
    },
    "UC_ALineEBeam2": {
        "bkg_level": 18,
        "left_ROI": 1,
        "top_ROI": 1,
        "roi_width": 1288,
        "roi_height": 963,
        "rotate": 0,
        "spatial_calibration": 0.00002394
    },
    "UC_ALineEBeam3": {
        "bkg_level": 0,
        "left_ROI": 175,    #orginal value: 180
        "top_ROI": 100,     #orginal value: 200
        "roi_width": 800,   #orginal value: 500
        "roi_height": 800,  #orginal value: 500
        "rotate": 0,
        "spatial_calibration": 0.0000244
    },
    "UC_VisaEBeam1": {
        "bkg_level": 15,
        "left_ROI": 558,
        "top_ROI": 290,
        "roi_width": 333,
        "roi_height": 333,
        "rotate": 0,
        "spatial_calibration": 0.0000075,
        "fiducial_cross1_location": [722, 285],
        "fiducial_cross2_location": [728, 636],
        "blue_cent_x": 764,
        "blue_cent_y": 449
    },
    "UC_VisaEBeam2": {
        "bkg_level": 20,
        "left_ROI": 110,
        "top_ROI": 75,
        "roi_width": 333,
        "roi_height": 333,
        "rotate": 0,
        "spatial_calibration": 0.0000075,
        "fiducial_cross1_location": [193, 97],
        "fiducial_cross2_location": [193, 465],
        "blue_cent_x": 216,
        "blue_cent_y": 252
    },
    "UC_VisaEBeam3": {
        "bkg_level": 15,
        "left_ROI": 160,
        "top_ROI": 140,
        "roi_width": 333,
        "roi_height": 333,
        "rotate": -3.5,
        "spatial_calibration": 0.0000075,
        "fiducial_cross1_location": [374, 127],
        "fiducial_cross2_location": [338, 485],
        "blue_cent_x": 315,
        "blue_cent_y": 264
    },
    "UC_VisaEBeam4": {
        "bkg_level": 15,
        "left_ROI": 180,
        "top_ROI": 200,
        "roi_width": 333,
        "roi_height": 333,
        "rotate": 2.5,
        "spatial_calibration": 0.0000075,
        "fiducial_cross1_location": [268, 175],
        "fiducial_cross2_location": [287, 533],
        "blue_cent_x": 318,
        "blue_cent_y": 336
    },
    "UC_VisaEBeam5": {
        "bkg_level": 25,
        "left_ROI": 135,
        "top_ROI": 90,
        "roi_width": 333,
        "roi_height": 333,
        "rotate": 0,
        "spatial_calibration": 0.0000075,
        "fiducial_cross1_location": [267, 95],
        "fiducial_cross2_location": [262, 440],
        "blue_cent_x": 270,
        "blue_cent_y": 232
    },
    "UC_VisaEBeam6": {
        "bkg_level": 25,
        "left_ROI": 195,
        "top_ROI": 85,
        "roi_width": 333,
        "roi_height": 333,
        "rotate": 0,
        "spatial_calibration": 0.0000075,
        "fiducial_cross1_location": [295, 85],
        "fiducial_cross2_location": [290, 425],
        "blue_cent_x": 289,
        "blue_cent_y": 204
    },
    "UC_VisaEBeam7": {
        "bkg_level": 15,
        "left_ROI": 120,
        "top_ROI": 100,
        "roi_width": 333,
        "roi_height": 333,
        "rotate": -8,
        "spatial_calibration": 0.0000075,
        "fiducial_cross1_location": [275, 328],
        "fiducial_cross2_location": [275, 550],
        "blue_cent_x": 201,
        "blue_cent_y": 289
    },
    "UC_VisaEBeam8": {
        "bkg_level": 30,
        "left_ROI": 105,
        "top_ROI": 100,
        "roi_width": 333,
        "roi_height": 333,
        "rotate": 0,
        "spatial_calibration": 0.0000075,
        "fiducial_cross1_location": [1, 1],   # placeholder, fill in if known
        "fiducial_cross2_location": [1, 1],   # placeholder, fill in if known
        "blue_cent_x": 1,                     # placeholder
        "blue_cent_y": 1                      # placeholder
    }
}

@dataclass
class EBeamCamerConfig:
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
    def from_camera_name(name: str) -> "EBeamCamerConfig":
        overrides = e_beam_camera_configs.get(name, {})
        return EBeamCamerConfig(camera_name=name, **overrides)

class EBeamProfileAnalyzer(ImageAnalyzer):
    def __init__(self, background: Background = None, camera_name: str = None, preprocessed: bool = False,):
        """
        Initialize EBeamProfileAnalyzer image_analyzer for a specific camera.

        Parameters
        ----------
        camera_name : str
            The key for the desired camera config (e.g. 'UC_VisaEBeam3')
        background : Optional[Background]
            background object to use with this image_analyzer. If none is sent, an empty one will be generated by super
        preprocessed : bool
            this is a flag to track if the preprocessor has already run so that it be skipped on subsequent calls
            of analyze_image
        """
        if camera_name not in e_beam_camera_configs:
            raise ValueError(f"Unknown camera name: {camera_name}")

        config = EBeamCamerConfig.from_camera_name(camera_name)
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
                            ) -> Tuple[list[np.array], dict[str, Union[int, float, bool, str]]]:
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

        stack = self.background.subtract_imagewise_mode(data=stack)

        # # Step 1: Learn background from percentile projection
        # self.background.set_percentile_background_from_stack(stack=stack, percentile=2.5)
        # stack = self.background.subtract(data=stack)
        #
        # # Step 2: Subtract per-image medians
        # stack = self.background.subtract_imagewise_median(data=stack)

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
        self.background.set_constant_background(self.config.bkg_level)
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
        # Check 'processed' flag from auxiliary_data (if provided), else fall back to self.preprocessed
        # doing this is essential to enable the multi processing in Array2DAnalysis
        processed_flag = auxiliary_data.get('preprocessed', self.preprocessed) if auxiliary_data else self.preprocessed
        fp = auxiliary_data.get('file_path','Unknown') if auxiliary_data else 'Unknown'
        logging.info(f'file path for this image was: {fp}')

        if not processed_flag:
            preprocessed_image = self.image_preprocess(image)
        else:
            preprocessed_image = image

        final_image = preprocessed_image
        beam_stats = beam_profile_stats(final_image)
        beam_stats_flat_dict = flatten_beam_stats(stats=beam_stats, prefix=self.camera_name)

        gauss_fit_params = gauss_fit(final_image)
        gauss_fit_params = {f"{self.camera_name}_{k}": v for k, v in gauss_fit_params.items()}

        # Generate horizontal and vertical lineouts (projections)
        horizontal_lineout = final_image.sum(axis=0)  # sum over rows → shape (width,)
        vertical_lineout = final_image.sum(axis=1)  # sum over columns → shape (height,)
        lineouts = [horizontal_lineout, vertical_lineout]

        return_dictionary = self.build_return_dictionary(return_image = final_image,
                                                         input_parameters = self.kwargs_dict,
                                                         return_scalars={**beam_stats_flat_dict,**gauss_fit_params},
                                                         return_lineouts=lineouts)

        if self.use_interactive:
            fig, ax = self.render_image(image=final_image,
                                        input_params_dict=self.kwargs_dict,
                                        analysis_results_dict=beam_stats,
                                        lineouts = lineouts)
            plt.show()
            plt.close(fig)

        return return_dictionary

    @staticmethod
    def render_image(
        image: np.ndarray,
        analysis_results_dict: Optional[dict[str, Union[float, int]]] = None,
        input_params_dict: Optional[dict[str, Union[float, int]]] = None,
        lineouts: Optional[List[np.array]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = 'plasma',
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Overlay-enhanced version of the base renderer for EBeamProfileAnalyzer or similar.
        """
        fig, ax = base_render_image(
            image=image,
            analysis_results_dict=analysis_results_dict,
            input_params_dict=input_params_dict,
            lineouts=lineouts,
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

        # Overlay lineouts if provided
        if lineouts is not None and len(lineouts) == 2:
            horiz, vert = lineouts
            img_height, img_width = image.shape

            # Clip negative values to 0
            horiz = np.clip(horiz, 0, None)
            vert = np.clip(vert, 0, None)

            # Normalize for overlay
            horiz_norm = horiz / np.max(horiz) * img_height * 0.2
            vert_norm = vert / np.max(vert) * img_width * 0.2

            # Overlay horizontal lineout at bottom (grows up from bottom edge)
            ax.plot(np.arange(len(horiz)), img_height - horiz_norm, color='cyan', lw=1.0, label='Horizontal lineout')

            # Overlay vertical lineout at left (grows right from left edge)
            ax.plot(vert_norm, np.arange(len(vert)), color='magenta', lw=1.0, label='Vertical lineout')

        return fig, ax

if __name__ == "__main__":
    dev_name = 'UC_VisaEBeam1'
    dev_name = 'UC_ALineEBeam3'
    test_dict = {'camera_name':dev_name}
    image_analyzer  = EBeamProfileAnalyzer(**test_dict)

    image_analyzer.use_interactive = True
    # Resolve path relative to this file
    script_dir = Path(__file__).resolve()
    for _ in range(4):
        script_dir = script_dir.parent
    file_path = script_dir / "tests" / "data" / "VisaEBeam_test_data" / f'{dev_name}_001.png'
    results = image_analyzer.analyze_image_file(image_filepath=file_path)
    print(results)
