"""EBeamProfileAnalyzer and configuration for electron beam profile imaging.

This module defines:
- `e_beam_camera_configs`: Predefined configurations for various E-beam cameras.
- `EBeamCamerConfig`: Dataclass encapsulating camera-specific settings.
- `EBeamProfileAnalyzer`: Analyzer for extracting beam statistics, Gaussian fits,
  and lineouts from E-beam profile images. Handles background subtraction, ROI cropping,
  and crosshair masking.

Intended for use with multiple beam imaging devices such as UC_VisaEBeam cameras,
ALine EBeam cameras, and spectrometer systems.

Dependencies
------------
- numpy
- cv2
- matplotlib
- dataclasses
- image_analysis base tools
"""

from __future__ import annotations

from typing import Union, Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

from image_analysis.base import ImageAnalyzer
from image_analysis.tools.rendering import base_render_image
from image_analysis.tools.basic_beam_stats import beam_profile_stats, flatten_beam_stats
from image_analysis.tools.background import Background
from image_analysis.tools.lcls_tools_gauss_fit import gauss_fit

# Predefined camera configurations
um = 1e-6
e_beam_camera_configs = {
    "HTT-C-ASSERTHighR": {
        "bkg_level": 1,
        "left_ROI": 10,
        "top_ROI": 10,
        "roi_width": 200,
        "roi_height": 200,
        "rotate": 0,
        "spatial_calibration": 19.6*um,
        "fiducial_cross1_location": [1, 1],  # placeholder, fill in if known
        "fiducial_cross2_location": [1, 1],  # placeholder, fill in if known
        "blue_cent_x": 1,  # placeholder
        "blue_cent_y": 1,  # placeholder
    },
}


@dataclass
class EBeamCamerConfig:
    """Dataclass encapsulating camera configuration parameters."""

    camera_name: str = "undefined"
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
    blue_cent_y: Optional[int] = None

    @staticmethod
    def from_camera_name(name: str) -> "EBeamCamerConfig":
        """Create a config instance from a camera name."""
        overrides = e_beam_camera_configs.get(name, {})
        return EBeamCamerConfig(camera_name=name, **overrides)


class EBeamProfileAnalyzer(ImageAnalyzer):
    """Analyzer for E-beam profile images with ROI, masking, and beam stats."""

    def __init__(
        self,
        background: Background = None,
        camera_name: str = None,
        preprocessed: bool = False,
    ):
        """
        Initialize analyzer for a specific E-beam camera.

        Parameters
        ----------
        background : Background, optional
            Background handler; generated if None.
        camera_name : str
            Camera configuration key (must be in `e_beam_camera_configs`).
        preprocessed : bool, default=False
            If True, skip preprocessing in subsequent calls.
        """
        if camera_name not in e_beam_camera_configs:
            raise ValueError(f"Unknown camera name: {camera_name}")

        config = EBeamCamerConfig.from_camera_name(camera_name)
        self.camera_name = camera_name
        self.config = config
        self.kwargs_dict = asdict(self.config)

        self.left = config.left_ROI
        self.top = config.top_ROI
        self.width = config.roi_width
        self.height = config.roi_height
        self.roi = (self.top, self.top + self.height, self.left, self.left + self.width)

        self.preprocessed = preprocessed
        self.run_analyze_image_asynchronously = True
        self.flag_logging = True
        self.min_val = 0
        self.use_interactive = False

        super().__init__(background=background)

    def apply_roi(self, data: np.ndarray) -> np.ndarray:
        """Crop image or image stack to configured ROI."""
        top, bottom, left, right = self.roi
        if data.ndim == 3:
            return data[:, top:bottom, left:right]
        elif data.ndim == 2:
            return data[top:bottom, left:right]
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    def analyze_image_batch(
        self, images: list[np.ndarray]
    ) -> Tuple[list[np.array], dict[str, Union[int, float, bool, str]]]:
        """
        Preprocess and background-subtract a batch of images.

        Parameters
        ----------
        images : list of numpy.ndarray
            List of images to process.

        Returns
        -------
        tuple
            (list of processed images, metadata dict with 'preprocessed' flag)
        """
        stack = np.stack(images, axis=0)
        stack = self.image_preprocess(stack)
        self.preprocessed = True
        stack = self.background.subtract_imagewise_mode(stack)
        return list(stack), {"preprocessed": self.preprocessed}

    @staticmethod
    def create_cross_mask(
        image: np.ndarray,
        cross_center: tuple[int, int],
        angle: Union[int, float],
        cross_height: int = 54,
        cross_width: int = 54,
        thickness: int = 10,
    ) -> np.ndarray:
        """Create binary mask with rotated cross centered at `cross_center`."""
        mask = np.ones_like(image, dtype=np.uint16)
        x_center, y_center = cross_center
        v_start, v_end = (
            max(y_center - cross_height, 0),
            min(y_center + cross_height, image.shape[0]),
        )
        h_start, h_end = (
            max(x_center - cross_width, 0),
            min(x_center + cross_width, image.shape[1]),
        )

        mask[v_start:v_end, x_center - thickness // 2 : x_center + thickness // 2] = 0
        mask[y_center - thickness // 2 : y_center + thickness // 2, h_start:h_end] = 0

        m = cv2.getRotationMatrix2D(
            (float(x_center), float(y_center)), float(angle), 1.0
        ).astype(np.float32)
        return cv2.warpAffine(
            mask,
            m,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def image_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Mask crosshairs, subtract constant background, and crop ROI."""
        if (
            self.config.fiducial_cross1_location
            and self.config.fiducial_cross2_location
        ):
            image = self.apply_cross_mask(image)
        self.background.set_constant_background(self.config.bkg_level)
        image = self.background.subtract(image)
        return self.apply_roi(image)

    def apply_cross_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply combined crosshair masks to 2D or 3D images."""

        def combined_mask(img: np.ndarray) -> np.ndarray:
            m1 = self.create_cross_mask(
                img, tuple(self.config.fiducial_cross1_location), self.config.rotate
            )
            m2 = self.create_cross_mask(
                img, tuple(self.config.fiducial_cross2_location), self.config.rotate
            )
            return m1 * m2

        if image.ndim == 3:
            mask = combined_mask(image[0])
            return image * mask[None, :, :]
        elif image.ndim == 2:
            return image * combined_mask(image)
        raise ValueError(f"Expected 2D or 3D array, got shape {image.shape}")

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[dict] = None
    ) -> dict[str, Union[float, int, str, np.ndarray]]:
        """Run beam analysis: preprocess, stats, Gaussian fit, lineouts."""
        processed_flag = (
            auxiliary_data.get("preprocessed", self.preprocessed)
            if auxiliary_data
            else self.preprocessed
        )
        fp = auxiliary_data.get("file_path", "Unknown") if auxiliary_data else "Unknown"
        logging.info(f"file path for this image was: {fp}")

        if not processed_flag:
            final_image = self.image_preprocess(image)
        else:
            final_image = image

        beam_stats_flat = flatten_beam_stats(
            beam_profile_stats(final_image), prefix=self.camera_name
        )
        gauss_params = {
            f"{self.camera_name}_{k}": v for k, v in gauss_fit(final_image).items()
        }

        horiz_lineout = final_image.sum(axis=0)
        vert_lineout = final_image.sum(axis=1)

        return_dict = self.build_return_dictionary(
            return_image=final_image,
            input_parameters=self.kwargs_dict,
            return_scalars={**beam_stats_flat, **gauss_params},
            return_lineouts=[horiz_lineout, vert_lineout],
        )

        if self.use_interactive:
            fig, ax = self.render_image(
                final_image,
                beam_stats_flat,
                self.kwargs_dict,
                [horiz_lineout, vert_lineout],
            )
            plt.show()
            plt.close(fig)

        return return_dict
    # Sam's static method
    # @staticmethod
    # def render_image(
    #     image: np.ndarray,
    #     analysis_results_dict: Optional[dict[str, Union[float, int]]] = None,
    #     input_params_dict: Optional[dict[str, Union[float, int]]] = None,
    #     lineouts: Optional[List[np.array]] = None,
    #     vmin: Optional[float] = None,
    #     vmax: Optional[float] = None,
    #     cmap: str = "jet",
    #     figsize: Tuple[float, float] = (4, 4),
    #     dpi: int = 150,
    #     ax: Optional[plt.Axes] = None,
    # ) -> tuple[plt.Figure, plt.Axes]:
    #     """Render image with optional beam centroid and lineouts overlay."""
    #     fig, ax = base_render_image(
    #         image=image,
    #         analysis_results_dict=analysis_results_dict,
    #         input_params_dict=input_params_dict,
    #         lineouts=lineouts,
    #         vmin=vmin,
    #         vmax=vmax,
    #         cmap=cmap,
    #         figsize=figsize,
    #         dpi=dpi,
    #         ax=ax,
    #     )

    #     # if input_params_dict and input_params_dict.get("blue_cent_x", None):
    #     #     cx = input_params_dict["blue_cent_x"] - input_params_dict.get("left_ROI", 0)
    #     #     cy = input_params_dict["blue_cent_y"] - input_params_dict.get("top_ROI", 0)
    #     #     ax.plot(cx, cy, "bo", markersize=5)

    #     if lineouts and len(lineouts) == 2:
    #         horiz, vert = np.clip(lineouts[0], 0, None), np.clip(lineouts[1], 0, None)
    #         img_h, img_w = image.shape
    #         horiz_norm = horiz / np.max(horiz) * img_h * 0.2
    #         vert_norm = vert / np.max(vert) * img_w * 0.2
    #         ax.plot(np.arange(len(horiz)), img_h - horiz_norm, color="cyan", lw=1.0)
    #         ax.plot(vert_norm, np.arange(len(vert)), color="magenta", lw=1.0)

    #     return fig, ax
    
    @staticmethod
    def render_image(
        image: np.ndarray,
        analysis_results_dict: Optional[dict[str, Union[float, int]]] = None,
        input_params_dict: Optional[dict[str, Union[float, int]]] = None,
        lineouts: Optional[List[np.array]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "jet",
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Render image with optional beam centroid and lineouts overlay."""

        # --- (This section is the same as before) ---
        spatial_cal = input_params_dict.get("spatial_calibration", 1.0)
        unit_scale = 1e6 
        units = "µm"
        cal_scale = spatial_cal * unit_scale

        img_h, img_w = image.shape
        phys_height = img_h * cal_scale
        phys_width = img_w * cal_scale
        extent = [0, phys_width, phys_height, 0]
        # --------------------------------------------

        # --- MODIFICATION: Replace base_render_image call ---
        # Instead of calling base_render_image, we replicate its core functionality here.
        # This allows us to correctly pass the 'extent' argument to imshow.
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        ax.imshow(
            image,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent  # Use the calculated extent here
        )
        # --- End of Modification ---

        ax.set_xlabel(f"X ({units})")
        ax.set_ylabel(f"Y ({units})")
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

        # (Lineout plotting code remains the same as before)
        if lineouts and len(lineouts) == 2:
            horiz, vert = np.clip(lineouts[0], 0, None), np.clip(lineouts[1], 0, None)
            
            horiz_norm = horiz / np.max(horiz) * phys_height * 0.2
            vert_norm = vert / np.max(vert) * phys_width * 0.2

            x_coords = np.arange(len(horiz)) * cal_scale
            y_coords = np.arange(len(vert)) * cal_scale

            ax.plot(x_coords, phys_height - horiz_norm, color="cyan", lw=1.0)
            ax.plot(vert_norm, y_coords, color="magenta", lw=1.0)

        return fig, ax

if __name__ == "__main__":
    dev_name = "UC_ALineEBeam3"
    analyzer = EBeamProfileAnalyzer(camera_name=dev_name)
    analyzer.use_interactive = True
    script_dir = Path(__file__).resolve()
    for _ in range(4):
        script_dir = script_dir.parent
    file_path = (
        script_dir / "tests" / "data" / "VisaEBeam_test_data" / f"{dev_name}_001.png"
    )
    results = analyzer.analyze_image_file(image_filepath=file_path)
    print(results)
