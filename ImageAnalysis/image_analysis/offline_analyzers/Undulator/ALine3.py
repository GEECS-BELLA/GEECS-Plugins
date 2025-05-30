from __future__ import annotations

from typing import Union, Optional
from pathlib import Path

import numpy as np
from scipy.ndimage import label, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.pyplot import Normalize
from matplotlib.patches import Ellipse

from image_analysis.base import ImageAnalyzer
from image_analysis.tools.rendering import base_render_image

import logging

class Aline3Analyzer(ImageAnalyzer):

    def __init__(self):
        """
        Parameters
        ----------
        """
        self.run_analyze_image_asynchronously = True
        self.flag_logging = True
        self.min_val = 0
        self.use_interactive = False

        super().__init__()

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
        stack = stack[:, 175:775, 175:775]

        # Step 1: Learn background from percentile projection
        self.background_obj.set_percentile_background_from_stack(stack=stack, percentile=2.5)
        stack = self.background_obj.subtract(data=stack)

        # Step 2: Subtract per-image medians
        stack = self.background_obj.subtract_imagewise_median(data=stack)

        # Step 3: Generate and apply apodization
        self.background_obj.generate_apodization_mask(stack=stack, percentile=91, sigma=5)
        stack = self.background_obj.apply_apodization(data=stack)

        # Optionally record min value for later use
        self.min_val = np.min(stack)
        logging.info(f'min value from the stack: {self.min_val}')

        return list(stack)  # maintain list format for downstream compatibility

    @staticmethod
    def compute_fwhm(profile: np.ndarray) -> float:
        """
        Compute the full width at half maximum (FWHM) of a 1D profile.
        Uses linear interpolation between neighboring points.
        """
        profile = np.asarray(profile, dtype=float)
        profile -= profile.min()  # remove baseline
        max_val = profile.max()
        if max_val == 0:
            return 0.0

        half_max = max_val / 2
        indices = np.where(profile >= half_max)[0]

        if len(indices) < 2:
            return 0.0

        left, right = indices[0], indices[-1]

        # Optional: refine edges using linear interpolation
        def interp_edge(i1, i2):
            y1, y2 = profile[i1], profile[i2]
            return i1 + (half_max - y1) / (y2 - y1)

        if left > 0:
            left_edge = interp_edge(left - 1, left)
        else:
            left_edge = left

        if right < len(profile) - 1:
            right_edge = interp_edge(right, right + 1)
        else:
            right_edge = right

        return right_edge - left_edge

    def beam_profile_stats(self, img: np.ndarray):
        img = np.clip(img.astype(float), 0, None)
        total = img.sum()
        if total == 0:
            raise ValueError("Image has zero total intensity.")

        x_proj = img.sum(axis=0)
        y_proj = img.sum(axis=1)

        x = np.arange(x_proj.size)
        y = np.arange(y_proj.size)

        x_mean = np.sum(x * x_proj) / x_proj.sum()
        x_rms = np.sqrt(np.sum((x - x_mean) ** 2 * x_proj) / x_proj.sum())
        x_fwhm = self.compute_fwhm(x_proj)
        x_peak = np.argmax(x_proj)

        y_mean = np.sum(y * y_proj) / y_proj.sum()
        y_rms = np.sqrt(np.sum((y - y_mean) ** 2 * y_proj) / y_proj.sum())
        y_fwhm = self.compute_fwhm(y_proj)
        y_peak = np.argmax(y_proj)

        return {
            "ALine3_x_mean": x_mean,
            "ALine3_x_rms": x_rms,
            "ALine3_x_fwhm": x_fwhm,
            "ALine3_x_peak": x_peak,
            "ALine3_y_mean": y_mean,
            "ALine3_y_rms": y_rms,
            "ALine3_y_fwhm": y_fwhm,
            "ALine3_y_peak": y_peak
        }

    def analyze_image(self, image: np.ndarray, auxiliary_data: Optional[dict] = None) -> dict[
        str, Union[float, int, str, np.ndarray]]:

        """
        Analyze an image from acave mag cam3.

        Parameters
        ----------
        image : np.array,
            the image.
        auxiliary_data: dict, containing any additional imformation needed for analysis

        Returns
        -------
        dict
            A dictionary with the processed image and placeholder for analysis results.
        """

        beam_stats = self.beam_profile_stats(image)
        analyzed_image = (image-self.min_val).astype(np.uint16)

        return_dictionary = self.build_return_dictionary(return_image=analyzed_image, return_scalars=beam_stats)

        if self.use_interactive:
            fig, ax = self.render_image(image=analyzed_image)
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

        return fig, ax


if __name__ == "__main__":
    image_analyzer  = Aline3Analyzer()
    image_analyzer.use_interactive = True
    # file_path = Path('Z:\\data\\Undulator\\Y2025\\05-May\\25_0507\\scans\\Scan029\\UC_ALineEBeam3\\Scan029_UC_ALineEBeam3_001.png')
    # file_path = Path('\Volumes\hdna2\data\Undulator\Y2025\05-May\25_0507\scans\Scan029\UC_ALineEBeam3\Scan029_UC_ALineEBeam3_001.png')
    file_path = Path('/Volumes/hdna2/data/Undulator/Y2025/05-May/25_0507/scans/Scan029/UC_ALineEBeam3/Scan029_UC_ALineEBeam3_001.png')


    print(file_path.exists())
    results = image_analyzer.analyze_image_file(image_filepath=file_path)
    print(results)