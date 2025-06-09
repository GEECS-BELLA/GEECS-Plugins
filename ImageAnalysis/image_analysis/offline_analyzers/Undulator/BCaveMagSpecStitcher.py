from __future__ import annotations

from typing import Union, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from image_analysis.tools.rendering import base_render_image
from image_analysis.tools.basic_beam_stats import beam_profile_stats
from image_analysis.offline_analyzers.Undulator.EBeamProfile import EBeamProfileAnalyzer

import logging

class BCaveMagSpecStitcherAnalyzer(EBeamProfileAnalyzer):

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
        processed_flag = auxiliary_data.get('preprocessed', self.preprocessed) if auxiliary_data else self.preprocessed
        fp = auxiliary_data.get('file_path','Unknown') if auxiliary_data else 'Unknown'
        logging.info(f'file path for this image was: {fp}')

        if not processed_flag:
            preprocessed_image = self.image_preprocess(image)
        else:
            preprocessed_image = image

        final_image = preprocessed_image
        # → New block: compute vertical sum, apply Gaussian weighting, and sum result ←
        # 1. Sum over rows (axis=0) → produces a 1D array of length = number of columns
        vertical_lineout = np.sum(final_image, axis=0)

        # 2. Build Gaussian (center at 400, sigma=20) matching the length of the lineout
        x = np.arange(vertical_lineout.shape[0])
        sigma = 20.0
        center = 250.0
        gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)

        # 3. Multiply lineout by Gaussian, then sum to get the optimization target
        weighted_lineout = vertical_lineout * gaussian
        optimization_target = np.sum(weighted_lineout)

        # Build the usual return dictionary (contains 'return_image', etc.)
        return_dictionary = self.build_return_dictionary(return_scalars={'optimization_target': optimization_target},
                                                         return_image=final_image,
                                                         input_parameters=self.kwargs_dict,
                                                         return_lineouts=[weighted_lineout]
                                                         )

        if self.use_interactive:
            fig, ax = self.render_image(image=final_image, input_params_dict=self.kwargs_dict, lineouts=[weighted_lineout])
            plt.show()
            plt.close(fig)

        return return_dictionary

    @staticmethod
    def render_image(
        image: np.ndarray,
        analysis_results_dict: Optional[dict[str, Union[float, int]]] = None,
        input_params_dict: Optional[dict[str, Union[float, int]]] = None,
        lineouts: Optional[np.array] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = 'plasma',
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
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

        lineout = lineouts[0]
        # Optional overlay of a line
        if lineout is not None:
            x_vals = np.arange(len(lineout))

            # STATIC normalization based on a fixed expected max value
            max_expected_value = 1e3  # <-- adjust this to your typical lineout range
            norm_lineout = np.clip(lineout / max_expected_value, 0, 1)

            scale = image.shape[0] * 0.3  # scale to 30% of image height
            y_offset = int(image.shape[0] * 0.1) * 0

            y_vals = y_offset + norm_lineout * scale

            ax.plot(x_vals, y_vals, color='cyan', linewidth=1.0, zorder=10)

            # Optional: force redraw limits
            ax.set_ylim([0, image.shape[0]])

        return fig, ax

if __name__ == "__main__":
    dev_name = 'U_BCaveMagSpec'
    test_dict = {'camera_name':dev_name}
    image_analyzer  = BCaveMagSpecStitcherAnalyzer(**test_dict)

    image_analyzer.use_interactive = True

    file_path = Path('/Volumes/hdna2/data/Undulator/Y2025/06-Jun/25_0605/scans/Scan018/U_BCaveMagSpec/Scan018_U_BCaveMagSpec_001.png')

    results = image_analyzer.analyze_image_file(image_filepath=file_path)
