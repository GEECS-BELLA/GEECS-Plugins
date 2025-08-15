"""Analyzer for BCaveMagSpec stitched images with Gaussian-weighted optimization target.

This module defines `BCaveMagSpecStitcherAnalyzer`, a subclass of `EBeamProfileAnalyzer`,
to process stitched spectrometer images from BCaveMagSpec. It applies preprocessing
(crosshair masking, ROI cropping, background subtraction), computes a vertical sum
lineout, applies a Gaussian weighting, and returns the weighted sum as an optimization
target. Optional interactive rendering with overlaid lineouts is supported.
"""

from __future__ import annotations

from typing import Union, Optional, Tuple, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from image_analysis.tools.rendering import base_render_image
from image_analysis.offline_analyzers.Undulator.EBeamProfile import EBeamProfileAnalyzer

import logging


class BCaveMagSpecStitcherAnalyzer(EBeamProfileAnalyzer):
    """Image analyzer for BCaveMagSpec stitched spectrometer images."""

    def image_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing: crosshair mask, background subtraction, ROI crop.

        Steps
        -----
        1. If both fiducial crosshair locations are configured, apply crosshair masks.
        2. Set a constant background of 0 and subtract it from the image.
        3. Crop the image to the configured region of interest.

        Parameters
        ----------
        image : numpy.ndarray
            2D single image `(H, W)` or 3D stack `(N, H, W)`.

        Returns
        -------
        numpy.ndarray
            Preprocessed image or image stack.
        """
        if (
            self.config.fiducial_cross1_location
            and self.config.fiducial_cross2_location
        ):
            image = self.apply_cross_mask(image)
        self.background.set_constant_background(0)
        image = self.background.subtract(image)
        image = self.apply_roi(image)
        return image

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[dict] = None
    ) -> dict[str, Union[float, int, str, np.ndarray]]:
        """Analyze BCaveMagSpec stitched image and compute Gaussian-weighted lineout sum.

        Parameters
        ----------
        image : numpy.ndarray
            Input image for analysis.
        auxiliary_data : dict, optional
            Optional metadata. Recognized keys:
            - 'preprocessed' (bool): True if image is already preprocessed.
            - 'file_path' (str or Path): Source file path for logging.

        Returns
        -------
        dict
            Standard `ImageAnalyzer` return dictionary containing:
            - 'optimization_target' : float, Gaussian-weighted sum of vertical lineout
            - 'return_image' : numpy.ndarray, final processed image
            - 'return_lineouts' : list containing the weighted lineout array
        """
        processed_flag = (
            auxiliary_data.get("preprocessed", self.preprocessed)
            if auxiliary_data
            else self.preprocessed
        )
        fp = auxiliary_data.get("file_path", "Unknown") if auxiliary_data else "Unknown"
        logging.info(f"file path for this image was: {fp}")

        if not processed_flag:
            preprocessed_image = self.image_preprocess(image)
        else:
            preprocessed_image = image

        final_image = preprocessed_image

        vertical_lineout = np.sum(final_image, axis=0)
        x = np.arange(vertical_lineout.shape[0])
        sigma = 20.0
        center = 250.0
        gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)

        weighted_lineout = vertical_lineout * gaussian
        optimization_target = np.sum(weighted_lineout)

        return_dictionary = self.build_return_dictionary(
            return_scalars={"optimization_target": optimization_target},
            return_image=final_image,
            input_parameters=self.kwargs_dict,
            return_lineouts=[weighted_lineout],
        )

        if self.use_interactive:
            fig, ax = self.render_image(
                image=final_image,
                input_params_dict=self.kwargs_dict,
                lineouts=[weighted_lineout],
            )
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
        cmap: str = "plasma",
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Render image with optional overlaid normalized lineout.

        Parameters
        ----------
        image : numpy.ndarray
            Image to display.
        analysis_results_dict : dict, optional
            Scalar analysis results to annotate on the image.
        input_params_dict : dict, optional
            Input parameters to annotate on the image.
        lineouts : list of numpy.ndarray, optional
            List containing 1D lineout arrays for overlay; only the first is plotted.
        vmin, vmax : float, optional
            Color scale limits.
        cmap : str, default='plasma'
            Colormap for image rendering.
        figsize : tuple of float, default=(4, 4)
            Figure size in inches.
        dpi : int, default=150
            Figure resolution in dots per inch.
        ax : matplotlib.axes.Axes, optional
            Existing Axes to draw on; if None, a new figure/axes is created.

        Returns
        -------
        tuple
            Matplotlib (Figure, Axes) with rendered image and overlays.
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
            ax=ax,
        )

        lineout = lineouts[0]
        if lineout is not None:
            x_vals = np.arange(len(lineout))
            max_expected_value = 1e3
            norm_lineout = np.clip(lineout / max_expected_value, 0, 1)
            scale = image.shape[0] * 0.3
            y_offset = int(image.shape[0] * 0.1) * 0
            y_vals = y_offset + norm_lineout * scale
            ax.plot(x_vals, y_vals, color="cyan", linewidth=1.0, zorder=10)
            ax.set_ylim([0, image.shape[0]])

        return fig, ax


if __name__ == "__main__":
    dev_name = "U_BCaveMagSpec"
    test_dict = {"camera_name": dev_name}
    image_analyzer = BCaveMagSpecStitcherAnalyzer(**test_dict)
    image_analyzer.use_interactive = True
    file_path = Path(
        "/Volumes/hdna2/data/Undulator/Y2025/06-Jun/25_0605/scans/Scan018/U_BCaveMagSpec/Scan018_U_BCaveMagSpec_001.png"
    )
    results = image_analyzer.analyze_image_file(image_filepath=file_path)
