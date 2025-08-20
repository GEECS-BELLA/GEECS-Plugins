"""HiResMagCam analyzer using a bowtie fit to derive an emittance proxy.

This module defines `HiResMagCamAnalyzer`, a specialization of
`EBeamProfileAnalyzer` for the high‑resolution magnetic camera. It applies
preprocessing (optional fiducial masking, per‑image mode background subtraction,
ROI crop) and evaluates a bowtie fit to produce a scalar score that can be used
as an emittance proxy. It also returns lineouts suitable for quick diagnostics.

Notes
-----
- The bowtie fit is provided by `image_analysis.algorithms.bowtie_fit`.
- The analyzer expects images that can be safely cast to `float32`.
- When `use_interactive` is True, `render_image` overlays a normalized lineout
  for fast visual feedback.
"""

from __future__ import annotations

from typing import Union, Optional, Tuple, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from image_analysis.tools.rendering import base_render_image
from image_analysis.offline_analyzers.Undulator.EBeamProfile import EBeamProfileAnalyzer
from image_analysis.algorithms.bowtie_fit import BowtieFitAlgorithm

import logging


class HiResMagCamAnalyzer(EBeamProfileAnalyzer):
    """Analyzer for HiResMagCam images that reports a bowtie-fit emittance proxy."""

    def __init__(self, camera_name: str = None):
        """Initialize analyzer and configure the bowtie fit algorithm.

        Parameters
        ----------
        camera_name : str, optional
            Camera configuration key passed through to `EBeamProfileAnalyzer`.
            If provided, it should match a key in your camera config registry.
        """
        self.algo = BowtieFitAlgorithm(
            n_beam_size_clearance=4, min_total_counts=2500, threshold_factor=10
        )

        self.run_analyze_image_asynchronously = True

        super().__init__(camera_name=camera_name)

    def image_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image(s): optional fiducial masking, per-image mode subtract, ROI.

        Steps
        -----
        1. Cast to ``float32``.
        2. If both fiducial locations are provided, apply crosshair masks.
        3. Subtract a per-image background using the mode of each image.
        4. Crop to the configured ROI.

        Parameters
        ----------
        image : numpy.ndarray
            Either a single 2D image ``(H, W)`` or a stack ``(N, H, W)``.

        Returns
        -------
        numpy.ndarray
            Preprocessed image or image stack in the same shape as input.
        """
        image = image.astype(np.float32)
        if (
            self.config.fiducial_cross1_location
            and self.config.fiducial_cross2_location
        ):
            image = self.apply_cross_mask(image)
        image = self.background.subtract_imagewise_mode(image)
        image = self.apply_roi(image)
        return image

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[dict] = None
    ) -> dict[str, Union[float, int, str, np.ndarray]]:
        """Run preprocessing, evaluate bowtie fit, and return results dictionary.

        Parameters
        ----------
        image : numpy.ndarray
            Input image for analysis.
        auxiliary_data : dict, optional
            Optional metadata. Recognized keys:
            - 'preprocessed' (bool): If True, skip internal preprocessing.
            - 'file_path' (str or pathlib.Path): Used only for logging.

        Returns
        -------
        dict
            Standard `ImageAnalyzer` result dictionary with:
            - ``return_image`` : numpy.ndarray
                Final processed image (thresholded at 10 counts).
            - ``return_scalars`` : dict
                Scalars including:
                * ``f'{camera_name}:emittance_proxy'`` : float bowtie score.
                * ``f'{camera_name}:total_counts'`` : float total counts in final image.
            - ``return_lineouts`` : list of numpy.ndarray
                Two 1D arrays: ``[sizes, weights]`` from the bowtie fit.
        """
        processed_flag = (
            auxiliary_data.get("preprocessed", self.preprocessed)
            if auxiliary_data
            else self.preprocessed
        )
        fp = auxiliary_data.get("file_path", "Unknown") if auxiliary_data else "Unknown"
        logging.info(f"file path for this image was: {fp}")

        if not processed_flag:
            preprocessed_image = self.image_preprocess(image.astype(np.float32))
        else:
            preprocessed_image = image

        final_image = preprocessed_image
        final_image[final_image < 10] = 0

        bowtie_result = self.algo.evaluate(final_image)

        lineouts = [np.array(bowtie_result.sizes), np.array(bowtie_result.weights)]

        return_dictionary = self.build_return_dictionary(
            return_scalars={
                f"{self.camera_name}:emittance_proxy": bowtie_result.score,
                f"{self.camera_name}:total_counts": np.sum(final_image),
            },
            return_image=final_image,
            input_parameters=self.kwargs_dict,
            return_lineouts=lineouts,
        )

        if self.use_interactive:
            fig, ax = self.render_image(
                image=final_image, input_params_dict=self.kwargs_dict, lineouts=lineouts
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
        """Render the image and overlay the normalized weight lineout if provided.

        Parameters
        ----------
        image : numpy.ndarray
            Image to display.
        analysis_results_dict : dict, optional
            Scalar results to annotate (unused here but passed to base renderer).
        input_params_dict : dict, optional
            Input parameters to annotate (unused here but passed to base renderer).
        lineouts : list of numpy.ndarray, optional
            If provided, the second element is treated as the weight lineout
            and plotted after normalization.
        vmin, vmax : float, optional
            Color limits for the image.
        cmap : str, default='plasma'
            Colormap for image rendering.
        figsize : tuple of float, default=(4, 4)
            Figure size in inches.
        dpi : int, default=150
            Figure DPI.
        ax : matplotlib.axes.Axes, optional
            If provided, draw into this axes; otherwise create a new figure.

        Returns
        -------
        tuple
            ``(figure, axes)`` from Matplotlib.
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

        lineout = lineouts[1]
        # Optional overlay of a line
        if lineout is not None:
            x_vals = np.arange(len(lineout))

            # Auto-normalize lineout to its own max (avoids assumption of "expected max").
            if np.max(lineout) > 0:
                norm_lineout = lineout / np.max(lineout)
            else:
                norm_lineout = lineout

            # Scale to a reasonable fraction of the image height.
            scale = image.shape[0] * 0.3  # use 30% of height
            y_vals = norm_lineout * scale  # no offset applied

            ax.plot(x_vals, y_vals, color="cyan", linewidth=1.0, zorder=10)
            ax.set_ylim([0, image.shape[0]])

        return fig, ax


if __name__ == "__main__":
    dev_name = "UC_HiResMagCam"
    test_dict = {"camera_name": dev_name}
    image_analyzer = HiResMagCamAnalyzer(**test_dict)

    image_analyzer.use_interactive = True

    # file_path = Path('/Volumes/hdna2/data/Undulator/Y2025/06-Jun/25_0605/scans/Scan018/U_BCaveMagSpec/Scan018_U_BCaveMagSpec_001.png')
    file_path = Path(
        "Z:/data/Undulator/Y2025/04-Apr/25_0429/scans/Scan015/UC_HiResMagCam/Scan015_UC_HiResMagCam_004.png"
    )

    results = image_analyzer.analyze_image_file(image_filepath=file_path)
    print(results)
