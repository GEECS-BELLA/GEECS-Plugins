"""BCaveMagSpecCam1 analyzer using a bowtie fit to derive emittance metrics.

This module defines `BCaveMagSpecCam1Analyzer`, a specialization of
`BeamAnalyzer` for the B-cave magnetic spectrometer camera 1
(``UC_BCaveMagSpecCam1``). It uses the standard BeamAnalyzer processing
pipeline, then fits the dispersed trace with the bowtie model to report an
emittance proxy alongside the fitted waist, divergence and fit quality.

It is a sibling of `hi_res_mag_cam_analyzer` rather than a reuse of it: the
two cameras share the bowtie algorithm but not their tuning, so keeping the
classes separate means B-cave parameter changes cannot regress HiResMagCam.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from image_analysis.tools.rendering import base_render_image
from geecs_schemas.analysis import BCaveMagSpecCam1Spec

from image_analysis.analyzers.beam_analyzer import BeamAnalyzer
from geecs_schemas.analysis.processing_2d import CameraConfig
from image_analysis.algorithms.bowtie_fit import BowtieFitAlgorithm
from image_analysis.types import ImageAnalyzerResult

import logging

logger = logging.getLogger(__name__)


class BCaveMagSpecCam1Analyzer(BeamAnalyzer):
    """Analyzer for BCaveMagSpecCam1 images that reports bowtie-fit emittance metrics.

    Extends BeamAnalyzer with a bowtie fit of the dispersed electron trace.
    The standard processing pipeline from the config runs first; the fit is
    applied to the processed frame after a noise-floor threshold.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        *,
        spec: Optional[BCaveMagSpecCam1Spec] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize BCaveMagSpecCam1 analyzer with bowtie fit algorithm.

        Parameters
        ----------
        camera_config : CameraConfig
            Validated camera configuration model. (Use
            ``image_analysis.config.loader.load_camera_config("UC_BCaveMagSpecCam1")``
            to get the standard config.)
        spec : BCaveMagSpecCam1Spec, optional
            Bow-tie fit parameters (``n_beam_size_clearance``,
            ``min_total_counts``, ``threshold_factor``, ``count_threshold``);
            defaults when omitted.
        output_name : str, optional
            Output identifier; forwarded to ``BeamAnalyzer``.
        """
        super().__init__(camera_config, output_name=output_name)
        spec = spec or BCaveMagSpecCam1Spec()
        self.spec = spec

        # Initialize bowtie fit algorithm with the spec's parameters
        self.algo = BowtieFitAlgorithm(
            n_beam_size_clearance=spec.n_beam_size_clearance,
            min_total_counts=spec.min_total_counts,
            threshold_factor=spec.threshold_factor,
        )

        # Store parameters for potential inspection
        self.n_beam_size_clearance = spec.n_beam_size_clearance
        self.min_total_counts = spec.min_total_counts
        self.threshold_factor = spec.threshold_factor
        self.count_threshold = spec.count_threshold

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[dict] = None
    ) -> ImageAnalyzerResult:
        """Run preprocessing, evaluate the bowtie fit, and return ImageAnalyzerResult.

        Uses BeamAnalyzer's standard processing pipeline, then applies the
        configured noise-floor threshold and the bowtie fit.

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
        ImageAnalyzerResult
            Result containing processed image, beam stats, and bowtie fit results.
        """
        initial_result: ImageAnalyzerResult = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        # Suppress the residual noise floor left after background subtraction.
        final_image = initial_result.processed_image.copy()
        final_image[final_image < self.count_threshold] = 0

        # Run bowtie fit algorithm
        bowtie_result = self.algo.evaluate(final_image)

        # Add bowtie fit results to scalars. Unlike HiResMagCam, the fitted
        # parameters are reported too: they are the physical quantities the
        # emittance proxy is derived from. bowtie_x0/bowtie_y0 are the waist
        # location in pixels of the *processed* image, so any ROI crop in the
        # camera config is already applied.
        bowtie_scalars = {
            "emittance_proxy": bowtie_result.score,
            "total_counts": np.sum(final_image),
            "bowtie_w0": bowtie_result.w0,
            "bowtie_theta": bowtie_result.theta,
            "bowtie_x0": bowtie_result.x0,
            "bowtie_y0": bowtie_result.y0,
            "bowtie_r_squared": bowtie_result.r_squared,
        }

        # Merge with existing scalars from beam analysis
        combined_scalars = {**initial_result.scalars, **bowtie_scalars}

        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=initial_result.processed_image,
            scalars=combined_scalars,
            metadata=initial_result.metadata,
        )

        # Add render data: projections + bowtie weights
        if initial_result.processed_image is not None:
            result.render_data = {
                "horizontal_projection": initial_result.processed_image.sum(axis=0),
                "vertical_projection": initial_result.processed_image.sum(axis=1),
                "bowtie_weights": np.array(bowtie_result.weights),
                "bowtie_centers": np.array(bowtie_result.centers),
            }

        return result

    @staticmethod
    def render_image(
        result: ImageAnalyzerResult,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "plasma",
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render the image and overlay the bowtie fit weight lineout.

        Draws into ``ax`` when one is supplied — the ephemeral render seam
        depends on it (see ImageAnalysis CLAUDE.md, "Ephemeral runs").
        """
        from image_analysis.tools.rendering import (
            add_line_overlay,
        )

        # Base rendering
        fig, ax = base_render_image(
            result=result,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
        )

        # Overlay bowtie weight lineout if available
        bowtie_weights = result.render_data.get("bowtie_weights")
        img_height = result.processed_image.shape[0] - 1
        if bowtie_weights is not None:
            add_line_overlay(
                ax=ax,
                lineout=bowtie_weights,
                direction="horizontal",
                scale=0.3,
                offset=img_height,
                color="cyan",
                linewidth=1.0,
                normalize=True,
                label="Bowtie Fit Weights",
            )

        return fig, ax

    def visualize(
        self,
        results: ImageAnalyzerResult,
        *,
        show: bool = True,
        close: bool = True,
        ax: Optional[plt.Axes] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "plasma",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render a visualization of the analyzed image with bowtie fit overlay.

        This is a simple convenience wrapper that calls :meth:`render_image`
        and optionally shows or closes the figure.
        """
        fig, ax = self.render_image(
            result=results,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
        )

        if show:
            plt.show()
        if close:
            plt.close(fig)

        return fig, ax
