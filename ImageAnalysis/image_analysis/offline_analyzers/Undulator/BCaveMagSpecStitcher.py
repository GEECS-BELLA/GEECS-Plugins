"""Analyzer for BCaveMagSpec stitched images with Gaussian-weighted optimization target.

This module defines `BCaveMagSpecStitcherAnalyzer`, a subclass of `StandardAnalyzer`,
to process stitched spectrometer images from BCaveMagSpec. It uses the standard
StandardAnalyzer processing pipeline (background subtraction, ROI cropping, optional
crosshair masking), then computes a vertical sum lineout, applies Gaussian weighting,
and returns the weighted sum as an optimization target.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from image_analysis.tools.rendering import base_render_image, add_line_overlay
from image_analysis.offline_analyzers.standard_analyzer import StandardAnalyzer
from image_analysis.types import ImageAnalyzerResult

import logging

logger = logging.getLogger(__name__)


class BCaveMagSpecStitcherAnalyzer(StandardAnalyzer):
    """Image analyzer for BCaveMagSpec stitched spectrometer images with Gaussian weighting.

    This analyzer extends StandardAnalyzer to add custom Gaussian-weighted optimization
    for the BCave magnetic spectrometer. It uses the standard processing pipeline
    from the config file, then applies Gaussian weighting to the vertical lineout
    for optimization purposes.
    """

    def __init__(
        self,
        camera_config_name: str = "U_BCaveMagSpec",
        config_overrides: Optional[Dict[str, Any]] = None,
        gaussian_sigma: float = 20.0,
        gaussian_center: float = 250.0,
    ):
        """Initialize BCaveMagSpecStitcher analyzer.

        Parameters
        ----------
        camera_config_name : str, default="U_BCaveMagSpec"
            Name of the camera configuration to load
        config_overrides : dict, optional
            Runtime overrides for configuration parameters
        gaussian_sigma : float, default=20.0
            Standard deviation of Gaussian weighting function
        gaussian_center : float, default=250.0
            Center position of Gaussian weighting function (in pixels)
        """
        super().__init__(camera_config_name, config_overrides)
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_center = gaussian_center

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[dict] = None
    ) -> ImageAnalyzerResult:
        """Analyze BCaveMagSpec stitched image and compute Gaussian-weighted lineout sum.

        This method uses StandardAnalyzer's standard processing pipeline, then adds
        custom Gaussian weighting to compute an optimization target.

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
        ImageAnalyzerResult
            Structured result containing:
            - processed_image: Final processed image
            - scalars: Dict with 'optimization_target' (Gaussian-weighted sum)
            - metadata: Analysis configuration and parameters
            - render_data: Dict with 'weighted_lineout' and 'vertical_lineout'
        """
        initial_result: ImageAnalyzerResult = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        # Compute vertical lineout (sum along vertical axis)
        horizontal_lineout = np.sum(initial_result.processed_image, axis=0)
        vertical_lineout = np.sum(initial_result.processed_image, axis=1)

        # Apply Gaussian weighting
        x = np.arange(horizontal_lineout.shape[0])
        gaussian = np.exp(
            -0.5 * ((x - self.gaussian_center) / self.gaussian_sigma) ** 2
        )
        weighted_lineout = horizontal_lineout * gaussian

        # Compute optimization target
        optimization_target = float(np.sum(weighted_lineout))

        # Build the result using ImageAnalyzerResult
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=initial_result.processed_image,
            scalars={"optimization_target": optimization_target},
            metadata={
                **initial_result.metadata,
                "gaussian_sigma": self.gaussian_sigma,
                "gaussian_center": self.gaussian_center,
            },
            render_data={
                "weighted_lineout": weighted_lineout,
                "vertical_lineout": vertical_lineout,
            },
            render_function=self.render_image,
        )

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
        """Render image with optional overlaid normalized lineout.

        Parameters
        ----------
        result : ImageAnalyzerResult
            The analysis result containing processed image and render data.
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
            result=result,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
        )

        # Overlay the weighted lineout on the image
        weighted_lineout = result.render_data.get("weighted_lineout")
        if weighted_lineout is not None and len(weighted_lineout) > 0:
            # Get image dimensions for proper scaling
            img_height = result.processed_image.shape[0]

            # Add the weighted lineout overlay at the top of the image
            add_line_overlay(
                ax=ax,
                lineout=weighted_lineout,
                direction="horizontal",
                scale=0.3,
                offset=img_height * 0.1,  # Slight offset from top
                color="cyan",
                linewidth=1.0,
                normalize=True,
                clip_positive=True,
            )

        return fig, ax

    def visualize(
        self,
        result: ImageAnalyzerResult,
        *,
        show: bool = True,
        close: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render a single visualization of the analyzed (post-ROI) image and overlays.

        This convenience method draws the processed image stored in
        ``result.processed_image`` using :meth:`render_image`, optionally
        calling :func:`matplotlib.pyplot.show` and closing the figure. It performs
        no additional computation.

        Parameters
        ----------
        result : ImageAnalyzerResult
            Result returned by :meth:`analyze_image` containing:
            - ``processed_image`` : ndarray — image to display
            - ``scalars`` : dict — scalar analysis results
            - ``metadata`` : dict — ROI offsets and input parameters
            - ``render_data`` : dict — lineouts for overlay
        show : bool, default True
            If True, call :func:`matplotlib.pyplot.show` after rendering.
        close : bool, default True
            If True, close the figure after showing (if ``show=True``) or
            immediately after rendering (if ``show=False``). Set to ``False`` to
            keep the figure open for further customization.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw into. If omitted, a new figure and axes are
            created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure that contains the rendering.
        ax : matplotlib.axes.Axes
            The axes on which the image was drawn.

        Notes
        -----
        * Lineouts and scalar overlays are assumed to correspond to the **post-ROI**
          image. If they were computed from the pre-ROI image, Matplotlib may try
          to autoscale the view unless limits are clamped in :meth:`render_image`.
        * For frameless saved images, consider:
          ``fig.savefig(path, bbox_inches='tight', pad_inches=0)``.
        """
        fig, ax = self.render_image(
            result=result,
            ax=ax,
        )
        if show:
            plt.show()
        if close:
            plt.close(fig)
        return fig, ax


if __name__ == "__main__":
    # Example usage
    from geecs_data_utils.config_roots import image_analysis_config

    current_dir = Path(__file__).resolve().parent.parent

    geecs_plugins_dir = current_dir.parent.parent.parent
    image_analysis_config.set_base_dir(geecs_plugins_dir / "image_analysis_configs")

    image_analyzer = BCaveMagSpecStitcherAnalyzer(
        camera_config_name="U_BCaveMagSpec", gaussian_sigma=20.0, gaussian_center=250.0
    )

    # Example file path (update to actual path)
    file_path = Path(
        "Z:/data/Undulator/Y2025/06-Jun/25_0605/scans/Scan018/U_BCaveMagSpec/Scan018_U_BCaveMagSpec_001.png"
    )

    if file_path.exists():
        result: ImageAnalyzerResult = image_analyzer.analyze_image_file(
            image_filepath=file_path
        )
        image_analyzer.visualize(result)
        print(f"Optimization target: {result.scalars['optimization_target']:.2f}")
    else:
        print(f"Test file not found: {file_path}")
        print("Create a test with your own image file.")
