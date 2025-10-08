"""Beam Profile Analyzer using the StandardAnalyzer framework.

This module provides a specialized analyzer for beam profile analysis that inherits
from StandardAnalyzer. It adds beam-specific capabilities:
- Beam statistics calculation (centroid, width, height, FWHM)
- Gaussian fitting parameters
- Beam quality metrics
- Specialized beam rendering with overlays
- Lineout generation and analysis

The BeamAnalyzer focuses purely on beam-specific analysis while leveraging
the StandardAnalyzer for all image processing pipeline functionality.
"""

from __future__ import annotations

import logging
from typing import Optional, Union, List, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Import the StandardAnalyzer parent class
from image_analysis.offline_analyzers.standard_analyzer import StandardAnalyzer

# Import beam-specific tools
from image_analysis.tools.rendering import base_render_image
from image_analysis.algorithms.basic_beam_stats import (
    beam_profile_stats,
    flatten_beam_stats,
)
from image_analysis.types import AnalyzerResultDict
from image_analysis.processing.array2d.config_models import (
    BackgroundConfig,
    BackgroundMethod,
    DynamicBackgroundConfig as DynamicComputationConfig,
)

logger = logging.getLogger(__name__)


def create_variation_analyzer(
    camera_config_name: str,
    percentile: float = 50.0,
    method: BackgroundMethod = BackgroundMethod.PERCENTILE_DATASET,
    additional_constant: float = 0.0,
    name_suffix: str = "_variation",
) -> "BeamAnalyzer":
    """Create a BeamAnalyzer configured for variation analysis.

    This is a convenience function that configures dynamic background subtraction
    for analyzing shot-to-shot beam fluctuations on stable beams. The dynamic
    background is computed from the dataset and subtracted from each image,
    revealing the parts of the beam that fluctuate from shot to shot.

    Parameters
    ----------
    camera_config_name : str
        Name of the camera configuration to load (e.g., "UC_ALineEBeam3")
    percentile : float, default=50.0
        Percentile for background computation (only used if method is PERCENTILE_DATASET).
        For stable beams, 50.0 (median) is typically a good choice.
    method : BackgroundMethod, default=PERCENTILE_DATASET
        Background computation method. Options include:
        - PERCENTILE_DATASET: Use a percentile of the dataset
        - MEDIAN: Use the median of the dataset
        - CONSTANT: Use a constant value
    name_suffix : str, default="_variation"
        Suffix to append to camera name for scalar result prefixes.
        This ensures variation analysis results are clearly distinguished
        from standard analysis results in the output data.
    additional_constant : float, default = 0.0
        additional background offset after dynamic bkg subtraction

    Returns
    -------
    BeamAnalyzer
        Configured analyzer for variation analysis with dynamic background enabled

    Notes
    -----
    The dynamic background is computed during batch processing and saved to
    ``{scan_dir}/computed_background.npy`` for inspection and reuse.
    """
    variation_bg_config = BackgroundConfig(
        enabled=True,
        method=BackgroundMethod.FROM_FILE,
        file_path=Path("{scan_dir}/computed_background.npy"),
        constant_level=0,
        dynamic_computation=DynamicComputationConfig(
            enabled=True,
            method=method,
            percentile=percentile,
            auto_save_path=Path("{scan_dir}/computed_background.npy"),
        ),
        additional_constant=additional_constant,
    )

    return BeamAnalyzer(
        camera_config_name=camera_config_name,
        config_overrides={"background": variation_bg_config},
        name_suffix=name_suffix,
    )


class BeamAnalyzer(StandardAnalyzer):
    """
    Beam profile analyzer using the StandardAnalyzer framework.

    This analyzer specializes the StandardAnalyzer for beam profile analysis by adding:
    - Beam statistics calculation (centroid, width, height, FWHM)
    - Gaussian fitting parameters
    - Beam quality metrics
    - Specialized beam rendering with overlays
    - Lineout generation and analysis

    All image processing pipeline functionality is inherited from StandardAnalyzer,
    making this class focused purely on beam-specific analysis.

    Parameters
    ----------
    camera_config_name : str
        Name of the camera configuration to load (e.g., "undulator_exit_cam")
    config_overrides : dict, optional
        Runtime overrides for configuration parameters
    """

    def __init__(
        self,
        camera_config_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        name_suffix: Optional[str] = None,
    ):
        """Initialize the beam analyzer with external configuration.

        Parameters
        ----------
        camera_config_name : str
            Name of the camera configuration to load (e.g., "UC_ALineEBeam3")
        config_overrides : dict, optional
            Runtime overrides for configuration parameters. Can contain Pydantic
            model instances or dictionaries for any configuration section.
        name_suffix : str, optional
            Suffix to append to camera name for scalar result prefixes.
            Useful for distinguishing multiple analysis passes on the same camera.
            For example, use "_variation" to distinguish variation analysis results
            from standard analysis results.
        """
        # Initialize parent class
        super().__init__(camera_config_name, config_overrides)

        # Apply name suffix if provided
        if name_suffix:
            self.camera_config.name = f"{self.camera_config.name}{name_suffix}"
            logger.info(f"Camera name set to: {self.camera_config.name}")

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> AnalyzerResultDict:
        """
        Run complete beam analysis using the processing pipeline.

        This method extends the StandardAnalyzer's analyze_image method to add
        beam-specific analysis including statistics calculation and lineouts.

        Parameters
        ----------
        image : np.ndarray
            Input image to analyze
        auxiliary_data : dict, optional
            Additional data including file path and preprocessing flags

        Returns
        -------
        AnalyzerResultDict
            Dictionary containing processed image, beam statistics, and lineouts
        """
        initial_result: AnalyzerResultDict = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        processed_image = initial_result["processed_image"]

        # Compute beam statistics
        beam_stats_flat = flatten_beam_stats(
            beam_profile_stats(processed_image),
            prefix=self.camera_config.name,
        )

        # Compute lineouts
        horiz_lineout = processed_image.sum(axis=0)
        vert_lineout = processed_image.sum(axis=1)

        return_dict = self.build_return_dictionary(
            return_image=processed_image,
            input_parameters=initial_result["analyzer_input_parameters"],
            return_scalars=beam_stats_flat,
            return_lineouts=[horiz_lineout, vert_lineout],
            coerce_lineout_length=False,
        )

        return return_dict

    @staticmethod
    def render_image(
        image: np.ndarray,
        analysis_results_dict: Optional[Dict[str, Union[float, int]]] = None,
        input_params_dict: Optional[Dict[str, Union[float, int]]] = None,
        lineouts: Optional[List[np.ndarray]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        figsize: Tuple[float, float] = (4, 4),
        cmap: str = "plasma",
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
        fixed_width_in: float = 4.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render beam image with beam-specific overlays.

        This method provides specialized rendering for beam analysis including
        beam centroid markers and lineout overlays.

        Parameters
        ----------
        image : np.ndarray
            Beam image to render
        analysis_results_dict : dict, optional
            Dictionary containing beam analysis results
        input_params_dict : dict, optional
            Dictionary containing input parameters and ROI info
        lineouts : list of np.ndarray, optional
            List containing [horizontal_lineout, vertical_lineout]
        vmin, vmax : float, optional
            Color scale limits
        figsize : tuple of float, default ``(4, 4)``
            Size of the created figure in inches (width, height). Ignored when an
            existing ``ax`` is supplied.
        cmap : str, default="plasma"
            Colormap name
        dpi : int, default=150
            Figure DPI
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        fixed_width_in : float, default=4.0
            Fixed width for figure sizing

        Returns
        -------
        tuple
            (figure, axes) matplotlib objects
        """
        h, w = image.shape[:2]
        height_in = max(1e-6, fixed_width_in * (h / float(w)))
        computed_figsize = (fixed_width_in, height_in)

        fig, ax = base_render_image(
            image=image,
            analysis_results_dict=analysis_results_dict,
            input_params_dict=input_params_dict,
            lineouts=lineouts,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=computed_figsize,
            dpi=dpi,
            ax=ax,
        )

        # Add beam centroid if available in analysis results
        if analysis_results_dict:
            # Look for centroid information with camera name prefix
            centroid_x_key = None
            centroid_y_key = None

            # Find centroid keys (they may have camera name prefix)
            for key in analysis_results_dict.keys():
                if key.endswith("_cent_x") or key == "cent_x":
                    centroid_x_key = key
                elif key.endswith("_cent_y") or key == "cent_y":
                    centroid_y_key = key

            # Also check for legacy format
            if (
                "blue_cent_x" in analysis_results_dict
                and "blue_cent_y" in analysis_results_dict
            ):
                centroid_x_key = "blue_cent_x"
                centroid_y_key = "blue_cent_y"

            # Plot centroid if found
            if centroid_x_key and centroid_y_key:
                cx = analysis_results_dict[centroid_x_key]
                cy = analysis_results_dict[centroid_y_key]

                # Adjust for ROI if present
                if input_params_dict:
                    cx -= input_params_dict.get("left_ROI", 0)
                    cy -= input_params_dict.get("top_ROI", 0)

                ax.plot(cx, cy, "bo", markersize=5, label="Beam Centroid")

        # Add lineouts overlay
        if lineouts and len(lineouts) == 2:
            horiz, vert = np.clip(lineouts[0], 0, None), np.clip(lineouts[1], 0, None)
            img_h, img_w = image.shape

            if len(horiz) > 0 and len(vert) > 0:
                # Normalize lineouts for overlay
                horiz_norm = horiz / np.max(horiz) * img_h * 0.2
                vert_norm = vert / np.max(vert) * img_w * 0.2

                # Plot lineouts
                ax.plot(
                    np.arange(len(horiz)),
                    img_h - horiz_norm,
                    color="cyan",
                    lw=1.0,
                    label="Horizontal Lineout",
                )
                ax.plot(
                    vert_norm,
                    np.arange(len(vert)),
                    color="magenta",
                    lw=1.0,
                    label="Vertical Lineout",
                )

        return fig, ax

    def visualize(
        self,
        results: AnalyzerResultDict,
        *,
        show: bool = True,
        close: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Render a single visualization of the analyzed (post-ROI) image and overlays.

        This convenience method draws the processed image stored in
        ``results["processed_image"]`` using :meth:`render_image`, optionally
        calling :func:`matplotlib.pyplot.show` and closing the figure. It performs
        no additional computation.

        Parameters
        ----------
        results : AnalyzerResultDict
            Dictionary returned by :meth:`analyze_image` (or compatible). The
            following keys are read when present:
            - ``"processed_image"`` : ndarray (required) — image to display
            - ``"analyzer_return_dictionary"`` : dict — scalar annotations
            - ``"input_parameters"`` : dict — ROI offsets and inputs
            - ``"analyzer_return_lineouts"`` : list[np.ndarray] — [horiz, vert] lineouts
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
            image=results["processed_image"],
            analysis_results_dict=results.get("analyzer_return_dictionary", {}),
            input_params_dict=results.get("analyzer_input_parameters", {}),
            lineouts=results.get("analyzer_return_lineouts"),
            ax=ax,
        )
        if show:
            plt.show()
        if close:
            plt.close(fig)
        return fig, ax
