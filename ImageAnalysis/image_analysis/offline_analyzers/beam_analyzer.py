"""Beam Profile Analyzer using the StandardAnalyzer framework.

This module provides a specialized analyzer for beam profile analysis that inherits
from StandardAnalyzer. It adds beam-specific capabilities:
- Beam statistics calculation (centroid, width, height, FWHM)
- Optional slope/straightness metrics
- Specialized beam rendering with overlays
- Lineout generation and analysis

The BeamAnalyzer focuses purely on beam-specific analysis while leveraging
the StandardAnalyzer for all image processing pipeline functionality.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

# Import the StandardAnalyzer parent class
from image_analysis.offline_analyzers.standard_analyzer import StandardAnalyzer

# Import beam-specific tools
from image_analysis.algorithms.basic_beam_stats import (
    beam_profile_stats,
    flatten_beam_stats,
)
from image_analysis.algorithms.beam_slopes import compute_beam_slopes
from image_analysis.types import ImageAnalyzerResult

logger = logging.getLogger(__name__)


class BeamAnalysisConfig(BaseModel):
    """Typed configuration for :class:`BeamAnalyzer`.

    This model is validated from ``camera_config.analysis`` at analyzer init
    time, giving users IDE autocompletion and config-file validation.

    Attributes
    ----------
    compute_slopes : bool
        Whether to compute beam slope (straightness) metrics.
    """

    compute_slopes: bool = Field(
        default=False,
        description=(
            "Whether to compute beam slope (straightness) metrics. "
            "Expensive: involves line-by-line stats + weighted linear fits."
        ),
    )


class BeamAnalyzer(StandardAnalyzer):
    """Beam profile analyzer using the StandardAnalyzer framework.

    This analyzer specializes the StandardAnalyzer for beam profile analysis by
    composing algorithm calls:

    - **Always**: basic beam stats (projections along x, y, x_45, y_45)
    - **Optional**: slope/straightness metrics (via ``compute_slopes`` config flag)

    All image processing pipeline functionality is inherited from StandardAnalyzer,
    making this class focused purely on beam-specific analysis.

    Parameters
    ----------
    camera_config_name : str
        Name of the camera configuration to load (e.g., "undulator_exit_cam")
    """

    def __init__(
        self,
        camera_config_name: str,
        name_suffix: Optional[str] = None,
        metric_suffix: Optional[str] = None,
    ):
        """Initialize the beam analyzer with external configuration.

        Parameters
        ----------
        camera_config_name : str
            Name of the camera configuration to load (e.g., "UC_ALineEBeam3")
        name_suffix : str, optional
            Suffix to append to camera name for scalar result prefixes.
            Useful for distinguishing multiple analysis passes on the same camera.
            For example, use "_variation" to distinguish variation analysis results
            from standard analysis results.
        metric_suffix : str, optional
            Suffix to append to all metric names (underscore is auto-prepended).
            For example, "curtis" becomes "_curtis" in the output keys.
            Useful for tracking different analysis variations while keeping the
            same camera name. (e.g., "camera_x_rms_curtis").
        """
        # Initialize parent class
        super().__init__(camera_config_name)

        # Validate analysis config (if present) into a typed model
        self.analysis_config = BeamAnalysisConfig.model_validate(
            self.camera_config.analysis or {}
        )

        # Store metric suffix for use in analyze_image
        self.metric_suffix = metric_suffix

        # Apply name suffix if provided
        if name_suffix:
            self.camera_config.name = f"{self.camera_config.name}{name_suffix}"
            logger.info(f"Camera name set to: {self.camera_config.name}")

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """Run complete beam analysis using the processing pipeline.

        This method extends the StandardAnalyzer's analyze_image method to add
        beam-specific analysis.  It always computes basic beam stats (projections)
        and optionally computes slope metrics when configured.

        Parameters
        ----------
        image : np.ndarray
            Input image to analyze
        auxiliary_data : dict, optional
            Additional data including file path and preprocessing flags

        Returns
        -------
        ImageAnalyzerResult
            Structured result containing processed image, beam statistics, and metadata
        """
        initial_result: ImageAnalyzerResult = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        processed_image = initial_result.processed_image

        # Always: basic beam stats (projections along x, y, x_45, y_45)
        beam_stats = beam_profile_stats(processed_image)
        scalars = flatten_beam_stats(
            beam_stats,
            prefix=self.camera_config.name,
            suffix=self.metric_suffix,
        )

        # Optional: slope/straightness metrics
        if self.analysis_config.compute_slopes:
            slope_scalars = compute_beam_slopes(
                processed_image,
                prefix=self.camera_config.name,
                suffix=self.metric_suffix,
            )
            scalars.update(slope_scalars)

        # Build result with beam-specific data
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=processed_image,
            scalars=scalars,
            metadata=initial_result.metadata,
        )

        # Add projection overlays for rendering
        if processed_image is not None:
            result.render_data = {
                "horizontal_projection": processed_image.sum(axis=0),
                "vertical_projection": processed_image.sum(axis=1),
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
        """Render beam image with beam-specific overlays.

        This method provides specialized rendering for beam analysis including
        XY projection lineouts and beam centroid markers using composable
        overlay functions.
        """
        from image_analysis.tools.rendering import (
            base_render_image,
            add_xy_projections,
            add_marker,
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

        # Add beam-specific overlays
        add_xy_projections(ax, result)
        add_marker(ax, (100, 100), size=1, color="blue")

        return fig, ax
