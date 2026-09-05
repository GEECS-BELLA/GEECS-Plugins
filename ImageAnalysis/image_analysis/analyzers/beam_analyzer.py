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

# Import the StandardAnalyzer parent class
from image_analysis.analyzers.standard_analyzer import StandardAnalyzer
import image_analysis.config.array2d_processing as cfg_2d

# Import beam-specific tools
from image_analysis.algorithms.basic_beam_stats import (
    beam_profile_stats,
    flatten_beam_stats,
)
from image_analysis.algorithms.beam_slopes import compute_beam_slopes
from geecs_schemas.analysis import BeamAnalyzerSpec

from image_analysis.types import ImageAnalyzerResult

logger = logging.getLogger(__name__)


#: The typed parameters of :class:`BeamAnalyzer` — the ``beam`` analyzer spec
#: from GEECS-Schemas (``analyzer: {kind: beam, ...}`` in a diagnostic).
BeamAnalysisConfig = BeamAnalyzerSpec


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
    camera_config : CameraConfig
        Validated camera configuration model.
    output_name : str, optional
        Output identifier — forwarded to :class:`StandardAnalyzer`.
        See its docstring for the contract.
    """

    def __init__(
        self,
        camera_config: cfg_2d.CameraConfig,
        *,
        spec: Optional[BeamAnalyzerSpec] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the beam analyzer with a validated camera config.

        Parameters
        ----------
        camera_config : CameraConfig
            Pre-validated camera configuration. Use
            ``image_analysis.config.loader.load_camera_config(name)`` to
            load from disk by name.
        output_name : str, optional
            Output identifier; forwarded to ``StandardAnalyzer``.
        """
        # Initialize parent class
        super().__init__(camera_config=camera_config, output_name=output_name)

        # The analyzer's own parameters: the ``beam`` spec (defaults when
        # constructed directly in a notebook without one).
        self.analysis_config: BeamAnalyzerSpec = spec or BeamAnalyzerSpec()

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

        # Position stats are expressed in the full-image coordinate system by
        # passing the ROI origin offset so that CoM and peak_location reflect
        # the beam position on the sensor, not within the cropped sub-region.
        roi = self.camera_config.roi
        roi_offset = (roi.x_min, roi.y_min) if roi is not None else (0, 0)

        # Always: basic beam stats (projections along x, y, x_45, y_45).
        # Scalars are bare-keyed; ScanAnalysis namespaces them via
        # metric_prefix/metric_suffix when storing per-shot results (#412).
        beam_stats = beam_profile_stats(processed_image, roi_offset=roi_offset)
        enabled = self.analysis_config.enabled_stats
        scalars = flatten_beam_stats(
            beam_stats,
            include=set(enabled) if enabled is not None else None,
        )

        # Optional: slope/straightness metrics
        if self.analysis_config.compute_slopes:
            scalars.update(compute_beam_slopes(processed_image))

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
