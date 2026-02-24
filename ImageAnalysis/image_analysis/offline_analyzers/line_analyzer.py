"""Line Profile Analyzer using the Standard1DAnalyzer framework.

This module provides a specialized analyzer for 1D line profile analysis that inherits
from Standard1DAnalyzer. It adds line-specific capabilities:
- Line statistics calculation (center of mass, FWHM, RMS width)
- Peak analysis (amplitude, position)
- Integrated signal calculation
- Unit-aware metric reporting

The LineAnalyzer focuses purely on line-specific analysis while leveraging
the Standard1DAnalyzer for all 1D data processing pipeline functionality.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict


# Import the Standard1DAnalyzer parent class
from image_analysis.offline_analyzers.standard_1d_analyzer import Standard1DAnalyzer

# Import line-specific tools
from image_analysis.algorithms.basic_line_stats import LineBasicStats
from image_analysis.types import Array1D, ImageAnalyzerResult

logger = logging.getLogger(__name__)


class LineAnalyzer(Standard1DAnalyzer):
    """
    Line profile analyzer using the Standard1DAnalyzer framework.

    This analyzer specializes the Standard1DAnalyzer for 1D line profile analysis by adding:
    - Line statistics calculation (CoM, FWHM, RMS, etc.)
    - Peak analysis
    - Integrated signal
    - Unit-aware metric reporting

    All 1D data processing pipeline functionality is inherited from Standard1DAnalyzer,
    making this class focused purely on line-specific analysis.

    Parameters
    ----------
    line_config_name : str
        Name of the line configuration to load (e.g., "spectrometer_config")
    metric_suffix : str, optional
        Suffix to append to all metric names (underscore is auto-prepended).
        For example, "calibrated" becomes "_calibrated" in the output keys.
        Useful for tracking different analysis variations while keeping the
        same line name (e.g., "spectrum_CoM_calibrated").
    """

    def __init__(
        self,
        line_config_name: str,
        metric_suffix: Optional[str] = None,
    ):
        """Initialize the line analyzer with external configuration.

        Parameters
        ----------
        line_config_name : str
            Name of the line configuration to load (e.g., "spectrometer_config")
        metric_suffix : str, optional
            Suffix to append to all metric names (underscore is auto-prepended).
            For example, "calibrated" becomes "_calibrated" in the output keys.
            Useful for distinguishing multiple analysis passes on the same line.
        """
        # Initialize parent class
        super().__init__(line_config_name)

        # Store metric suffix for use in analyze_image
        self.metric_suffix = metric_suffix

        logger.info(
            "Initialized LineAnalyzer for line: %s%s",
            self.line_config.name,
            f" (suffix: {metric_suffix})" if metric_suffix else "",
        )

    def analyze_image(
        self, image: Array1D, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """
        Run complete line analysis using the processing pipeline.

        This method extends the Standard1DAnalyzer's analyze_image method to add
        line-specific analysis including statistics calculation.

        Parameters
        ----------
        image : Array1D
            Input 1D data to analyze (Nx2 array: x values, y values)
        auxiliary_data : dict, optional
            Additional data including file path and preprocessing flags

        Returns
        -------
        ImageAnalyzerResult
            Structured result containing processed line data, statistics, and metadata
        """
        # Call parent to get processed line_data and metadata (with resolved units!)
        initial_result: ImageAnalyzerResult = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        processed_line_data = initial_result.line_data

        # Compute line statistics with units from metadata
        line_stats = LineBasicStats(
            line_data=processed_line_data,
            x_units=initial_result.metadata.get("x_units"),
            y_units=initial_result.metadata.get("y_units"),
        )

        # Generate scalars dict with prefix and optional suffix
        scalars = line_stats.to_dict(
            prefix=self.line_config.name, suffix=self.metric_suffix
        )

        # Build result with line-specific data
        result = ImageAnalyzerResult(
            data_type="1d",
            line_data=processed_line_data,
            scalars=scalars,
            metadata=initial_result.metadata,
        )

        return result
