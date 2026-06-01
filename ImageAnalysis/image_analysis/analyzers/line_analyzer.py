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
from image_analysis.analyzers.standard_1d_analyzer import Standard1DAnalyzer
from image_analysis.config.array1d_processing import Line1DConfig

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
    line_config : Line1DConfig
        Validated line configuration model.
    """

    def __init__(self, line_config: Line1DConfig):
        """Initialize the line analyzer with a validated line config.

        Scalar-key prefix/suffix used to live here as ``metric_prefix`` /
        ``metric_suffix`` kwargs; they were promoted to the
        diagnostic-config layer per #412. ScanAnalysis now applies the
        prefix/suffix when storing per-shot results, and this analyzer
        emits bare scalar keys.
        """
        super().__init__(line_config)
        logger.info("Initialized LineAnalyzer for line: %s", self.line_config.name)

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

        # Bare-keyed scalars; ScanAnalysis namespaces them per #412.
        scalars = line_stats.to_dict()

        # Build result with line-specific data
        result = ImageAnalyzerResult(
            data_type="1d",
            line_data=processed_line_data,
            scalars=scalars,
            metadata=initial_result.metadata,
        )

        return result
