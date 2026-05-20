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
from typing import Optional, Dict, Union


# Import the Standard1DAnalyzer parent class
from image_analysis.offline_analyzers.standard_1d_analyzer import Standard1DAnalyzer
from image_analysis.processing.array1d.config_models import Line1DConfig

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
    metric_prefix : str, optional
        Override the prefix used for scalar metric keys. Defaults to
        ``line_config.name``. Useful when one ``Line1DConfig`` is reused by
        multiple analyzer instances that should report under different names
        (e.g., a stitcher that loads from a per-device config but emits
        metrics under a composite name).
    """

    def __init__(
        self,
        line_config_name: Union[str, Line1DConfig],
        metric_suffix: Optional[str] = None,
        metric_prefix: Optional[str] = None,
    ):
        """Initialize the line analyzer with external configuration.

        Parameters
        ----------
        line_config_name : str or Line1DConfig
            Name of the line configuration to load (e.g., "spectrometer_config"),
            or a pre-constructed ``Line1DConfig`` instance.
        metric_suffix : str, optional
            Suffix to append to all metric names (underscore is auto-prepended).
            For example, "calibrated" becomes "_calibrated" in the output keys.
            Useful for distinguishing multiple analysis passes on the same line.
        metric_prefix : str, optional
            Override the prefix used for scalar metric keys. Defaults to
            ``line_config.name``.
        """
        # Initialize parent class
        super().__init__(line_config_name)

        # Store metric suffix/prefix for use in analyze_image
        self.metric_suffix = metric_suffix
        self.metric_prefix = metric_prefix

        logger.info(
            "Initialized LineAnalyzer for line: %s%s%s",
            self.line_config.name,
            f" (prefix: {metric_prefix})" if metric_prefix else "",
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
        prefix = self.metric_prefix or self.line_config.name
        scalars = line_stats.to_dict(prefix=prefix, suffix=self.metric_suffix)

        # Build result with line-specific data
        result = ImageAnalyzerResult(
            data_type="1d",
            line_data=processed_line_data,
            scalars=scalars,
            metadata=initial_result.metadata,
        )

        return result
