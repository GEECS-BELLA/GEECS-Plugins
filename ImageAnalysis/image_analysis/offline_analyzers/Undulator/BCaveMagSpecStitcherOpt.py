"""Optimization definition for bcavemagspec."""

from __future__ import annotations

import logging
from typing import Optional, Dict
import numpy as np


# Import the Standard1DAnalyzer parent class
from image_analysis.offline_analyzers.standard_1d_analyzer import Standard1DAnalyzer

# Import line-specific tools
from image_analysis.types import Array1D, ImageAnalyzerResult

logger = logging.getLogger(__name__)


def objective(spectrum, E0=100.0, sigma_gate=5.0):
    """Objective function for BCaveMagSpec optimization."""
    E = spectrum[:, 0]
    q = spectrum[:, 1]

    dE = np.gradient(E)
    w = q * dE

    if np.sum(w) < 1e-12:
        return 0.0

    # --- HARD anchor to target energy ---
    gate = np.exp(-((E - E0) ** 2) / (2 * sigma_gate**2))
    w_local = w * gate

    if np.sum(w_local) < 1e-12:
        return 0.0

    # --- weighted median + MAD on gated distribution ---
    def weighted_median(x, w):
        idx = np.argsort(x)
        x, w = x[idx], w[idx]
        cdf = np.cumsum(w) / np.sum(w)
        return np.interp(0.5, cdf, x)

    def weighted_mad(x, w):
        med = weighted_median(x, w)
        mad = weighted_median(np.abs(x - med), w)
        return med, mad

    E_med, E_mad = weighted_mad(E, w_local)

    delta_E = 3.5 * E_mad

    window = (E > E_med - delta_E) & (E < E_med + delta_E)

    Q_window = np.sum(w_local[window])

    # --- spectral density (what you care about) ---
    f_spectral = Q_window / (delta_E + 1e-12)

    # --- spread penalty (local only) ---
    spread_penalty = E_mad / (E0 + 1e-12)

    # --- final objective ---
    J = f_spectral / (1 + spread_penalty)

    return J


class BCaveMagOpt(Standard1DAnalyzer):
    """Line profile analyzer using the Standard1DAnalyzer framework."""

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

        obj = objective(processed_line_data)
        scalars = {f"{self.line_config.name}_objective": obj}

        # Build result with line-specific data
        result = ImageAnalyzerResult(
            data_type="1d",
            line_data=processed_line_data,
            scalars=scalars,
            metadata=initial_result.metadata,
        )

        return result
