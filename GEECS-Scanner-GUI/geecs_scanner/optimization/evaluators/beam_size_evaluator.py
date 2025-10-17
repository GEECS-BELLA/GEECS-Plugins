"""
Example evaluator for minimizing beam size using MultiDeviceScanEvaluator.

Classes
-------
BeamSizeEvaluator
    Evaluator for minimizing beam FWHM on any camera device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
    MultiDeviceScanEvaluator,
)


class BeamSizeEvaluator(MultiDeviceScanEvaluator):
    """
    Evaluator for minimizing beam size based on FWHM measurements.

    Parameters
    ----------
    calibration : float, default=24.4e-3
        Spatial calibration factor in mm/pixel (or desired units/pixel).
    **kwargs
        Additional keyword arguments passed to MultiDeviceScanEvaluator,
        including 'analyzers', 'scan_data_manager', and 'data_logger'.
    """

    def __init__(self, calibration: float = 24.4e-3, **kwargs):
        """Initialize the beam size evaluator."""
        super().__init__(**kwargs)
        self.calibration = calibration
        # Get device name from first analyzer config
        self.device_name = self.analyzer_configs[0].device_name

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """
        Compute beam size objective from FWHM measurements.

        Parameters
        ----------
        scalar_results : dict
            Dictionary of scalar results from all analyzers.
            Expected to contain x_fwhm and y_fwhm metrics.
        bin_number : int
            Current bin number being evaluated (not used in this implementation).

        Returns
        -------
        float
            Beam size objective value: (x_fwhm * cal)² + (y_fwhm * cal)²
        """
        # Use get_scalar() helper for flexible key lookup
        x_fwhm = self.get_scalar(self.device_name, "x_fwhm", scalar_results)
        y_fwhm = self.get_scalar(self.device_name, "y_fwhm", scalar_results)

        # Compute sum of squares with calibration
        objective = (x_fwhm * self.calibration) ** 2 + (y_fwhm * self.calibration) ** 2

        return objective
