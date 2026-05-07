"""
Example evaluator for maximizing total counts using MultiDeviceScanEvaluator.

Classes
-------
MaxCountsEvaluator
    Evaluator for maximizing beam total counts on any camera device.
"""

from __future__ import annotations

from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
    MultiDeviceScanEvaluator,
)


class EBeamSourceOpt(MultiDeviceScanEvaluator):
    """Maximize spectral density on BCaveMagSpec."""

    def __init__(self, calibration: float = 24.4e-3, **kwargs):
        super().__init__(**kwargs)
        self.calibration = calibration
        self.device_name = self.analyzer_configs[0].device_name
        self.objective_tag = "SpectralDensity"

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """Compute objective."""
        total = self.get_scalar(
            self.device_name, "U_BCaveMagSpec-interpSpec_objective", scalar_results
        )
        return -total
