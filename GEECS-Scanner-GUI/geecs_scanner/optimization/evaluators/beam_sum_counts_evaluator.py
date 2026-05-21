"""
Evaluator for maximizing total counts using MultiDeviceScanEvaluator.

Classes
-------
MaxCountsEvaluator
    Maximize beam total counts on any camera device.
"""

from __future__ import annotations

from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
    MultiDeviceScanEvaluator,
)


class MaxCountsEvaluator(MultiDeviceScanEvaluator):
    """Maximize total counts (returns negative for minimization)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objective_tag = "TotalCounts"

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """Compute objective."""
        return -self.get_scalar(self.primary_device, "image_total", scalar_results)

    def compute_observables(self, scalar_results: dict, bin_number: int) -> dict:
        """Compute observables."""
        return {
            "x_CoM": self.get_scalar(self.primary_device, "x_CoM", scalar_results),
            "y_CoM": self.get_scalar(self.primary_device, "y_CoM", scalar_results),
            "image_peak_value": self.get_scalar(
                self.primary_device, "image_peak_value", scalar_results
            ),
        }
