"""
Example evaluator for maximizing total counts using MultiDeviceScanEvaluator.

Classes
-------
MaxCountsEvaluator
    Evaluator for maximizing beam total counts on any camera device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
    MultiDeviceScanEvaluator,
)


class MaxCountsEvaluator(MultiDeviceScanEvaluator):
    """Maximize total counts."""

    def __init__(self, calibration: float = 24.4e-3, **kwargs):
        super().__init__(**kwargs)
        self.calibration = calibration
        self.device_name = self.analyzer_configs[0].device_name
        self.objective_tag = "TotalCounts"  # shows up as "Objective:TotalCounts"

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """Compute objective."""
        total = self.get_scalar(self.device_name, "image_total", scalar_results)
        return -total

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> dict[str, float]:
        """Compute observables."""
        # Optional: more visibility in logs
        x_CoM = self.get_scalar(self.device_name, "x_CoM", scalar_results)
        y_CoM = self.get_scalar(self.device_name, "y_CoM", scalar_results)
        peak_value = self.get_scalar(
            self.device_name, "image_peak_value", scalar_results
        )

        return {"x_CoM": x_CoM, "y_CoM": y_CoM, "image_peak_value": peak_value}
