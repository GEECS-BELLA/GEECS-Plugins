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
    """Minimize beam size from FWHM: (x_fwhm * cal)^2 + (y_fwhm * cal)^2."""

    def __init__(self, calibration: float = 24.4e-3, **kwargs):
        super().__init__(**kwargs)
        self.calibration = calibration
        self.device_name = self.analyzer_configs[0].device_name
        self.objective_tag = "BeamSize"  # shows up as "Objective:BeamSize"
        # self.output_key = "f"              # already set in BaseEvaluator; keep default

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """Compute objective."""
        x_fwhm = self.get_scalar(self.device_name, "x_fwhm", scalar_results)
        y_fwhm = self.get_scalar(self.device_name, "y_fwhm", scalar_results)
        return (x_fwhm * self.calibration) ** 2 + (y_fwhm * self.calibration) ** 2

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> dict[str, float]:
        """Compute observables."""
        # Optional: more visibility in logs
        x = self.get_scalar(self.device_name, "x_fwhm", scalar_results)
        y = self.get_scalar(self.device_name, "y_fwhm", scalar_results)
        x_cal = x * self.calibration
        y_cal = y * self.calibration
        return {
            "x_fwhm_px": x,
            "y_fwhm_px": y,
            "x_fwhm_units": x_cal,
            "y_fwhm_units": y_cal,
            "size_quadrature_units2": x_cal**2 + y_cal**2,  # equals objective
        }
