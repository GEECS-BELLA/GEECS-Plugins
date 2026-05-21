"""
Evaluator for minimizing beam size using MultiDeviceScanEvaluator.

Classes
-------
BeamSizeEvaluator
    Minimize beam FWHM on any camera device.
"""

from __future__ import annotations

from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
    MultiDeviceScanEvaluator,
)


class BeamSizeEvaluator(MultiDeviceScanEvaluator):
    """Minimize beam size: (x_fwhm * calibration)² + (y_fwhm * calibration)²."""

    def __init__(self, calibration: float = 24.4e-3, **kwargs):
        super().__init__(**kwargs)
        self.calibration = calibration
        self.objective_tag = "BeamSize"

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """Compute objective."""
        x = self.get_scalar(self.primary_device, "x_fwhm", scalar_results)
        y = self.get_scalar(self.primary_device, "y_fwhm", scalar_results)
        return (x * self.calibration) ** 2 + (y * self.calibration) ** 2

    def compute_observables(self, scalar_results: dict, bin_number: int) -> dict:
        """Compute observables."""
        x = self.get_scalar(self.primary_device, "x_fwhm", scalar_results)
        y = self.get_scalar(self.primary_device, "y_fwhm", scalar_results)
        x_cal = x * self.calibration
        y_cal = y * self.calibration
        return {
            "x_fwhm_px": x,
            "y_fwhm_px": y,
            "x_fwhm_units": x_cal,
            "y_fwhm_units": y_cal,
            "size_quadrature_units2": x_cal**2 + y_cal**2,
        }
