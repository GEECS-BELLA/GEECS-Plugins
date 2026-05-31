"""Minimize beam FWHM on a single camera diagnostic.

Classes
-------
BeamSizeEvaluator
    Minimize ``(x_fwhm * cal)² + (y_fwhm * cal)²`` on whichever camera
    diagnostic is listed first in ``analyzers``.
"""

from __future__ import annotations

from geecs_scanner.optimization.base_evaluator import BaseEvaluator


class BeamSizeEvaluator(BaseEvaluator):
    """Minimize quadrature sum of FWHMs (in physical units)."""

    def __init__(self, calibration: float = 24.4e-3, **kwargs):
        super().__init__(**kwargs)
        self.calibration = calibration

    def compute_objective(self, scalars, bin_number):
        """Quadrature sum of calibrated FWHMs."""
        dev = self.primary_device
        x = scalars[f"{dev}_x_fwhm"]
        y = scalars[f"{dev}_y_fwhm"]
        return (x * self.calibration) ** 2 + (y * self.calibration) ** 2

    def compute_observables(self, scalars, bin_number):
        """Expose pixel + calibrated FWHMs alongside the objective."""
        dev = self.primary_device
        x = scalars[f"{dev}_x_fwhm"]
        y = scalars[f"{dev}_y_fwhm"]
        x_cal = x * self.calibration
        y_cal = y * self.calibration
        return {
            "x_fwhm_px": x,
            "y_fwhm_px": y,
            "x_fwhm_units": x_cal,
            "y_fwhm_units": y_cal,
            "size_quadrature_units2": x_cal**2 + y_cal**2,
        }
