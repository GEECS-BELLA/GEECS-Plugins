"""Maximize total image counts on a single camera diagnostic.

Classes
-------
MaxCountsEvaluator
    Maximize ``image_total`` (returns negative for Xopt minimization
    semantics).
"""

from __future__ import annotations

from geecs_scanner.optimization.base_evaluator import BaseEvaluator


class MaxCountsEvaluator(BaseEvaluator):
    """Maximize total counts; negated for MINIMIZE direction."""

    def compute_objective(self, scalars, bin_number):
        """Negate ``image_total`` so MINIMIZE direction drives it up."""
        return -scalars[f"{self.primary_device}_image_total"]

    def compute_observables(self, scalars, bin_number):
        """Expose centroid + peak alongside the objective."""
        dev = self.primary_device
        return {
            "x_CoM": scalars[f"{dev}_x_CoM"],
            "y_CoM": scalars[f"{dev}_y_CoM"],
            "image_peak_value": scalars[f"{dev}_image_peak_value"],
        }
