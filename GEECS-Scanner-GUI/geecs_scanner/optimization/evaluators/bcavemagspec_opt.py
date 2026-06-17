"""Maximize spectral density on the BCave mag-spec stitcher.

Classes
-------
EBeamSourceOpt
    Maximize ``U_BCaveMagSpec-interpSpec_objective`` (negated for MINIMIZE).
"""

from __future__ import annotations

from geecs_scanner.optimization.base_evaluator import BaseEvaluator


class EBeamSourceOpt(BaseEvaluator):
    """Maximize the stitched mag-spec objective scalar."""

    def compute_objective(self, scalars, bin_number):
        """Negate the analyzer's stitched-objective scalar."""
        return -scalars[f"{self.primary_device}_U_BCaveMagSpec-interpSpec_objective"]
