"""
Example evaluator for tracking beam centroid position using MultiDeviceScanEvaluator.

Classes
-------
BeamPositionEvaluator
    Evaluator that reports calibrated centroid observables alongside the optimization objective.
    Includes a lightweight simulation mode that derives observables directly from
    requested setpoints for offline testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    pass

import numpy as np
import logging

from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
    MultiDeviceScanEvaluator,
)

logger = logging.getLogger(__name__)


class BeamPositionEvaluator(MultiDeviceScanEvaluator):
    """
    Minimal evaluator that publishes an x_CoM observable.

    Objective is a constant (0.0); use this when the optimizer needs observables only.
    """

    def __init__(self, calibration: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.calibration = calibration
        self.device_name = self.analyzer_configs[
            0
        ].device_name  # unused, but kept for consistency
        self.objective_tag = "BeamPosition"  # shows as "Objective:BeamPosition"

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """Dummy placeholder."""
        # No objective here—just pass observables
        return 0.0

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> Dict[str, float]:
        """Publish calibrated centroid x_CoM derived from current_data_bin."""
        try:
            control_val = float(np.mean(self.current_data_bin["U_S1H:Current"]))
            measure_val = float(
                np.mean(self.current_data_bin["U_EMQTripletBipolar:Current_Limit.Ch1"])
            )
        except Exception as e:
            logger.warning("Failed to compute x_CoM from current_data_bin: %s", e)
            return {}

        # Simple deterministic affine relation (tight & reproducible)
        centroid_pixels = (measure_val - 1.0) * (control_val - 1.0)

        # Calibrated output expected by VOCS/schema
        return {"x_CoM": float(centroid_pixels * self.calibration)}
