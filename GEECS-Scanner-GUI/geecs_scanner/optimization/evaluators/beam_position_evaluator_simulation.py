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

        # BAX doesn't use objectives - override parent's output_key
        self.output_key = None

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """
        BAX doesn't use objectives - it only models observables.

        This method is kept for compatibility but should not be called
        when using BAX generators. If you see this being called, check
        that your VOCS has an empty objectives dict.
        """
        # Return None to indicate no objective (BAX doesn't use it)
        return None

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> Dict[str, float]:
        """Publish calibrated centroid x_CoM derived from current_data_bin."""
        try:
            control_val_x = float(np.mean(self.current_data_bin["U_S1H:Current"]))
            # control_val_y = float(np.mean(self.current_data_bin["U_S1V:Current"]))
            measure_val = float(
                np.mean(self.current_data_bin["U_EMQTripletBipolar:Current_Limit.Ch1"])
            )
        except Exception as e:
            logger.warning("Failed to compute x_CoM from current_data_bin: %s", e)
            return {}

        # Realistic ray-tracing simulation:
        # x_CoM = offset × (1 + quad_kick)
        # where offset = (S1H - 1.0) and quad_kick = 0.001 × EMQ
        # Expanded: x_CoM = 1000×(S1H-1) + (S1H-1)×EMQ
        quad_kick_factor = 1.0 * measure_val

        offset_x = control_val_x - 1.0
        centroid_pixels_x = 1000.0 * offset_x * (1.0 + quad_kick_factor)
        centroid_pixels_x += (np.random.random_sample() - 0.5) * 25.0  # -0.05 to +0.05
        centroid_pixels_x = float(centroid_pixels_x * self.calibration)

        # offset_y = control_val_y + 1.0
        # centroid_pixels_y = 1000.0 * offset_y * (1.0 + quad_kick_factor)
        # centroid_pixels_y += (np.random.random_sample() - 0.5) * 25.0  # -0.05 to +0.05
        # centroid_pixels_y = float(centroid_pixels_y * self.calibration)
        #
        # result = {"x_CoM": centroid_pixels_x, "y_CoM": centroid_pixels_y}

        result = {"x_CoM": centroid_pixels_x}

        # Calibrated output expected by VOCS/schema
        return result
