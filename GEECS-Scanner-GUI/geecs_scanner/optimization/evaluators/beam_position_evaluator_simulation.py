"""
Simulation evaluator for BAX multipoint alignment testing.

Overrides ``compute_observables`` to derive beam centroid position from a
physics-based formula rather than real camera images, enabling offline
convergence testing without hardware.

Classes
-------
BeamPositionSimulationEvaluator
    Computes x_CoM from corrector/quadrupole setpoints via a parametric
    ray-tracing model.  Use ``BeamPositionEvaluator`` for real hardware runs.
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


class BeamPositionSimulationEvaluator(MultiDeviceScanEvaluator):
    """
    Simulation-mode evaluator that publishes x_CoM derived from setpoints.

    The centroid model is:
        x_CoM = scale_factor * (S1H - reference_setpoint) * (1 + EMQ) + noise

    Parameters
    ----------
    control_variable_name : str
        Name of the horizontal corrector variable in current_data_bin.
    measurement_variable_name : str
        Name of the quadrupole current variable in current_data_bin.
    scale_factor : float
        Pixels per unit corrector offset (1000.0 for nominal HTU geometry).
    reference_setpoint : float
        Corrector setpoint that produces zero offset (1.0 A nominal).
    noise_amplitude : float
        Peak-to-peak pixel noise added to the centroid (±noise_amplitude/2).
    calibration : float
        Pixel-to-physical-unit scale applied after the simulation.
    """

    def __init__(
        self,
        control_variable_name: str = "U_S1H:Current",
        measurement_variable_name: str = "U_EMQTripletBipolar:Current_Limit.Ch1",
        scale_factor: float = 1000.0,
        reference_setpoint: float = 1.0,
        noise_amplitude: float = 25.0,
        calibration: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.control_variable_name = control_variable_name
        self.measurement_variable_name = measurement_variable_name
        self.scale_factor = scale_factor
        self.reference_setpoint = reference_setpoint
        self.noise_amplitude = noise_amplitude
        self.calibration = calibration

        self.objective_tag = "BeamPosition"
        self.output_key = None  # BAX doesn't use objectives

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """Not used — BAX models observables only."""
        return None

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> Dict[str, float]:
        """Compute x_CoM from corrector and quadrupole setpoints via simulation."""
        try:
            control_val = float(
                np.mean(self.current_data_bin[self.control_variable_name])
            )
            measure_val = float(
                np.mean(self.current_data_bin[self.measurement_variable_name])
            )
        except Exception as e:
            logger.warning(
                "Failed to read simulation inputs from current_data_bin: %s", e
            )
            return {}

        offset = control_val - self.reference_setpoint
        centroid_x = self.scale_factor * offset * (1.0 + measure_val)
        centroid_x += (np.random.random_sample() - 0.5) * self.noise_amplitude
        centroid_x = float(centroid_x * self.calibration)

        return {"x_CoM": centroid_x}


# Backwards-compatible alias used in older scan configs
BeamPositionEvaluator = BeamPositionSimulationEvaluator
