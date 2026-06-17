"""Simulation evaluator for BAX multipoint alignment testing.

Derives beam centroid from corrector/quadrupole setpoints via a parametric
ray-tracing model — no analyzer needed. The setpoints come from the
DataLogger as s-file scalars (declared via the ``scalars:`` config block).

Classes
-------
BeamPositionSimulationEvaluator
    Computes ``x_CoM`` from corrector/quadrupole setpoints.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from geecs_scanner.optimization.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class BeamPositionSimulationEvaluator(BaseEvaluator):
    """Simulation-mode BAX evaluator: ``x_CoM`` from setpoints, no hardware.

    The centroid model is::

        x_CoM = scale_factor * (control - reference_setpoint) * (1 + measure) + noise

    Both ``control`` and ``measure`` are read from the s-file via the
    ``scalars`` config block (so the YAML declares them explicitly rather
    than the Python class hardcoding the column names).
    """

    # BAX: observables only, no objective.
    output_key = None

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
        # Auto-add the two simulation inputs to the ``scalars`` config so
        # they get pulled into the per-shot scalars dict by the base class.
        # Caller-supplied ``scalars`` are preserved and extended.
        scalars = list(kwargs.pop("scalars", None) or [])
        for name in (control_variable_name, measurement_variable_name):
            if name not in scalars:
                scalars.append(name)

        super().__init__(scalars=scalars, **kwargs)
        self.control_variable_name = control_variable_name
        self.measurement_variable_name = measurement_variable_name
        self.scale_factor = scale_factor
        self.reference_setpoint = reference_setpoint
        self.noise_amplitude = noise_amplitude
        self.calibration = calibration

    def compute_observables(self, scalars, bin_number) -> Dict[str, float]:
        """Compute ``x_CoM`` from the corrector + quadrupole setpoints."""
        try:
            control_val = scalars[self.control_variable_name]
            measure_val = scalars[self.measurement_variable_name]
        except KeyError as e:
            logger.warning("Simulation evaluator: missing input column %s", e)
            return {}

        offset = control_val - self.reference_setpoint
        centroid_x = self.scale_factor * offset * (1.0 + measure_val)
        centroid_x += (np.random.random_sample() - 0.5) * self.noise_amplitude
        return {"x_CoM": float(centroid_x * self.calibration)}
