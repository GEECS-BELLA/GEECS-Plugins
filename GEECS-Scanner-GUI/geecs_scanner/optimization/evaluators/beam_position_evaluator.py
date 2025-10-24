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
    Evaluator for tracking beam position.

    Parameters
    ----------
    calibration : float, default=24.4e-3
        Spatial calibration factor in mm/pixel (or desired units/pixel).
    simulate : bool, default=False
        If True, skip analyzer execution and synthesize observables from the
        setpoints provided to ``get_value`` using a fixed linear model.
    **kwargs
        Additional keyword arguments passed to MultiDeviceScanEvaluator,
        including 'analyzers', 'scan_data_manager', and 'data_logger'.
    """

    def __init__(
        self,
        calibration: float = 1,
        simulate: bool = True,
        **kwargs,
    ):
        """Initialize the beam position evaluator."""
        super().__init__(**kwargs)
        self.calibration = calibration
        # Get device name from first analyzer config
        self.device_name = self.analyzer_configs[0].device_name
        self.observable_key = "x_CoM"
        self.simulate = simulate

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """
        Compute beam position objective from CoM measurements.

        Parameters
        ----------
        scalar_results : dict
            Dictionary of scalar results from all analyzers.
            Expected to contain x_CoM and y_CoM metrics.
        bin_number : int
            Current bin number being evaluated (not used in this implementation).

        Returns
        -------
        float
            Beam position observable value
        """

        return 1

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> Dict[str, float]:
        """
        Provide calibrated centroid observables required by downstream generators.

        Returns a dictionary containing ``x_CoM`` (and optionally ``y_CoM``) expressed
        in physical units. These observable keys must match the VOCS configuration.
        """
        observables: Dict[str, float] = {}


        control_val = np.mean(self.current_data_bin['U_S1H:Current'])
        measurement_val = np.mean(self.current_data_bin["U_EMQTripletBipolar:Current_Limit.Ch1"])

        centroid_pixels = (measurement_val - 1) * (control_val - 1) + np.random.normal(
            0, 0.05
        )

        return {"x_CoM":centroid_pixels}
