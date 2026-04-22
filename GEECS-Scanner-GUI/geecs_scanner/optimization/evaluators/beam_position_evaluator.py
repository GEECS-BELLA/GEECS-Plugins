"""
Real-hardware beam position evaluator for BAX multipoint alignment.

Reads centroid observables (x_CoM, y_CoM) produced by the configured
image analyzer rather than deriving them from a simulation formula.

Classes
-------
BeamPositionEvaluator
    Evaluator that passes image-analyzer centroid outputs through to Xopt
    as observables.  For simulation/offline testing use
    ``BeamPositionSimulationEvaluator`` from
    ``beam_position_evaluator_simulation``.
"""

from __future__ import annotations

from typing import Dict, List

import logging

from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
    MultiDeviceScanEvaluator,
)

logger = logging.getLogger(__name__)


class BeamPositionEvaluator(MultiDeviceScanEvaluator):
    """
    Evaluator that publishes image-analyzer centroid positions as BAX observables.

    Parameters
    ----------
    observable_names : list of str
        Keys to extract from ``scalar_results`` and forward to Xopt.
        Default is ``["x_CoM"]``.  Add ``"y_CoM"`` for 2-D alignment.
    **kwargs
        Forwarded to ``MultiDeviceScanEvaluator``.
    """

    def __init__(
        self,
        observable_names: List[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.observable_names = (
            observable_names if observable_names is not None else ["x_CoM"]
        )
        self.objective_tag = "BeamPosition"
        self.output_key = None  # BAX models observables only

    def compute_objective(self, scalar_results: dict, bin_number: int) -> float:
        """Not used — BAX models observables only."""
        return None

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> Dict[str, float]:
        """Extract centroid observables from image-analyzer scalar results."""
        device_name = self.analyzer_configs[0].device_name
        result: Dict[str, float] = {}
        for obs in self.observable_names:
            try:
                result[obs] = float(self.get_scalar(device_name, obs, scalar_results))
            except KeyError as e:
                logger.warning("Could not extract observable '%s': %s", obs, e)
        return result
