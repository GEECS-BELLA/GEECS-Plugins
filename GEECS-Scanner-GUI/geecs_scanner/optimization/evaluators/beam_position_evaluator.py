"""
Real-hardware beam position evaluator for BAX multipoint alignment.

Reads centroid observables produced by the configured image analyzer
rather than deriving them from a simulation formula.

Classes
-------
BeamPositionEvaluator
    Publishes image-analyzer centroid outputs as BAX observables.
"""

from __future__ import annotations

import logging
from typing import Dict, List

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
        Forwarded to :class:`MultiDeviceScanEvaluator`.
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
        self.output_key = None  # BAX models observables only — no objective

    def compute_observables(
        self, scalar_results: dict, bin_number: int
    ) -> Dict[str, float]:
        """Extract centroid observables from image-analyzer scalar results."""
        result: Dict[str, float] = {}
        for obs in self.observable_names:
            try:
                result[obs] = float(
                    self.get_scalar(self.primary_device, obs, scalar_results)
                )
            except KeyError as e:
                logger.warning("Could not extract observable '%s': %s", obs, e)
        return result
