"""Real-hardware beam position evaluator for BAX multipoint alignment.

Reads centroid observables from an image-analyzer diagnostic and forwards
them to Xopt as named observables. BAX-style algorithms don't have an
objective; they model observables.

Classes
-------
BeamPositionEvaluator
    Publishes image-analyzer centroid outputs as BAX observables.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from geecs_scanner.optimization.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class BeamPositionEvaluator(BaseEvaluator):
    """BAX evaluator: forwards image-analyzer centroid keys as observables.

    Parameters
    ----------
    observable_names : list of str, optional
        Per-analyzer metric names (e.g. ``["x_CoM", "y_CoM"]``) to extract
        from the primary device's analyzer output. Defaults to
        ``["x_CoM"]``.
    """

    # BAX: observables only, no objective.
    output_key = None

    def __init__(self, observable_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.observable_names = (
            observable_names if observable_names is not None else ["x_CoM"]
        )

    def compute_observables(self, scalars, bin_number) -> Dict[str, float]:
        """Pull each requested metric off the primary device's analyzer output."""
        dev = self.primary_device
        out: Dict[str, float] = {}
        for name in self.observable_names:
            key = f"{dev}_{name}"
            if key in scalars:
                out[name] = float(scalars[key])
            else:
                logger.warning("BAX evaluator: missing observable '%s'", key)
        return out
