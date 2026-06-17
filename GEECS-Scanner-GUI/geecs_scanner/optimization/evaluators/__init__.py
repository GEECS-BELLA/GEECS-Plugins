"""Optimization evaluators for the GEECS scanner.

All concrete evaluators inherit from :class:`BaseEvaluator` and override
:meth:`compute_objective` and/or :meth:`compute_observables` (both peer
hooks; either or both can be implemented). The base class handles
diagnostic-driven analyzers, s-file scalar columns, and the per-shot
vs per-bin dispatch.
"""

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.optimization.evaluators.beam_position_evaluator import (
    BeamPositionEvaluator,
)
from geecs_scanner.optimization.evaluators.beam_position_evaluator_simulation import (
    BeamPositionSimulationEvaluator,
)
from geecs_scanner.optimization.evaluators.beam_size_evaluator import BeamSizeEvaluator
from geecs_scanner.optimization.evaluators.beam_sum_counts_evaluator import (
    MaxCountsEvaluator,
)

__all__ = [
    "BaseEvaluator",
    "BeamPositionEvaluator",
    "BeamPositionSimulationEvaluator",
    "BeamSizeEvaluator",
    "MaxCountsEvaluator",
]
