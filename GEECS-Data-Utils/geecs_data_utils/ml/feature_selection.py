"""Backward-compatible imports for code still using ``ml.feature_selection``.

Prefer ``from geecs_data_utils.analysis import CorrelationReport``.
"""

from geecs_data_utils.analysis.correlation import CorrelationMethod, CorrelationReport

__all__ = ["CorrelationMethod", "CorrelationReport"]
