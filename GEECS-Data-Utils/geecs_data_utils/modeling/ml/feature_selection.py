"""Thin aliases for correlation helpers used next to ML workflows.

Notes
-----
Implementations live in :mod:`geecs_data_utils.analysis.correlation`. Import
from this module only when you want the symbol path under ``modeling.ml``.

See Also
--------
geecs_data_utils.analysis.correlation :
    Canonical definitions of ``CorrelationReport`` and ``CorrelationMethod``.
"""

from geecs_data_utils.analysis.correlation import CorrelationMethod, CorrelationReport

__all__ = ["CorrelationMethod", "CorrelationReport"]
