"""Gateway-local alias for the shared naming policy.

The normalization policy itself lives in :mod:`geecs_ca_gateway.pv_naming`
(the one module both producer and consumers import — see its docstring).
"""

from __future__ import annotations

from geecs_ca_gateway.pv_naming import normalize_component as normalize_pv_component
from geecs_ca_gateway.pv_naming import setpoint_pv

__all__ = ["normalize_pv_component", "setpoint_pv"]
