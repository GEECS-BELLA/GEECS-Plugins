"""Compatibility helpers over Xopt 3.x / gest-api typed VOCS objects.

Xopt 3.x adopted the `gest-api <https://github.com/campa-consortium/gest-api>`_
``VOCS`` standard.  Variables and objectives are now *typed objects* rather than
bare ``[min, max]`` lists and ``"MAXIMIZE"`` strings:

- ``vocs.variables[name]`` is a ``ContinuousVariable`` (``.domain``) or
  ``DiscreteVariable`` (``.values``), not a 2-element list.
- ``vocs.objectives[name]`` is a ``MaximizeObjective`` / ``MinimizeObjective``,
  not the string ``"MAXIMIZE"`` / ``"MINIMIZE"``.

These helpers centralise the typed-VOCS coupling so call sites stay readable and
the gest-api import lives in exactly one place.
"""

from __future__ import annotations

from typing import Dict, Tuple

from gest_api.vocs import MaximizeObjective
from xopt.vocs import VOCS, get_variable_bounds

__all__ = ["is_maximize", "variable_bounds", "bounds_of"]


def is_maximize(vocs: VOCS, objective_name: str) -> bool:
    """Return True if the named objective is a maximize objective.

    Replaces the pre-3.x ``str(vocs.objectives[name]).upper() == "MAXIMIZE"``
    idiom, which silently breaks under typed objectives (``str()`` of a
    ``MaximizeObjective`` no longer contains ``"MAXIMIZE"``).
    """
    return isinstance(vocs.objectives[objective_name], MaximizeObjective)


def variable_bounds(vocs: VOCS) -> Dict[str, Tuple[float, float]]:
    """Return ``{variable_name: (lo, hi)}`` for every VOCS variable.

    Thin pass-through to :func:`xopt.vocs.get_variable_bounds`, which handles
    both continuous (``.domain``) and discrete (min/max of ``.values``)
    variables.  Use this instead of unpacking ``vocs.variables[name]`` as a
    ``[lo, hi]`` list, which no longer works in Xopt 3.x.
    """
    return get_variable_bounds(vocs)


def bounds_of(vocs: VOCS, variable_name: str) -> Tuple[float, float]:
    """Return ``(lo, hi)`` bounds for a single VOCS variable."""
    return get_variable_bounds(vocs)[variable_name]
