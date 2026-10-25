"""The derived rules, one module per record template: what a family file cannot say.

The family file says which records a device has and how each is judged
fresh; a template's **rules** derive what the stream carries beside the
roles: a state word (worst first over a dozen fields), the run lock, the
saturation chip, which roles the history keeps, how its arrays become
traces. Each module builds one :class:`Rules` from the per-kind pure
functions the per-kind pages already judge by (``motor.axis_state``,
``power.supply_state``, ``camera.lock_word``, ``camera.saturation``,
``scope_feed.traces_payload``), so each rule has one copy.

:func:`rules_for` is the template → rules table; a template without a
module streams its roles and nothing derived.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Optional

from geecs_panels.bridge.arrays import Trace, trace
from geecs_panels.bridge.feed import Feed, HistorySpec
from geecs_panels.contracts import DeviceState, RunLock, Saturation

#: (feed, axis) -> the axis's state word.
AxisStateRule = Callable[[Feed, str], DeviceState]
#: (feed, axis states) -> the device's state word.
StateRule = Callable[[Feed, Mapping[str, DeviceState]], DeviceState]
#: (feed, stale_after) -> the saturation chip.
SaturationRule = Callable[[Feed, float], Saturation]
#: (feed, role -> array read, columns) -> name -> trace.
ArraysRule = Callable[[Feed, Mapping[str, list[float]], int], dict[str, Trace]]


def plain_arrays(
    feed: Feed, arrays: Mapping[str, list[float]], columns: int
) -> dict[str, Trace]:
    """Each array role against its index, decimated; on-request scalars are not traces."""
    return {
        name: trace(values, columns)
        for name, values in arrays.items()
        if feed.device.role(name).type == "array"
    }


@dataclass(frozen=True)
class Rules:
    """What one template derives; every part optional."""

    history: Optional[HistorySpec] = None
    axis_state: Optional[AxisStateRule] = None
    state: Optional[StateRule] = None
    lock: Optional[Callable[[Feed], RunLock]] = None
    saturation: Optional[SaturationRule] = None
    arrays: ArraysRule = plain_arrays


#: No derived rules: the roles alone.
NO_RULES = Rules()


def rules_for(template: str) -> Rules:
    """The template's rules; :data:`NO_RULES` for one without a module."""
    from geecs_panels.bridge.rules import camera, motor, scope, supply

    table = {
        "ad_camera_chain": camera.RULES,
        "motor_record": motor.RULES,
        "scope_contract": scope.RULES,
        "fastps": supply.RULES,
        "easydriver": supply.RULES,
    }
    return table.get(template, NO_RULES)
