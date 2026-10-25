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

from geecs_panels.bridge.arrays import (
    EVERY_S,
    MAX_COLUMNS,
    ArraysReader,
    ArraysReply,
    ReadPvs,
    Trace,
    floats,
    trace,
)
from geecs_panels.bridge.feed import Feed, HistorySpec, OnUpdate
from geecs_panels.contracts import DeviceState, RunLock, Saturation
from geecs_panels.feed import ArraysSource, ArraysUnavailable

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


class PlainArrays:
    """A device's on-request roles read together (shared, at most once per :data:`EVERY_S`), then traced.

    The template's ``arrays`` turns the read into traces (each against its
    index, by default) for the columns asked; the read is the same for
    every caller within the interval.
    """

    def __init__(self, feed: Feed, read: Optional[ReadPvs], rules: "Rules") -> None:
        self.feed, self.rules = feed, rules
        device = feed.device
        self.roles = [r for r in device.roles if r.read_mode == "on_request"]
        axis = device.channel_axes[0]
        pvs = [device.prefix + axis + r.suffix for r in self.roles]
        self.source: Optional[ArraysSource] = None
        if read is not None and self.roles:

            async def fetch() -> dict[str, list[float]]:
                values = await read(pvs)
                return {r.name: floats(v) for r, v in zip(self.roles, values)}

            self.source = ArraysSource(fetch, EVERY_S)

    async def reply(self, columns: int) -> ArraysReply:
        """The traces of the read held or made now; a failed read says why."""
        name = self.feed.device.name
        if self.source is None:
            return ArraysReply(device=name)
        try:
            held = await self.source.read()
        except ArraysUnavailable as exc:
            return ArraysReply(device=name, error=str(exc))
        columns = max(1, min(columns, MAX_COLUMNS))
        return ArraysReply(
            device=name, traces=self.rules.arrays(self.feed, held, columns)
        )


#: (feed, the transport's on-request read, the rules) -> the device's arrays reader.
ReaderFactory = Callable[[Feed, Optional[ReadPvs], "Rules"], ArraysReader]


@dataclass(frozen=True)
class Rules:
    """What one template derives; every part optional."""

    history: Optional[HistorySpec] = None
    axis_state: Optional[AxisStateRule] = None
    state: Optional[StateRule] = None
    lock: Optional[Callable[[Feed], RunLock]] = None
    saturation: Optional[SaturationRule] = None
    #: How a read of the on-request roles becomes traces (:class:`PlainArrays`).
    arrays: ArraysRule = plain_arrays
    #: The device's arrays reader: :class:`PlainArrays`, or the template's own.
    reader: ReaderFactory = PlainArrays
    #: Watches each post, for a rule that needs a record's history (the scope's timebase).
    on_update: Optional[OnUpdate] = None


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
