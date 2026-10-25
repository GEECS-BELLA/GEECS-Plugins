"""The motor record's rules (``motor_record``): each axis's state word, the device's, the history.

Every word is the record's (:func:`geecs_panels.motor.axis_state`): worst
first, a known failure (no communication, INVALID, a controller problem)
outranks a field that dropped, which reads ``unknown`` naming it (rule 9).
The device's word is its worst axis's, with that axis's words beside it.
The history keeps each axis's readback, sampled on the stream's tick.
"""

from __future__ import annotations

from collections.abc import Mapping

from geecs_panels.bridge.feed import Feed, HistorySpec
from geecs_panels.bridge.rules import Rules
from geecs_panels.contracts import DeviceState
from geecs_schemas.epics.family import Role

from geecs_panels.motor import axis_state, check_write, device_state, msta_flags

#: Each axis's readback, sampled on the stream's tick at most once a second
#: (a quiet positioner then sends one small frame a second, its points).
HISTORY = HistorySpec(keys=("rbv",), every_s=1.0)


def axis(feed: Feed, name: str) -> DeviceState:
    """One axis's state word, from its fields as the record has them (``None`` when dropped)."""
    fields: dict[str, object] = {
        role.name: feed.value(role.name, name) for role in feed.device.roles
    }
    msta = fields.get("msta")
    fields["flags"] = None if msta is None else msta_flags(msta)
    fields["reachable"] = feed.any_connected(name)
    word, status = axis_state(fields)
    return DeviceState(word=word, status=status)  # type: ignore[arg-type]  # the kit's words


def device(feed: Feed, axes: Mapping[str, DeviceState]) -> DeviceState:
    """The worst axis's word; with several axes, which axis says it."""
    if not axes:
        return DeviceState(word="unknown", status="no axes")
    word = device_state([s.word for s in axes.values()])
    name, worst = next((n, s) for n, s in axes.items() if s.word == word)
    status = worst.status if len(axes) == 1 else f"{name}: {worst.status}"
    return DeviceState(word=word, status=status)  # type: ignore[arg-type]  # the kit's words


def check(
    feed: Feed, role: Role, axis: str, value: object, choices: list[str]
) -> object:
    """``motor.check_write``: a target inside the soft limits (when read and distinct), a velocity within VMAX."""
    fields = {r.name: feed.value(r.name, axis) for r in feed.device.roles}
    return check_write(role.suffix, value, fields)


RULES = Rules(history=HISTORY, axis_state=axis, state=device, check=check)
