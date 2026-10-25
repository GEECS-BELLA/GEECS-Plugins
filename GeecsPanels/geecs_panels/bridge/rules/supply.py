"""A power supply's rules (``fastps``, ``easydriver``): the state word and the history.

The word is :func:`geecs_panels.power.supply_state` over the supply's
roles and their fields: no reply (the readback INVALID) and a tripped
fault bit are ``failed``, naming it; a channel the word is judged from
that dropped is ``unknown``, naming it (rule 9; a known failure still
wins); a refused command is ``degraded``; an energized output is
``running``. The fault bits are the family's ``fault`` roles, by role
name. The history keeps the current and the voltage, sampled on the
stream's tick.
"""

from __future__ import annotations

from collections.abc import Mapping

from geecs_panels.bridge.feed import Feed, HistorySpec
from geecs_panels.bridge.rules import Rules
from geecs_panels.contracts import DeviceState
from geecs_schemas.epics.family import Role

from geecs_panels.power import ACTIONS, ROLES, check_write, supply_state

#: The readbacks, sampled on the stream's tick at most once a second (a
#: quiet supply then sends one small frame a second, its points).
HISTORY = HistorySpec(keys=("current", "voltage"), every_s=1.0)


def snapshot(feed: Feed) -> dict[str, object]:
    """The supply as :func:`supply_state` reads it: ``role``, ``role.field``, ``bit.<fault role>``."""
    snap: dict[str, object] = {}
    for role in feed.device.roles:
        if role.fault:
            snap[f"bit.{role.name}"] = feed.value(role.name)
            continue
        snap[role.name] = feed.value(role.name)
        for name in role.fields:
            snap[f"{role.name}.{name.lower()}"] = feed.field_value(role.name, name)
    return snap


def state(feed: Feed, _axes: Mapping[str, DeviceState]) -> DeviceState:
    """The supply's state word, judged worst first."""
    bits = [role.name for role in feed.device.roles if role.fault]
    word, status = supply_state(snapshot(feed), feed.any_connected(), bits)
    return DeviceState(word=word, status=status)  # type: ignore[arg-type]  # the kit's words


def check(
    feed: Feed, role: Role, axis: str, value: object, choices: list[str]
) -> object:
    """``power.check_write``: a setpoint inside its drive limits, read and distinct, short of an INVALID HIHI/LOLO.

    An ``ao`` clamps a put past ``DRVH`` and writes the limit, so this is
    the gate that refuses (Appendix A rule 11); limits or severities not
    read refuse.
    """
    action = next(a for a, r in ACTIONS.items() if r == role.name)
    roles = {r.name: r.suffix for r in feed.device.roles if r.name in ROLES}
    _, checked = check_write(action, value, snapshot(feed), roles)
    return checked


RULES = Rules(history=HISTORY, state=state, check=check)
