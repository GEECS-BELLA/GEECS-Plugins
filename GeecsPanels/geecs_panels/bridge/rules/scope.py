"""The scope contract's rules (``scope_contract``): its capture as one trace per channel, and the timebase guard.

The per-kind scope page's :class:`~geecs_panels.scope_feed.ScopeTraces`
reads the capture (rule 3: on request, never monitored): one
``ARR:ArrayData`` array, channels varying fastest, read with its time axis,
shape and identity in one request, and only when the monitored identity
(``arr_uid``) says it is a new capture; it is split and min/max-decimated
(:func:`geecs_panels.scope_feed.traces_payload`) in a worker thread, off
the event loop, once for every viewer, so a one-sample spike survives and
no viewer's poll stalls the streams.

**The timebase guard** (the per-kind page's ``ScopeFeed.timebase_uid``):
the time axis changes as soon as the sample interval, the sample count or
the pre-trigger does, while ``ArrayData`` still holds the capture taken
before. :func:`watch` notes the capture identity at the change; a capture
not after it is answered ``waiting``, never drawn against the new axis.
An IOC restart renumbers captures, so a dropped ``arr_uid`` forgets the mark.
"""

from __future__ import annotations

from typing import Optional

from geecs_panels.bridge.arrays import (
    EVERY_S,
    ArraysReply,
    ReadPvs,
    Trace,
    floats,
)
from geecs_panels.bridge.feed import Channel, Feed
from geecs_panels.bridge.rules import Rules
from geecs_panels.feed import ArraysUnavailable
from geecs_schemas.epics.family import Role

from geecs_panels.scope import CHANNELS, TIMEBASE_KEYS, check_write
from geecs_panels.scope_feed import ScopeTraces

#: The roles whose change makes the held capture's time axis wrong: the
#: per-kind scope page's keys, which the family names its roles after.
TIMEBASE_ROLES = TIMEBASE_KEYS
#: Read together with every capture: role → the key ``traces_payload`` reads.
_CAPTURE = {
    "time_axis": "t",
    "data": "data",
    "arr_uid": "uid",
    "arr_channels": "channels",
    "arr_samples": "samples",
}
#: Where :func:`watch` keeps the mark in ``Feed.state``.
MARK = "timebase_uid"


def watch(feed: Feed, channel: Channel, previous: object, value: object) -> None:
    """Note the capture identity when the timebase changes; forget it when the identity drops."""
    _, name, fld = channel
    if fld:
        return
    if name == "arr_uid" and value is None:
        feed.state.pop(MARK, None)
    elif (
        name in TIMEBASE_ROLES
        and previous is not None
        and value is not None
        and value != previous
    ):
        uid = feed.value("arr_uid")
        number = isinstance(uid, (int, float)) and not isinstance(uid, bool)
        feed.state[MARK] = float(uid) if number else -1.0  # type: ignore[arg-type]


def present(feed: Feed) -> list[str]:
    """The channels the model has, in order: the rows of the capture array."""
    return [
        ch
        for ch in CHANNELS
        if feed.device.has_role(f"ch_{ch.lower()}_present")
        and feed.value(f"ch_{ch.lower()}_present") == "Present"
    ]


class Capture:
    """The scope's capture for every caller: read once per capture, built once, off the loop."""

    def __init__(self, feed: Feed, read: Optional[ReadPvs], _rules: Rules) -> None:
        self.feed = feed
        device = feed.device
        roles = [r for r in _CAPTURE if device.has_role(r)]
        pvs = [device.prefix + device.role(r).suffix for r in roles]
        self.traces: Optional[ScopeTraces] = None
        if read is not None and {"time_axis", "data"} <= set(roles):

            async def fetch_raw() -> dict[str, list[float]]:
                values = await read(pvs)
                return {_CAPTURE[r]: floats(v) for r, v in zip(roles, values)}

            self.traces = ScopeTraces(fetch_raw, lambda: present(feed), EVERY_S)

    async def reply(self, columns: int) -> ArraysReply:
        """The held capture's traces (decimated once, to the scope page's width), or why none."""
        name = self.feed.device.name
        if self.traces is None:
            return ArraysReply(device=name)
        try:
            await self.traces.read(self.feed.value("arr_uid"))
            built = self.traces.payload(self.feed.state.get(MARK))  # type: ignore[arg-type]
        except ArraysUnavailable as exc:
            return ArraysReply(device=name, error=str(exc))
        traces = built.get("traces") or {}
        assert isinstance(traces, dict)
        return ArraysReply(
            device=name,
            traces={ch: Trace(x=t["t"], y=t["v"]) for ch, t in traces.items()},
            waiting=built.get("waiting"),  # type: ignore[arg-type]
        )


def check(
    feed: Feed, role: Role, axis: str, value: object, choices: list[str]
) -> object:
    """``scope.check_write`` by the record written, an enum against the choices its record reported."""
    suffix = role.write_suffix or role.suffix
    return check_write(suffix, value, {suffix: choices})


RULES = Rules(reader=Capture, on_update=watch, check=check)
