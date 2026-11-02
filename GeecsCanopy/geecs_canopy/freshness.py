"""Freshness, judged continuously from the module's internal Hare subscription.

Verdicts claim no more than the evidence (``Planning/operator_surfaces.md``
§3) and keep Appendix A of ``Planning/epics_estate.md``: the producing
counter's advance is freshness, a changing value never is, a ``Compute*``
flag off reads ``off`` (rule 1); disconnected is not stale, a known fault
outranks a dropped field (rule 9); NaN travels as null (rule 10). ``tsNs``
is never read: a quiet live record looks old by it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from geecs_core.pv_naming import DEVICE_STATUS_VARIABLE, ioc_heartbeat_pv
from geecs_schemas import Role

from geecs_canopy.hare import Wire
from geecs_canopy.native import Resolved

#: The words: live, stale, off, not answering, unreachable, error, unknown. ``not running``
#: needs an independent service-status source (§3) the estate lacks: never emitted yet.

#: ``menuAlarmStat`` READ, COMM, TIMEOUT: the IOC answers, the hardware behind it does not.
_NOT_ANSWERING, _UDF = frozenset({1, 9, 10}), 17


#: An IOC's devIocStats heartbeat: GEECS-Core's rule, the one the archiver keeps.
heartbeat_pv = ioc_heartbeat_pv


def is_number(value: Any) -> bool:
    """A number, not a bool."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def json_safe(value: Any) -> Any:
    """Turn NaN and infinities into ``None``, inside lists too (rule 10): JSON has no NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


class Tracker:
    """The last frame of every watched PV, and when each one last advanced."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.latest: dict[str, Wire] = {}
        self._first: dict[str, float] = {}
        self._advanced: dict[str, float] = {}
        self._clock = clock

    def on_frame(self, data: dict[str, Wire]) -> None:
        """Take one ``initial`` or ``diff`` frame's data. A counter advances only between two connected readings."""
        now = self._clock()
        for pv, wire in data.items():
            before = self.latest.get(pv)
            self.latest[pv] = wire
            self._first.setdefault(pv, now)
            a = before.get("value") if before and before.get("connected") else None
            b = wire.get("value") if wire.get("connected") else None
            if is_number(a) and is_number(b) and b > a:
                self._advanced[pv] = now

    def counter(self, pv: str, stale_after_s: float) -> str:
        """``live`` if the counter advanced within the window, ``stale`` if watched longer without, else ``unknown``."""
        wire = self.latest.get(pv)
        if wire is None:
            return "unknown"
        if not wire.get("connected"):
            return "unreachable"
        now = self._clock()
        if now - self._advanced.get(pv, -math.inf) <= stale_after_s:
            return "live"
        return "stale" if now - self._first[pv] > stale_after_s else "unknown"


def judge(
    wire: Wire | None, flag: Wire | None, counter: str | None, *, flagged: bool = False
) -> str:
    """One role's verdict from its frame, its ``computed_by`` flag's (``flagged``: it has one) and its ``fresh_by`` counter's."""
    if wire is None:
        return "unknown"
    if not wire.get("connected"):
        return "unreachable"
    severity, status = wire.get("severity"), wire.get("alarmStatus")
    if severity == "INVALID" and status == _UDF:
        return "unknown"  # the record has never been processed: no value yet
    if severity == "INVALID" and status in _NOT_ANSWERING:
        return "not answering"
    if severity in ("MAJOR", "INVALID"):
        return "error"
    if flagged and not (flag and flag.get("connected")):
        return "unknown"  # the counter cannot say the computation is on
    if flag is not None and flag.get("connected") and flag.get("value") in (0, False):
        return "off"
    return counter or "live"


def device_state(verdicts: list[str], faulted: bool, heartbeat: str | None) -> str:
    """The device's one word: a tripped fault first (rule 9), then the worst evidence."""
    if faulted:
        return "error"
    if heartbeat == "unreachable" or (verdicts and set(verdicts) == {"unreachable"}):
        return "unreachable"
    if heartbeat == "stale":  # the IOC's channel is up and its own counter stopped
        return "stale"
    for word in ("not answering", "error"):
        if word in verdicts:
            return word
    if "unreachable" in verdicts or "unknown" in verdicts:
        return "unknown"  # a dropped field is not a stale device (rule 9)
    return "stale" if "stale" in verdicts else "live"


def _judged(device: Resolved, role: Role) -> bool:
    """Whether the module watches and judges ``role``: a native device's monitored roles; a DB device's ``connected`` alone.

    The gateway serves thousands of PVs and Hare caps subscriptions: a DB
    device's values are the page's to read over Hare, and a put ``get``s.
    """
    if device.source == "gateway":
        return role.name == DEVICE_STATUS_VARIABLE
    return role.read_mode == "monitor"


def watched_pvs(device: Resolved) -> list[str]:
    """Every PV the module subscribes for ``device``: its judged roles, and its IOC's heartbeat."""
    pvs = [
        device.pv(role, axis)
        for axis, _ in device.bases()
        for role in device.roles
        if _judged(device, role)
    ]
    return pvs + ([heartbeat_pv(device.ioc)] if device.ioc else [])


def connected_state(wire: Wire | None) -> str:
    """A DB device's one word from the gateway's ``connected`` PV: Disconnected is unreachable."""
    if wire is None:
        return "unknown"
    if not wire.get("connected"):
        return "unreachable"  # the gateway itself does not answer
    return {0: "unreachable", 1: "live", "Disconnected": "unreachable", "Connected": "live"}.get(wire.get("value"), "unknown")  # fmt: skip


def view(device: Resolved, tracker: Tracker, stale_after_s: float) -> dict[str, Any]:
    """Answer for the device: state, heartbeat, and every role per axis with its verdict."""
    by_name = {role.name: role for role in device.roles}
    writable = device.writable
    hb_pv = device.ioc and heartbeat_pv(device.ioc)
    hb = tracker.counter(hb_pv, stale_after_s) if hb_pv else None
    roles, verdicts, faulted = [], [], False
    for axis, _ in device.bases():
        latest = {r.name: tracker.latest.get(device.pv(r, axis)) for r in device.roles}
        for role in device.roles:
            wire, verdict = latest[role.name] or {}, None
            if device.source == "gateway" and _judged(device, role):
                verdict = connected_state(latest[role.name])  # the device's state
                verdicts.append(verdict)
            elif _judged(device, role):
                counter = None
                if source := role.fresh_by or (role.type == "counter" and role.name):
                    counter_pv = device.pv(by_name[source], axis)
                    counter = tracker.counter(counter_pv, stale_after_s)
                flag = latest.get(role.computed_by or "")
                flagged = bool(role.computed_by)
                verdict = judge(latest[role.name], flag, counter, flagged=flagged)
                if (
                    not role.writable
                ):  # the state is what the device reports, not its inputs
                    verdicts.append(verdict)
            faulted |= bool(role.fault and wire.get("connected") and wire.get("value"))
            entry = {"role": role.name, "axis": axis, "read": role.read_mode}
            entry |= role.model_dump(include={"label", "type", "confirm", "locked"})
            entry["pv"] = device.pv(role, axis)
            entry["writePv"] = (
                role.writable and device.pv(role, axis, write=True) or None
            )
            entry |= {"writable": role.name in writable, "verdict": verdict}
            entry |= {k: wire.get(k) for k in ("severity", "alarmStatus", "connected")}
            roles.append(entry | {"value": json_safe(wire.get("value"))})
    state = device_state(verdicts, faulted, hb)
    return {"state": state, "heartbeat": hb, "roles": roles}
