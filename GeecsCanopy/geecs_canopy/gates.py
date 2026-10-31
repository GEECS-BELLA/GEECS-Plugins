"""The put path, a function so a push-2 restore runs Squirrel's saved rows through it one by one.

Checks in order, each a distinct outcome and one ``JournalEntry``: the
operator policy (default deny); the device's allow list (Appendix A rule 8);
the hazard confirm (rule 18); the run lock, failing closed (rule 6); the
bounds gate before Hare, failing closed on unread limits (rule 11); then
Hare's put with ``wait`` and a per-role timeout, and the value the record
took (an ``ao`` clamps beyond DRVH silently). An ``attempted`` row is
written before Hare is called, and no put goes out without it; the outcome
row names it, so an attempt without an outcome is a dispatch that never
answered.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from geecs_schemas import JournalEntry, Role

from geecs_canopy.freshness import is_number, json_safe
from geecs_canopy.hare import HareRefused, Wire, WriteValue
from geecs_canopy.journal import JournalUnavailable
from geecs_canopy.module import DevicesUnavailable, Module
from geecs_canopy.native import Resolved

#: The family roles carrying a camera's run lock and its holder; any lock state but Free is held.
LOCK_ROLE, OWNER_ROLE, LOCK_FREE = "lock", "owner", "Free"

#: The module's own slugs and their status: refused (the request, the bounds), denied (its guards).
_REFUSED = {"devices_unavailable": 503, "journal_unavailable": 503, "not_found": 404, "bad_request": 400, "out_of_bounds": 422, "bounds_unknown": 422}  # fmt: skip
_DENIED = {"not_permitted": 403, "not_allowed": 403, "confirm_required": 428, "run_lock_held": 423, "run_lock_unreadable": 423}  # fmt: skip


@dataclass(frozen=True)
class PutRequest:
    """One write: a device's role (on an axis), the value, who asks."""

    device: str
    role: str
    value: WriteValue
    operator: str
    axis: str | None = None
    confirm: bool = False
    key_name: str | None = None


@dataclass
class PutOutcome:
    """The HTTP status, the journal row, and on acceptance what the record took."""

    status: int
    entry: JournalEntry
    extra: dict[str, Any] = field(default_factory=dict)

    def body(self) -> dict[str, Any]:
        """Answer the route: outcome, names, row number; ``code`` on failure, ``detail`` when there is one."""
        e = self.entry
        body = {k: getattr(e, k) for k in ("outcome", "device", "role", "axis")}
        body |= {"journalId": e.id, **self.extra}
        if e.outcome != "accepted":
            body["code"] = e.canopy_code
        return body | ({"detail": e.detail} if e.detail else {})


class _Stop(Exception):
    def __init__(
        self,
        code: str,
        detail: str,
        status: int | None = None,
        attempt: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.code, self.detail, self.attempt = code, detail, attempt
        outcome = "denied" if code in _DENIED else "refused"
        self.status, self.outcome = status or {**_REFUSED, **_DENIED}[code], outcome


def _bounds(role: Role, value: WriteValue, meta: Wire) -> tuple[str, str] | None:
    """``(code, why)`` when ``value`` may not be written to ``role``, else ``None``."""
    no = "out_of_bounds"
    if role.type in ("bit", "command"):
        ok = (is_number(value) or isinstance(value, bool)) and value in (0, 1)
        return None if ok else (no, f"{role.name} takes 0 or 1, not {value!r}")
    if role.type == "enum":
        choices = meta.get("enumChoices") or []
        if not choices:
            return "bounds_unknown", f"{role.name}'s choices are unread"
        finite = is_number(value) and math.isfinite(value)
        whole = finite and value == int(value) and 0 <= value < len(choices)
        label = choices[int(value)] if whole else value
        if label not in choices or (role.choices and label not in role.choices):
            return no, f"{value!r} is none of {role.choices or choices}"
        return None
    if role.text:
        return None if isinstance(value, str) else (no, f"{role.name} takes text")
    if not is_number(value) or not math.isfinite(value):
        return no, f"{role.name} takes a number, not {value!r}"
    if role.integer and value != int(value):
        return no, f"{role.name} takes a whole number, not {value!r}"
    low, high = role.limits or (meta.get("controlLow"), meta.get("controlHigh"))
    if low is None or high is None or low >= high:  # unset DRVL/DRVH read 0/0
        return "bounds_unknown", f"{role.name} has no limits; refused until it has"
    return (
        None if low <= value <= high else (no, f"{value!r} is outside [{low}, {high}]")
    )


async def _check_lock(module: Module, device: Resolved, axis: str | None) -> None:
    roles = {role.name: role for role in device.roles}
    if LOCK_ROLE not in roles:
        raise _Stop("run_lock_unreadable", f"{device.name} has no run lock")
    pv = device.pv(roles[LOCK_ROLE], axis)
    wire, meta = await asyncio.gather(module.hare.get(pv), module.hare.describe(pv))
    choices, value = meta.get("enumChoices") or [], wire.get("value")
    if not wire.get("connected") or not (
        is_number(value) and 0 <= value < len(choices)
    ):
        raise _Stop("run_lock_unreadable", f"{pv} cannot be read")
    if choices[int(value)] != LOCK_FREE:
        owner = roles.get(OWNER_ROLE)
        held_by = owner and (await module.hare.get(device.pv(owner, axis))).get("value")
        raise _Stop("run_lock_held", f"a run holds {device.name}: {held_by or '?'}")


def _entry(req: PutRequest, outcome: str, **fields: Any) -> JournalEntry:
    names = {k: v for k, v in asdict(req).items() if k != "confirm"}
    return JournalEntry(
        time=datetime.now(timezone.utc), outcome=outcome, **names, **fields
    )


async def _write(
    module: Module, device: Resolved, role: Role, req: PutRequest
) -> tuple[dict, str | None, int | None]:
    """Run the device's gates and the put; return what the record took, a clamp note, the attempt row."""
    if role.confirm == "hazard" and not req.confirm:
        raise _Stop("confirm_required", f"{role.name} needs the hazard confirm")
    if role.locked:
        await _check_lock(module, device, req.axis)
    pv = device.pv(role, req.axis, write=True)
    if refusal := _bounds(role, req.value, await module.hare.describe(pv)):
        raise _Stop(*refusal)
    s = module.settings
    timeout = s.settle_timeout_s if role.settled_by else s.put_timeout_s
    # A role that settles itself (an acquire) calls back only when the work ends.
    itself, unsettled = role.settled_by == role.name, None
    try:
        row = _entry(req, "attempted")
        attempt = (await asyncio.to_thread(module.journal.record, row)).id
    except JournalUnavailable as exc:
        why = f"no put without its journal row: {exc}"
        raise _Stop("journal_unavailable", why) from exc
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        await module.hare.put(pv, req.value, wait=not itself, timeout=timeout)
    except HareRefused as exc:
        # A move's put-callback outlasting the timeout is no refusal: the IOC
        # took it and the motion goes on (an access refusal answers at once).
        # The same timeout covers connecting, so an unconnected PV stays refused.
        late = role.settled_by and loop.time() - started >= 0.9 * timeout
        if not late or not (await module.hare.get(pv)).get("connected"):
            raise _Stop(exc.code, exc.detail, exc.status, attempt) from exc
        unsettled = f"dispatched; not settled after {timeout} s: the work goes on"
    taken = (await module.hare.get(pv)).get("value")
    extra = {"requested": req.value}
    if role.type != "command":  # a pressed command's record reads 0 again
        extra["value"] = json_safe(taken)
    if readback := {r.name: r for r in device.roles}.get(role.readback or ""):
        wire = await module.hare.get(device.pv(readback, req.axis))
        extra["readback"] = json_safe(wire.get("value"))
    if itself or unsettled:
        note = unsettled or f"dispatched; {role.name} settles by its readback"
        return extra, note, attempt
    numbers = role.type == "setpoint" and is_number(taken) and is_number(req.value)
    clamped = numbers and not math.isclose(taken, req.value)
    return extra, f"clamped: the record took {taken!r}" if clamped else None, attempt


async def _decide(
    module: Module, req: PutRequest
) -> tuple[dict, str | None, int | None]:
    try:
        device = (await module.devices()).get(req.device)
    except DevicesUnavailable as exc:
        raise _Stop("devices_unavailable", str(exc)) from exc
    if device is None:
        raise _Stop("not_found", f"no device {req.device!r}")
    if (req.axis or None) not in (device.axes or (None,)):
        raise _Stop("bad_request", f"{req.device} takes an axis of {list(device.axes)}")
    try:
        policy = await module.policy()
    except Exception as exc:  # unreadable, not YAML, not valid: deny, and journal it
        raise _Stop("not_permitted", f"the policy is unreadable: {exc}") from exc
    if not policy.permits(req.operator, device.name, device.family.template, req.role):
        raise _Stop("not_permitted", f"no grant lets {req.operator!r} write that")
    role = {r.name: r for r in device.roles}.get(req.role)
    if role is None or req.role not in device.writable_roles():
        raise _Stop("not_allowed", f"{req.role!r} is in none of {list(device.actions)}")
    return await _write(module, device, role, req)


async def put_role(module: Module, req: PutRequest) -> PutOutcome:
    """Run one write through every check, journal it, and say how it ended."""
    try:
        extra, detail, attempt = await _decide(module, req)
        status, outcome, code = 200, "accepted", None
    except _Stop as stop:
        extra, detail, attempt = {}, stop.detail, stop.attempt
        status, outcome, code = stop.status, stop.outcome, stop.code
    entry = _entry(req, outcome, canopy_code=code, detail=detail, attempt=attempt)
    if code == "journal_unavailable":
        return PutOutcome(status, entry, extra)  # no second wait on a failed journal
    # Off the loop: a locked journal must not stall Hare's websockets. If it
    # fails now, Hare's answer stands and the attempted row is the evidence.
    try:
        entry = await asyncio.to_thread(module.journal.record, entry)
    except JournalUnavailable as exc:
        extra["journalError"] = str(exc)
    return PutOutcome(status, entry, extra)
