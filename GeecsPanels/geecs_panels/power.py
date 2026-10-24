"""The power-supply panel's contract: the roles a supply's PVs play, the write allow-list, the state words.

Pure: no CA, no I/O. Supplies have no shared record set the way positioners
have the motor record: each family's IOC names its own PVs (fpstest1's
FAST-PS and eztest1's Easy-Driver share the gateway's ``current``,
``current:SP``, ``enable_output:SP`` … but not their status bits, and only
the FAST-PS has a regulation loop). So a supply's bench entry maps
**roles** to its PV suffixes, and the panel speaks roles only:

- ``current_sp``, ``current``, ``voltage``, ``output_sp``, ``output``,
  ``reset`` — every supply names them (the front-panel minimum for a power
  supply: setpoint and readbacks, output on/off, plus the fault reset);
- ``loop_mode`` (shown, never written) and ``voltage_sp`` — when it has them;
- ``bits`` — the fault and interlock bits, each a ``bi`` that reads 1 when
  tripped, by the label the page shows.

**Writes energize things.** A setpoint is checked against its record's
drive limits (``.DRVH``/``.DRVL``) here first, and refused while they are
unread or the record sets none (equal limits: an ``ao`` then clamps
nothing). This is the gate that refuses: an ``ao`` CLAMPS a put beyond its
drive limits and writes the limit, so the IOC alone would turn a 1.0 A
request into DRVH on the wire (the IOC's own refusal is HIHI/LOLO inside
them with IVOA "Don't drive outputs", set at load). The output takes 0 or
1, the reset a press. A write the IOC
accepts can still be refused by the supply (a StreamDevice ``#NAK`` puts the
record in an INVALID alarm), so the live transport reads the record's
alarm after each put (:class:`geecs_panels.ca.LiveSupplies`).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from typing import Optional


#: The registry kind the power-supply panel serves.
POWER_KIND = "power_supply"


class NotWritable(ValueError):
    """A write the allow-list refuses; the message says why."""


class NotAllowed(NotWritable):
    """An action the panel takes, but not on this device (its ``allow``)."""


#: Roles every supply names.
REQUIRED_ROLES = ("current_sp", "current", "voltage", "output_sp", "output", "reset")
#: Roles a supply names when it has them.
OPTIONAL_ROLES = ("voltage_sp", "loop_mode")
ROLES = REQUIRED_ROLES + OPTIONAL_ROLES

#: The roles read as strings (an enum by its state name).
STRING_ROLES = frozenset({"loop_mode"})

#: Fields read beside a role's value: key suffix → record field.
ROLE_FIELDS: dict[str, dict[str, str]] = {
    "current_sp": {"drvh": ".DRVH", "drvl": ".DRVL", "hihi": ".HIHI", "lolo": ".LOLO",
                   "hhsv": ".HHSV", "llsv": ".LLSV", "egu": ".EGU", "prec": ".PREC",
                   "sevr": ".SEVR", "stat": ".STAT"},
    "voltage_sp": {"drvh": ".DRVH", "drvl": ".DRVL", "hihi": ".HIHI", "lolo": ".LOLO",
                   "hhsv": ".HHSV", "llsv": ".LLSV", "egu": ".EGU", "prec": ".PREC",
                   "sevr": ".SEVR", "stat": ".STAT"},
    "current": {"egu": ".EGU", "prec": ".PREC", "sevr": ".SEVR", "stat": ".STAT"},
    "voltage": {"egu": ".EGU", "prec": ".PREC"},
    "output_sp": {"sevr": ".SEVR", "stat": ".STAT"},
}  # fmt: skip

#: Field keys read as strings.
STRING_FIELDS = frozenset({"egu", "sevr", "stat", "hhsv", "llsv"})

#: The panel's actions on a supply, each with the role it writes. A device
#: may permit a subset (its bench entry's ``allow``); one that names none
#: permits every action whose role it has.
ACTIONS: dict[str, str] = {
    "current": "current_sp",
    "voltage": "voltage_sp",
    "output": "output_sp",
    "reset": "reset",
}


def monitored(
    roles: Mapping[str, str], bits: Mapping[str, str]
) -> dict[str, tuple[str, bool]]:
    """Snapshot key → (PV suffix, read as string) for one supply: its roles, their fields, its bits.

    ``reset`` is a press-only record with nothing to show, so it is never
    monitored.
    """
    keys: dict[str, tuple[str, bool]] = {}
    for role in ROLES:
        suffix = roles.get(role)
        if suffix is None or role == "reset":
            continue
        keys[role] = (suffix, role in STRING_ROLES)
        for key, field in ROLE_FIELDS.get(role, {}).items():
            keys[f"{role}.{key}"] = (suffix + field, key in STRING_FIELDS)
    for label, suffix in bits.items():
        keys[f"bit.{label}"] = (suffix, False)
    return keys


def check_roles(roles: Mapping[str, str], allow: Optional[Collection[str]]) -> None:
    """Refuse a role map missing a required role or naming an unknown one, or an unknown action."""
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    if missing:
        raise ValueError(f"a power supply names every required role; missing {missing}")
    unknown = sorted(set(roles) - set(ROLES))
    if unknown:
        raise ValueError(f"unknown role(s) {unknown}; the roles are {list(ROLES)}")
    if allow is not None:
        bad = sorted(set(allow) - set(ACTIONS))
        if bad:
            raise ValueError(
                f"unknown power-supply action(s) {bad}; the actions are {list(ACTIONS)}"
            )
        absent = sorted(a for a in allow if ACTIONS[a] not in roles)
        if absent:
            raise ValueError(
                f"action(s) {absent} need a role this supply does not name"
            )


def allowed_actions(
    roles: Mapping[str, str], allow: Optional[Collection[str]]
) -> list[str]:
    """The actions the device permits, in the panel's order: those whose role it names, within *allow*."""
    return [
        a for a, role in ACTIONS.items()
        if role in roles and (allow is None or a in allow)
    ]  # fmt: skip


def _number(action: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NotWritable(f"{action} takes a number, not {value!r}")
    if not math.isfinite(value):
        raise NotWritable(f"{action} takes a finite number")
    return float(value)


def _within_drive_limits(role: str, number: float, snap: Mapping[str, object]) -> float:
    low, high = snap.get(f"{role}.drvl"), snap.get(f"{role}.drvh")
    egu = snap.get(f"{role}.egu") or ""
    if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
        raise NotWritable("the setpoint's drive limits have not been read: refused")
    if low >= high:
        raise NotWritable(
            "the setpoint record sets no drive limits (DRVH = DRVL): refused"
        )
    if not low <= number <= high:
        raise NotWritable(
            f"{number:g} {egu} is outside the drive limits {low:g} to {high:g} {egu}".replace(
                "  ", " "
            )
        )
    # The IOC's own refusal (HIHI/LOLO with an INVALID severity and IVOA
    # "Don't drive outputs"): a value at or past either is not written.
    for bound, severity, past in (("hihi", "hhsv", number >= _f(snap, role, "hihi")),
                                  ("lolo", "llsv", number <= _f(snap, role, "lolo"))):  # fmt: skip
        if snap.get(f"{role}.{severity}") == "INVALID" and past:
            limit = snap.get(f"{role}.{bound}")
            raise NotWritable(
                f"{number:g} {egu} reaches the IOC's {bound.upper()} {limit:g} {egu}, "
                "which it refuses to write".replace("  ", " ")
            )
    return number


def _f(snap: Mapping[str, object], role: str, key: str) -> float:
    value = snap.get(f"{role}.{key}")
    return float(value) if isinstance(value, (int, float)) else math.nan


def check_write(
    action: str,
    value: object,
    snap: Mapping[str, object],
    roles: Mapping[str, str],
    allow: Optional[Collection[str]] = None,
) -> tuple[str, object]:
    """The role to write and the value to put for an allowed action, or :class:`NotWritable`.

    *snap* is the supply's snapshot values (``current_sp.drvh`` …);
    *allow* the device's permitted actions (``None``: all it has). A
    setpoint must sit within its record's drive limits, read and distinct;
    the output takes 0 or 1 (or a bool); the reset takes 1.
    """
    role = ACTIONS.get(action)
    if role is None:
        raise NotWritable(f"{action!r} is not a power-supply action")
    if action not in allowed_actions(roles, allow):
        if role not in roles:
            raise NotWritable(f"this supply has no {role} record")
        raise NotAllowed(f"{action} is not permitted on this device")
    if action in ("current", "voltage"):
        return role, _within_drive_limits(role, _number(action, value), snap)
    if action == "output":
        if value not in (0, 1) or (
            not isinstance(value, bool) and not isinstance(value, int)
        ):
            raise NotWritable("output takes 0 (off) or 1 (on)")
        return role, int(value)
    if value not in (1, True):
        raise NotWritable("reset takes 1 (a press)")
    return role, 1


def supply_state(
    snap: Mapping[str, object], reachable: bool, bits: Collection[str]
) -> tuple[str, str]:
    """The kit state word and the words beside it, judged worst first.

    No field connected is ``unknown``; an INVALID readback (no reply from
    the supply) is ``failed``, as is any tripped fault or interlock bit
    (named); a setpoint or output command the supply or the IOC refused (its
    record INVALID, other than UDF: a record never written since the IOC
    started) is ``degraded``; an energized output is ``running``; an output
    that is off is ``ok``.
    """
    if not reachable:
        return "unknown", "connecting"
    if snap.get("current.sevr") == "INVALID":
        return "failed", f"no reply ({snap.get('current.stat') or 'INVALID'})"
    tripped = [label for label in bits if snap.get(f"bit.{label}") == 1]
    if tripped:
        return "failed", ", ".join(label.replace("_", " ") for label in tripped)
    for role, what in (("current_sp", "setpoint"), ("voltage_sp", "voltage setpoint"),
                       ("output_sp", "output command")):  # fmt: skip
        stat = snap.get(f"{role}.stat")
        # UDF: never written since the IOC started — not a refusal.
        if snap.get(f"{role}.sevr") == "INVALID" and stat != "UDF":
            return "degraded", f"{what} refused ({stat or 'INVALID'})"
    if snap.get("output") == 1:
        return "running", "output on"
    if snap.get("output") == 0:
        return "ok", "output off"
    return "unknown", "output state not read"
