"""The positioner panel's contract: the motor record's fields, the write allow-list, the state words.

Pure: no CA, no I/O. A positioner (the registry's ``epics_motor``) is one
EPICS motor record per axis under the device's prefix
(``<experiment>:<device>:<axis>``), so the contract is the record itself
(`epics-modules/motor`): the fields every motor driver fills, no vendor's
word anywhere. What the panel shows per axis is the front-panel minimum
for a stage: the position readback with its setpoint beside it, absolute
and relative moves, stop, the soft limits and the limit switches, the
homing state, the velocity — each with its alarm colour.

**Writes move things.** The design note (`Planning/device_panels.md` §1)
sends a moving kind's writes through the scanner's ``POST /api/move``,
serialized with scans; that route needs the scanner to know native axes
(``NATIVE_DEVICES.md`` §10, slice L), which it does not yet. Until then
the prototype writes the record directly behind this allow-list: a target
is checked against the record's soft limits here first (the record
refuses it too, with ``LVIO``), the tweak step and velocity are checked
for sense, and the press-only fields take 1. Nothing a scan records is
written.

The field table here is the kind's suffix table of ``NATIVE_DEVICES.md``
§10, which that design places beside the kind models in
``geecs_schemas.epics`` once the scanner's readback and the archive pass
need it too; until then it lives with its one consumer, as the camera's
and the scope's do.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

#: The registry kind the positioner panel serves.
MOTOR_KIND = "epics_motor"

#: Per-axis fields: snapshot key → record field. All monitored.
AXIS_FIELDS: dict[str, str] = {
    "desc": ".DESC",
    "egu": ".EGU",
    "prec": ".PREC",
    "rbv": ".RBV",
    "val": ".VAL",
    "dmov": ".DMOV",
    "movn": ".MOVN",
    "hlm": ".HLM",
    "llm": ".LLM",
    "hls": ".HLS",
    "lls": ".LLS",
    "lvio": ".LVIO",
    "velo": ".VELO",
    "vmax": ".VMAX",
    "vbas": ".VBAS",
    "accl": ".ACCL",
    "rdbd": ".RDBD",
    "twv": ".TWV",
    "msta": ".MSTA",
    "sevr": ".SEVR",
    "stat": ".STAT",
    "set": ".SET",
    "spmg": ".SPMG",
    "off": ".OFF",
}

#: Read as strings: the enums by their state names, the texts.
STRING_KEYS = frozenset({"desc", "egu", "sevr", "stat", "set", "spmg"})

#: ``.MSTA`` bits by name, numbered as the record documents them (1-based).
MSTA_BITS: dict[str, int] = {
    "direction_positive": 1,
    "done": 2,
    "plus_ls": 3,
    "home_ls": 4,
    "position": 6,
    "slip_stall": 7,
    "home": 8,
    "present": 9,
    "problem": 10,
    "moving": 11,
    "gain_support": 12,
    "comm_err": 13,
    "minus_ls": 14,
    "homed": 15,
}


def msta_flags(value: object) -> dict[str, bool]:
    """The record's ``.MSTA`` status word as named flags (all False when unread)."""
    try:
        word = int(value)  # type: ignore[arg-type]  # an unread value is None
    except (TypeError, ValueError):
        return {name: False for name in MSTA_BITS}
    return {name: bool(word & (1 << (bit - 1))) for name, bit in MSTA_BITS.items()}


#: The fields an axis's state word is judged from, and its readback: while
#: any is not connected (``None`` in the snapshot) a stopped, healthy axis
#: cannot be told from one whose motion, switches or alarm go unheard.
STATE_KEYS = ("rbv", "dmov", "movn", "msta", "sevr", "hls", "lls", "lvio", "spmg")


def axis_state(axis: Mapping[str, object]) -> tuple[str, str]:
    """The kit state word and the words beside it for one axis's snapshot.

    Judged from the record, worst first: no communication (``.MSTA``
    ``comm_err``, or ``.SEVR`` INVALID with a COMM/TIMEOUT status), any
    other INVALID (``UDF`` at an IOC start) and a controller problem are
    ``failed``; a field of :data:`STATE_KEYS` not connected is ``unknown``
    (naming them), as is an axis none of whose fields has connected;
    motion held by ``.SPMG`` (Stop / Pause: a move request moves nothing)
    is ``degraded``; a move in progress is ``running``; a limit switch, a
    target refused by the soft limits (``.LVIO``) or an alarm is
    ``degraded``; a stopped axis is ``ok``.
    """
    if not axis.get("reachable"):
        return "unknown", "connecting"
    flags = axis.get("flags") or {}
    sevr, stat = axis.get("sevr"), axis.get("stat")
    fault = _fault(flags, sevr, stat)
    if fault is not None:
        return fault
    unread = [key for key in STATE_KEYS if axis.get(key) is None]
    if unread:
        return "unknown", "not connected: " + " ".join(unread)
    spmg = axis.get("spmg")
    if spmg in ("Stop", "Pause"):
        return "degraded", f"motion held (SPMG {spmg})"
    if axis.get("movn") or flags.get("moving") or axis.get("dmov") == 0:
        return "running", "moving"
    if axis.get("hls") or flags.get("plus_ls"):
        return "degraded", "at the high limit switch"
    if axis.get("lls") or flags.get("minus_ls"):
        return "degraded", "at the low limit switch"
    if axis.get("lvio"):
        return "degraded", "target outside the soft limits"
    if sevr in ("MINOR", "MAJOR"):
        return "degraded", f"alarm {axis.get('stat') or sevr}"
    return "ok", "stopped"


def _fault(
    flags: Mapping[str, bool], sevr: object, stat: object
) -> tuple[str, str] | None:
    """A known failure, which outranks everything: no communication, invalid, a controller problem."""
    if flags.get("comm_err") or (sevr == "INVALID" and stat in ("COMM", "TIMEOUT")):
        return "failed", "no communication"
    if sevr == "INVALID":
        return "failed", f"invalid ({stat or 'no status'})"
    if flags.get("problem"):
        return "failed", "controller problem"
    if flags.get("slip_stall"):
        return "failed", "slip or stall"
    return None


def device_state(states: list[str]) -> str:
    """The worst of the axes' state words, for the device's one chip."""
    for word in ("failed", "unknown", "degraded", "running", "ok"):
        if word in states:
            return word
    return "unknown"


class NotWritable(ValueError):
    """A write the allow-list refuses; the message says why."""


#: Press-only fields: a 1 makes the record act (tweak, stop, home).
PRESS_FIELDS = frozenset({".TWF", ".TWR", ".STOP", ".HOMF", ".HOMR"})

#: Puts that complete only when the motion ends: dispatched without waiting
#: for completion (the stream shows the move; ``.DMOV`` says when it is done).
NO_WAIT = frozenset({".VAL", ".TWF", ".TWR", ".HOMF", ".HOMR"})

#: Every field the panel may write, and nothing else.
WRITABLE = frozenset({".VAL", ".TWV", ".VELO", *PRESS_FIELDS})


def _number(field: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NotWritable(f"{field} takes a number, not {value!r}")
    if not math.isfinite(value):
        raise NotWritable(f"{field} takes a finite number")
    return float(value)


def _limits(axis: Mapping[str, object]) -> tuple[float, float] | None:
    low, high = axis.get("llm"), axis.get("hlm")
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return float(low), float(high)
    return None


def check_write(field: str, value: object, axis: Mapping[str, object]) -> object:
    """The value to put for an allowed write, or :class:`NotWritable`.

    *axis* is the axis's snapshot: a ``.VAL`` target is checked against its
    soft limits (``.LLM``/``.HLM``) when both have been read and differ.
    Limits that read equal are left to the record: its own no-limit rule
    is both *dial* limits zero, of which equal user limits are the usual
    sign, and a one-point range is the record's to refuse (``LVIO``), as
    it refuses anything this check lets through.  ``.VELO`` is checked
    against ``.VMAX`` when the record has one; a press-only field takes 1.
    """
    if field not in WRITABLE:
        raise NotWritable(f"{field} is not writable from the panel")
    if field in PRESS_FIELDS:
        if value not in (1, True):
            raise NotWritable(f"{field} takes 1 (a press)")
        return 1
    number = _number(field, value)
    if field == ".VAL":
        limits = _limits(axis)
        if limits is not None and limits[0] != limits[1]:
            low, high = limits
            if not low <= number <= high:
                egu = axis.get("egu") or ""
                raise NotWritable(
                    f"{number:g} {egu} is outside the soft limits "
                    f"{low:g} to {high:g} {egu}".replace("  ", " ")
                )
        return number
    if number <= 0:
        raise NotWritable(f"{field} takes a positive number")
    if field == ".VELO":
        vmax = axis.get("vmax")
        if isinstance(vmax, (int, float)) and vmax > 0 and number > vmax:
            raise NotWritable(f".VELO takes at most {vmax:g} (the record's VMAX)")
    return number
