"""The PV bridge's wire contracts: the device stream, puts, the journal, the raw PV stream.

Models only, no routes (``Planning/epics_estate.md`` §2.3; the bridge lane
wires them). The web repository's typed clients are generated from these
models' JSON Schemas (``python -m geecs_panels.contracts`` prints them), so
**a model here changes only together with those clients regenerated**.

Rules the stream carries (Appendix A):

- A value is judged fresh by the producing plugin's counter, never by the
  value changing; a value whose ``Compute*`` flag is off is ``off`` (rule 1).
- Disconnected is not stale: a dropped value is ``null`` with freshness
  ``unknown`` (rule 9).
- NaN and infinities travel as ``null``, or the stream dies (rule 10).

Times are Unix seconds (wall clock, the bridge's), so a page plots them
without parsing; an age is the page's to compute (``server_time`` or its
own clock minus ``advanced_at``), so a steady value is never re-sent just
because it got older.

Two nulls, two meanings: inside a reading, a null ``value`` is a value
that is not there (disconnected, never read, not finite); a frame-level
optional block (``state``, ``lock``, ``saturation``, ``history``) that is
null or absent is unchanged in a delta, and not applicable in a full
frame. ``Freshness`` replaces ``camera.Freshness``'s ``disconnected`` with
``unknown`` (rule 9): the bridge lane retires the old word.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

#: How fresh one value is: ``fresh`` (its producer is advancing, or a setting
#: that is connected; the kit marks it with nothing), ``stale`` (the kit's
#: pane state: connected, but its producer has not advanced within the
#: bridge's threshold), ``off`` (its ``Compute*`` flag is off: frozen, not
#: measured), ``unknown`` (not connected, or never read).
Freshness = Literal["fresh", "stale", "off", "unknown"]

#: The kit's status words (GeecsWebTheme ``kit.css``), a device's or an axis's state.
StatusWord = Literal[
    "queued", "running", "paused", "ok", "degraded", "failed", "unknown"
]

#: An EPICS alarm severity.
Severity = Literal["NO_ALARM", "MINOR", "MAJOR", "INVALID"]

#: A value on the wire: NaN and infinities become ``None`` before they get here.
Value = Optional[Union[StrictBool, StrictInt, StrictFloat, StrictStr]]


def _finite(value: object) -> object:
    """Return ``None`` for NaN and infinities (rule 10: the stream must not die on one)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


_EXAMPLE = re.compile(r"Examples\n\s*-+\n(?P<body>.*)", re.S)


def docstring_examples(
    model: type[BaseModel],
) -> list[dict[str, Any]]:  # Any: free-form JSON
    """The JSON examples in a model's docstring ``Examples`` section, one per paragraph."""
    match = _EXAMPLE.search(model.__doc__ or "")
    if match is None:
        return []
    blocks = re.split(r"\n\s*\n", match.group("body").strip())
    return [json.loads(block) for block in blocks if block.strip()]


def _schema_examples(
    schema: dict[str, Any], model: type[BaseModel]
) -> None:  # Any: JSON Schema
    examples = docstring_examples(model)
    if examples:
        schema["examples"] = examples


class Contract(BaseModel):
    """Base of every wire model: unknown keys refused, docstring examples in the schema."""

    model_config = ConfigDict(extra="forbid", json_schema_extra=_schema_examples)


# ---- the device stream -------------------------------------------------------


class Reading(Contract):
    """One value as the stream carries it, always whole: the value, its alarm, when its source last advanced, how fresh it is.

    A reading is complete in every frame that carries it, full or delta:
    a delta sends a changed role or PV whole (``fields`` included), never
    a partial one to merge. It is re-sent when anything in it changes,
    the freshness word included (fresh → stale), but not as it ages.
    """

    value: Value = Field(
        None,
        description=(
            "The value (an enum by its state name); null while disconnected, "
            "never read, or not finite."
        ),
    )
    severity: Optional[Severity] = Field(
        None, description="The record's alarm severity; null while unread."
    )
    advanced_at: Optional[float] = Field(
        None,
        description=(
            "When its freshness source last advanced, Unix seconds (the producing "
            "plugin's counter, else the value's own monitor); null: not seen to. "
            "Its age is server_time (or the page's clock) minus this."
        ),
    )
    freshness: Freshness = Field(description="fresh, stale, off or unknown.")

    @field_validator("value", mode="before")
    @classmethod
    def _finite_value(cls, value: object) -> object:
        return _finite(value)


class RoleReading(Reading):
    """One role of a device: its reading, and the record fields its family reads beside it (all of them, every time)."""

    fields: dict[str, Value] = Field(
        default_factory=dict,
        description="Record field → value, the role's family 'fields' (DRVH, DRVL …).",
    )

    @field_validator("fields", mode="before")
    @classmethod
    def _finite_fields(cls, fields: object) -> object:
        if isinstance(fields, dict):
            return {k: _finite(v) for k, v in fields.items()}
        return fields


class DeviceState(Contract):
    """A derived state word and the words beside it (``motor.axis_state``, ``power.supply_state``)."""

    word: StatusWord = Field(description="The kit's status word.")
    status: str = Field(
        description="What the page shows beside it: 'at the high limit switch'."
    )


class RunLock(Contract):
    """The run lock (``camera.lock_state``): a run holding the device refuses its locked roles."""

    state: Literal["free", "locked", "unknown"] = Field(
        description="free, locked, or unknown (not served or not read: writes are refused)."
    )
    owner: Optional[str] = Field(None, description="The run holding it, while locked.")


class Saturation(Contract):
    """Whether a camera's raw frame (STAT0) reaches its pixel format's full scale."""

    state: Literal["saturated", "ok", "unknown"] = Field(description="The chip's word.")
    fraction: Optional[float] = Field(None, description="Raw max / full scale.")


class History(Contract):
    """The server-kept history the frame carries, sampled on the producer's tick.

    Examples
    --------
    {"keys": ["cx", "cy"], "points": [[1761400000.0, 512.1, 480.0], [1761400000.5, null, 480.2]]}
    """

    keys: list[str] = Field(
        description="The roles, in each point's order after its time."
    )
    points: list[list[Optional[float]]] = Field(
        description="[time, value…] per sample, oldest first; null is a gap (not computed, not connected)."
    )

    @field_validator("points", mode="before")
    @classmethod
    def _finite_points(cls, points: object) -> object:
        if isinstance(points, list):
            return [
                [_finite(v) for v in p] if isinstance(p, list) else p for p in points
            ]
        return points


class AxisFrame(Contract):
    """One axis of a per-axis family (a motor record): its roles and its state."""

    roles: dict[str, RoleReading] = Field(
        default_factory=dict,
        description="Role → reading (a delta frame: changed roles only).",
    )
    state: Optional[DeviceState] = Field(
        None,
        description="The axis's state word; in a delta frame, only when it changed.",
    )


class DeviceFrame(Contract):
    """One event of ``GET /api/devices/{name}/events``: a device's live state.

    The first frame after a connect (or a reconnect) is **full**: every
    monitored role, the state, and, for a per-axis family, every axis. Later
    frames are **deltas**: ``roles`` / ``axes`` carry only what changed, and
    ``state``, ``lock``, ``saturation`` and ``history`` appear only when
    they changed. Each role a frame carries is whole (:class:`Reading`).
    A role that dropped arrives with ``value`` null and freshness
    ``unknown``. ``seq`` is the SSE ``id:`` cursor.

    Examples
    --------
    {"device": "U_ChicaneInner", "family": "fastps", "seq": 1, "server_time": 1761400000.0, "full": true, "connected": true, "roles": {"current": {"value": 0.25, "severity": "NO_ALARM", "advanced_at": 1761399999.6, "freshness": "fresh", "fields": {"EGU": "A", "PREC": 3}}, "output": {"value": 1, "severity": "NO_ALARM", "advanced_at": 1761399998.0, "freshness": "fresh"}, "interlock_1": {"value": null, "freshness": "unknown"}}, "state": {"word": "unknown", "status": "not connected: bit.interlock_1"}}

    {"device": "U_ChicaneInner", "family": "fastps", "seq": 2, "server_time": 1761400000.5, "full": false, "connected": true, "roles": {"current": {"value": 0.31, "severity": "NO_ALARM", "advanced_at": 1761400000.5, "freshness": "fresh", "fields": {"EGU": "A", "PREC": 3}}}}

    {"device": "U_ESP302_01", "family": "motor_record", "seq": 7, "server_time": 1761400001.0, "full": false, "connected": true, "axes": {"m1": {"roles": {"rbv": {"value": 1.25, "severity": "NO_ALARM", "advanced_at": 1761400000.9, "freshness": "fresh"}}, "state": {"word": "running", "status": "moving"}}}, "state": {"word": "running", "status": "moving"}}

    {"device": "UC_ExampleCam", "family": "ad_camera_chain", "seq": 1, "server_time": 1761400000.0, "full": true, "connected": true, "roles": {"cx": {"value": 512.0, "severity": "NO_ALARM", "advanced_at": 1761399999.95, "freshness": "fresh"}, "max": {"value": 4000, "advanced_at": 1761399999.95, "freshness": "off"}}, "lock": {"state": "free"}, "saturation": {"state": "ok", "fraction": 0.61}, "history": {"keys": ["cx"], "points": [[1761399999.5, 511.9]]}}
    """

    device: str = Field(description="The device's GEECS name.")
    family: str = Field(description="Its family (record template).")
    seq: int = Field(
        ge=0, description="The event's cursor, the SSE id; increases by one per frame."
    )
    server_time: float = Field(description="When the bridge sent it, Unix seconds.")
    full: bool = Field(
        description="True: the whole device (after a connect); false: changed fields only."
    )
    connected: bool = Field(
        description="Whether any of the device's records is connected."
    )
    roles: dict[str, RoleReading] = Field(
        default_factory=dict,
        description="Role → reading, for a family without axes (a delta: changed roles only).",
    )
    axes: dict[str, AxisFrame] = Field(
        default_factory=dict,
        description="Axis → its roles and state, for a per-axis family (a delta: changed axes only).",
    )
    state: Optional[DeviceState] = Field(
        None,
        description=(
            "The device's state word (worst of its axes'), for a family that has "
            "one (motor, supply); in a full frame whenever it does. A camera's "
            "state is its lock and saturation."
        ),
    )
    lock: Optional[RunLock] = Field(
        None, description="The run lock, for a family that has one."
    )
    saturation: Optional[Saturation] = Field(
        None, description="A camera's saturation chip."
    )
    history: Optional[History] = Field(
        None,
        description="New history points (full: the last minutes; delta: since the last frame).",
    )


# ---- puts and the journal ------------------------------------------------------


class PutRequest(Contract):
    """``POST /api/put``: one write to one role of one device, by a named operator.

    Examples
    --------
    {"device": "U_ChicaneInner", "role": "current_sp", "value": 0.5, "operator": "S. Barber"}

    {"device": "U_ChicaneInner", "role": "output_sp", "value": 1, "operator": "S. Barber", "confirm": true}

    {"device": "U_ESP302_01", "axis": "m2", "role": "val", "value": 12.5, "operator": "S. Barber"}
    """

    device: str = Field(min_length=1, description="The device's GEECS name.")
    role: str = Field(
        min_length=1, description="The role written (a family role name)."
    )
    axis: Optional[str] = Field(
        None,
        description="The axis, for a per-axis family (the motor record); else absent.",
    )
    value: Union[StrictBool, StrictInt, StrictFloat, StrictStr] = Field(
        description="The value: a number, a flag, text, or an enum's state name."
    )
    operator: str = Field(
        min_length=1,
        max_length=60,
        description="Who is writing, journaled with the put (identity, not authentication).",
    )
    confirm: bool = Field(
        False,
        description="The operator confirmed a hazard role's write in the page's dialog.",
    )


#: How a put ended: ``accepted`` (the IOC took it and its record is not in an
#: INVALID alarm), ``refused`` (the value: outside a limit, the bridge's or the
#: record's; the IOC or the device refusing it), ``denied`` (the write: not
#: permitted on this device, the run lock held or unread, a hazard
#: unconfirmed, the device's stream stale or disconnected, a read-only bridge).
PutOutcome = Literal["accepted", "refused", "denied"]


class PutResult(Contract):
    """The answer to a put, and what the journal recorded.

    Examples
    --------
    {"outcome": "accepted", "value": 0.5, "severity": "NO_ALARM", "message": "", "journal_id": 41}

    {"outcome": "refused", "value": 1.5, "severity": null, "message": "1.5 A is outside the drive limits 0 to 1 A", "journal_id": 42}

    {"outcome": "denied", "value": 1, "message": "output_sp needs the hazard confirm", "journal_id": 43}
    """

    outcome: PutOutcome = Field(description="accepted, refused or denied.")
    value: Value = Field(
        None, description="The value put (after coercion), or the one asked for."
    )
    severity: Optional[Severity] = Field(
        None, description="The record's alarm severity after the put; null if not read."
    )
    message: str = Field(
        "", description="Why it was refused or denied, or the record's alarm status."
    )
    journal_id: int = Field(ge=1, description="The journal entry recording this put.")


class JournalEntry(Contract):
    """One line of ``GET /api/journal``: who wrote what, when, and how it ended.

    Examples
    --------
    {"id": 41, "time": 1761400002.0, "operator": "S. Barber", "device": "U_ChicaneInner", "role": "current_sp", "value": 0.5, "outcome": "accepted", "severity": "NO_ALARM", "message": ""}
    """

    id: int = Field(ge=1, description="The entry's number, increasing.")
    time: float = Field(description="When the put was answered, Unix seconds.")
    operator: str = Field(description="Who wrote.")
    device: str = Field(description="The device.")
    role: str = Field(description="The role.")
    axis: Optional[str] = Field(None, description="The axis, for a per-axis family.")
    value: Value = Field(None, description="The value asked for.")
    outcome: PutOutcome = Field(description="accepted, refused or denied.")
    severity: Optional[Severity] = Field(
        None, description="The record's severity after the put."
    )
    message: str = Field(
        "", description="Why it was refused or denied, or the alarm status."
    )


class LockRelease(Contract):
    """``POST /api/devices/{name}/unlock``: release a stuck run lock (a killed worker), naming who.

    Not a put: the lock is no writable role. The bridge writes the lock
    free and nothing else, journals it like a put (role ``lock``) and
    answers with a :class:`PutResult` (Appendix A rule 6).

    Examples
    --------
    {"device": "UC_ExampleCam", "operator": "S. Barber"}
    """

    device: str = Field(
        min_length=1, description="The device whose run lock is released."
    )
    operator: str = Field(
        min_length=1, max_length=60, description="Who is releasing it, journaled."
    )


# ---- the raw PV stream ---------------------------------------------------------


class SubscriptionRequest(Contract):
    """``POST /api/subscriptions``: a handle on a set of bare PVs (the inspector, a panel's ``pv`` widget).

    Examples
    --------
    {"pvs": ["undulator:chicane:field", "undulator:shutter:state"]}
    """

    pvs: list[str] = Field(
        min_length=1,
        max_length=200,
        description="Full PV names; monitors are shared across handles.",
    )

    @field_validator("pvs")
    @classmethod
    def _names(cls, pvs: list[str]) -> list[str]:
        bad = [pv for pv in pvs if not pv or any(c.isspace() for c in pv)]
        if bad:
            raise ValueError(f"not PV names: {bad}")
        if len(set(pvs)) != len(pvs):
            raise ValueError("a PV is named twice")
        return pvs


class SubscriptionAck(Contract):
    """The handle: stream it at ``GET /api/stream/{handle}`` until the grace period runs out unread.

    Examples
    --------
    {"handle": "h-3f9c", "pvs": ["undulator:chicane:field", "undulator:shutter:state"], "grace_seconds": 30.0}
    """

    handle: str = Field(min_length=1, description="The handle's id.")
    pvs: list[str] = Field(description="The PVs it streams, as asked.")
    grace_seconds: float = Field(
        gt=0, description="How long the handle outlives its last listener."
    )


class PvFrame(Contract):
    """One event of ``GET /api/stream/{handle}``: the handle's PVs, full first, then changed only.

    Examples
    --------
    {"handle": "h-3f9c", "seq": 1, "server_time": 1761400000.0, "full": true, "pvs": {"undulator:chicane:field": {"value": 0.42, "severity": "NO_ALARM", "advanced_at": 1761399999.8, "freshness": "fresh"}, "undulator:shutter:state": {"value": null, "freshness": "unknown"}}}
    """

    handle: str = Field(description="The handle.")
    seq: int = Field(ge=0, description="The event's cursor, the SSE id.")
    server_time: float = Field(description="When the bridge sent it, Unix seconds.")
    full: bool = Field(
        description="True: every PV (after a connect); false: changed PVs only."
    )
    pvs: dict[str, Reading] = Field(description="PV → reading.")


#: Every published contract: name → model. The export prints these.
CONTRACTS: dict[str, type[Contract]] = {
    "device_frame": DeviceFrame,
    "put_request": PutRequest,
    "put_result": PutResult,
    "journal_entry": JournalEntry,
    "lock_release": LockRelease,
    "subscription_request": SubscriptionRequest,
    "subscription_ack": SubscriptionAck,
    "pv_frame": PvFrame,
}


def json_schemas() -> dict[str, dict[str, Any]]:  # Any: JSON Schema documents
    """Every contract's JSON Schema, by name."""
    return {name: model.model_json_schema() for name, model in CONTRACTS.items()}


def main() -> None:
    """Print every contract's JSON Schema as one JSON object (name → schema)."""
    print(json.dumps(json_schemas(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
