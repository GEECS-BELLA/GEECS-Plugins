"""The PV bridge's wire contracts: the device stream, puts, the journal, the raw PV stream, snapshots.

Models only, no routes (``Planning/epics_estate.md`` §2.3; the bridge lane
wires them). The web repository's typed clients are generated from these
models' JSON Schemas (``python -m geecs_panels.contracts`` prints the
events document ``scripts/export_openapi.py`` commits as
``GeecsPanels/api/events.schema.json``), so
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

from geecs_schemas.epics.snapshot import Snapshot, SnapshotScope, SnapshotValue
from pydantic.json_schema import models_json_schema
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
    """One role of a device: its reading, and the record fields read beside it (all of them, every time).

    ``fields`` carries the role's family ``fields`` (each a monitor of the
    record's field: the supplies' ``DRVH``/``DRVL``/``HIHI``/``LOLO``, the
    motor ``val``'s ``HLM``/``LLM``) and, for a ``setpoint``, ``EGU`` and
    ``PREC`` from the record's ``DBR_CTRL`` (read at subscribe, refreshed
    when the record's properties change, null while unread) unless the
    family monitors them itself. ``choices`` carries an ``enum``'s state
    names from the same ``DBR_CTRL`` read, so a page keeps no list of its
    own (and a put checks against the same names); on the wire it is absent
    for every other role and while unread. A record's control and alarm
    limits are named only where the family names the field (``DRVH`` on an
    ``ao``, ``HLM`` on a motor): which field ``DBR_CTRL``'s limits come
    from depends on the record type, so the bridge never guesses a name.
    """

    fields: dict[str, Value] = Field(
        default_factory=dict,
        description=(
            "Record field → value: the role's family 'fields' (DRVH, DRVL …), "
            "and a setpoint's EGU and PREC."
        ),
    )
    choices: Optional[list[str]] = Field(
        None,
        description=(
            "An enum's state names, in the record's order (DBR_CTRL). On the "
            "wire absent for any other role and while unread. A put takes the "
            "family's 'choices' when it names them, else these."
        ),
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
    """One ``event: device`` frame of ``GET /api/devices/{name}/events``: a device's live state.

    The first frame after a connect (or a reconnect) is **full**: every
    monitored role, the state (for a family that has one), and, for a
    per-axis family, every axis. Later
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


#: Seconds a device stream may stay quiet before it sends a ``heartbeat``:
#: a page that has seen no event for twice this long is cut off, not quiet.
HEARTBEAT_S = 5.0


class Heartbeat(Contract):
    """One ``event: heartbeat`` of a device stream: nothing changed for :data:`HEARTBEAT_S` seconds.

    Sent only while the stream is quiet, never instead of a frame, and
    without an SSE ``id:`` (``Last-Event-ID`` stays the last frame's), so a
    page tells a quiet device (heartbeats arrive) from a dead connection
    (nothing arrives for twice :data:`HEARTBEAT_S`). ``seq`` is the last
    frame this listener was sent.

    Examples
    --------
    {"device": "U_ChicaneInner", "seq": 1761400000001, "server_time": 1761400005.0}
    """

    device: str = Field(description="The device's GEECS name.")
    seq: int = Field(ge=0, description="The seq of the last frame sent on this stream.")
    server_time: float = Field(description="When the bridge sent it, Unix seconds.")


class DeviceStatus(Contract):
    """One device's line of ``GET /api/devices/state``: its connection, state word and lock, from its live feed.

    Only a device whose feed is live (a page streams it, or its grace
    period has not run out) reports; the route never opens a feed. A
    device without one reads ``live`` false, ``connected`` null and its
    blocks ``unknown``. A block the family has no rule for is absent.

    Examples
    --------
    {"live": true, "connected": true, "state": {"word": "ok", "status": "output on"}}

    {"live": false, "connected": null, "state": {"word": "unknown", "status": "no live stream"}, "lock": {"state": "unknown"}}
    """

    live: bool = Field(
        description="Whether the bridge holds a feed of the device now (none is opened to answer)."
    )
    connected: Optional[bool] = Field(
        None,
        description="Whether any of its records is connected; null without a live feed.",
    )
    state: Optional[DeviceState] = Field(
        None,
        description=(
            "Its state word, for a family that has one (motor, supply): 'unknown' "
            "without a live feed."
        ),
    )
    lock: Optional[RunLock] = Field(
        None,
        description="Its run lock, for a family that has one: 'unknown' without a live feed.",
    )
    saturation: Optional[Saturation] = Field(
        None,
        description="A camera's saturation chip: 'unknown' without a live feed.",
    )


class DeviceStates(Contract):
    """``GET /api/devices/state``: every served device's status in one request (the landing page's).

    Examples
    --------
    {"server_time": 1761400000.0, "devices": {"U_ChicaneInner": {"live": true, "connected": true, "state": {"word": "ok", "status": "output on"}}, "UC_ExampleCam": {"live": false, "connected": null, "lock": {"state": "unknown"}, "saturation": {"state": "unknown"}}}}
    """

    server_time: float = Field(description="When the bridge answered, Unix seconds.")
    devices: dict[str, DeviceStatus] = Field(
        description="Device name → its status, every device the catalog serves."
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
    snapshot: Optional[str] = Field(
        None,
        description="The snapshot a restore wrote this row from (its id); null for a page's own put.",
    )


class LockRelease(Contract):
    """``POST /api/devices/{name}/unlock``: release a stuck run lock (a killed worker), naming who.

    Not a put: the lock is no writable role. The bridge writes the lock
    free and nothing else, journals it like a put (role ``lock``) and
    answers with a :class:`PutResult` (Appendix A rule 6). The device is
    the path's alone, so a body cannot name another.

    Examples
    --------
    {"operator": "S. Barber"}
    """

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
    """One ``event: pv`` frame of ``GET /api/stream/{handle}``: the handle's PVs, full first, then changed only.

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


# ---- snapshots (Planning/epics_estate.md §2.6) ------------------------------------------


class SnapshotCapture(Contract):
    """``POST /api/snapshots``: take a snapshot of a scope's devices from their live streams.

    Examples
    --------
    {"name": "chicane tuned, 25 pC", "operator": "S. Barber", "note": "after the jet realignment", "scope": {"panel": "chicane"}}

    {"name": "stage parked", "operator": "S. Barber", "golden": true, "scope": {"devices": ["U_ESP302_01"]}}
    """

    name: str = Field(min_length=1, max_length=120, description="What to call it.")
    operator: str = Field(
        min_length=1, max_length=60, description="Who takes it (its author)."
    )
    note: str = Field("", max_length=2000, description="Why, in a line or a few.")
    golden: bool = Field(False, description="Mark it a reference setup.")
    scope: SnapshotScope = Field(description="The devices, or a composed panel's.")


class SnapshotSummary(Contract):
    """One line of ``GET /api/snapshots`` (and of ``/api/snapshots/active``): a snapshot without its rows.

    Examples
    --------
    {"id": "s19a1b2c3d4e-0f1e2d3c", "name": "chicane tuned, 25 pC", "created": 1761400000.0, "author": "S. Barber", "note": "", "golden": false, "scope": {"panel": "chicane"}, "devices": ["U_ChicaneInner", "U_ESP302_01"], "rows": 9}
    """

    id: str = Field(description="The snapshot's id: one URL segment.")
    name: str = Field(description="What it is called.")
    created: float = Field(description="When it was taken, Unix seconds.")
    author: str = Field(description="Who took it.")
    note: str = Field("", description="Why.")
    golden: bool = Field(False, description="A reference setup.")
    scope: SnapshotScope = Field(description="What it was taken over.")
    devices: list[str] = Field(description="The devices its rows name, in order.")
    rows: int = Field(ge=1, description="How many rows it keeps.")


class SavedSnapshot(Contract):
    """A snapshot and its id: ``POST /api/snapshots``' answer and ``GET /api/snapshots/{id}``'s.

    Examples
    --------
    {"id": "s19a1b2c3d4e-0f1e2d3c", "snapshot": {"schema_version": 1, "name": "chicane tuned, 25 pC", "created": 1761400000.0, "author": "S. Barber", "scope": {"devices": ["U_ChicaneInner"]}, "rows": [{"device": "U_ChicaneInner", "role": "current_sp", "setpoint": 0.25, "readback": 0.249}]}}
    """

    id: str = Field(description="The snapshot's id: one URL segment.")
    snapshot: Snapshot = Field(description="The document (GEECS-Schemas' Snapshot).")


#: How a row compares with live: ``within`` its tolerance, ``differs``, or
#: ``unreachable`` (its device not served, not connected).
RowState = Literal["within", "differs", "unreachable"]


class RowDiff(Contract):
    """One row of ``GET /api/snapshots/{id}/diff``: the saved value against live.

    Examples
    --------
    {"device": "U_ChicaneInner", "role": "current_sp", "saved": 0.25, "live": 0.3, "readback_saved": 0.249, "readback_live": 0.301, "tolerance": 0.0005, "state": "differs"}

    {"device": "U_ESP302_01", "axis": "m1", "role": "val", "saved": 1.25, "state": "unreachable", "message": "not connected"}
    """

    device: str = Field(description="The device.")
    axis: Optional[str] = Field(None, description="The axis, for a per-axis family.")
    role: str = Field(description="The role.")
    saved: SnapshotValue = Field(description="The setpoint the snapshot keeps.")
    live: Value = Field(None, description="Its value now; null when unreachable.")
    readback_saved: Optional[SnapshotValue] = Field(
        None, description="Its readback at capture."
    )
    readback_live: Value = Field(None, description="Its readback now.")
    tolerance: Optional[float] = Field(
        None,
        description=(
            "How far live may sit and still be within: the family's 'tolerance', "
            "else half the last digit PREC shows, else 0 (exact)."
        ),
    )
    state: RowState = Field(description="within, differs or unreachable.")
    message: str = Field("", description="Why it is unreachable.")


class SnapshotDiff(Contract):
    """``GET /api/snapshots/{id}/diff``: every row against live, and whether all are within (the active verdict).

    Examples
    --------
    {"id": "s19a1b2c3d4e-0f1e2d3c", "server_time": 1761400100.0, "within": false, "rows": [{"device": "U_ChicaneInner", "role": "current_sp", "saved": 0.25, "live": 0.25, "tolerance": 0.0005, "state": "within"}, {"device": "U_ESP302_01", "axis": "m1", "role": "val", "saved": 1.25, "state": "unreachable", "message": "not connected"}], "unreachable": ["U_ESP302_01"]}
    """

    id: str = Field(description="The snapshot.")
    server_time: float = Field(description="When live was read, Unix seconds.")
    within: bool = Field(
        description="Every row within its tolerance of live: the beamline is in this snapshot."
    )
    rows: list[RowDiff] = Field(description="Each row, in the snapshot's order.")
    unreachable: list[str] = Field(
        default_factory=list, description="The devices with an unreachable row."
    )


class RowRef(Contract):
    """A row of a snapshot, by device, axis and role: the operator's tick in the restore view."""

    device: str = Field(min_length=1, description="The device.")
    axis: Optional[str] = Field(None, description="The axis, for a per-axis family.")
    role: str = Field(min_length=1, description="The role.")


class SnapshotRestore(Contract):
    """``POST /api/snapshots/{id}/restore``: write the ticked rows back, through the put gates.

    Never the whole snapshot by default: ``rows`` names each row. Stop on
    the first refusal or continue is chosen before the run.

    Examples
    --------
    {"rows": [{"device": "U_ChicaneInner", "role": "current_sp"}, {"device": "U_ESP302_01", "axis": "m1", "role": "val"}], "operator": "S. Barber", "confirm": true, "stop_on_refusal": true}
    """

    rows: list[RowRef] = Field(
        min_length=1, description="The rows to write, each a row of the snapshot."
    )
    operator: str = Field(
        min_length=1,
        max_length=60,
        description="Who restores, journaled with each row.",
    )
    confirm: bool = Field(
        False,
        description=(
            "The operator confirmed the dialog naming the hazard rows and the "
            "motion rows (each needs it)."
        ),
    )
    stop_on_refusal: bool = Field(
        description="Stop at the first refused, denied or unsettled row (true), or go on (false)."
    )


#: How a restore row ended: the put's outcome (``accepted``, ``refused``,
#: ``denied``), ``unreachable`` (skipped: its device's stream closed, stale
#: or disconnected, or not served), ``unsettled`` (a motion accepted but
#: not done at the target in time, or stopped short), ``not_run`` (after
#: an earlier row stopped the run).
RestoreStatus = Literal[
    "accepted", "refused", "denied", "unreachable", "unsettled", "not_run"
]


class RestoreRow(Contract):
    """One row of a restore: what it wrote and how it ended (its put's result, journaled).

    Examples
    --------
    {"device": "U_ESP302_01", "axis": "m1", "role": "val", "value": 1.25, "status": "accepted", "result": {"outcome": "accepted", "value": 1.25, "message": "dispatched; done when dmov says", "journal_id": 51}, "settled": true, "message": "settled"}

    {"device": "UC_ExampleCam", "role": "exposure", "value": 0.01, "status": "unreachable", "result": {"outcome": "denied", "value": 0.01, "message": "no stream of UC_ExampleCam is open: reload the page", "journal_id": 52}, "message": "no stream of UC_ExampleCam is open: reload the page"}
    """

    device: str = Field(description="The device.")
    axis: Optional[str] = Field(None, description="The axis, for a per-axis family.")
    role: str = Field(description="The role.")
    value: SnapshotValue = Field(
        description="The value written back (the saved setpoint)."
    )
    status: RestoreStatus = Field(
        description="accepted, refused, denied, unreachable, unsettled or not_run."
    )
    result: Optional[PutResult] = Field(
        None,
        description="The put's answer, journaled with the snapshot's id; null: no put (not run, not served).",
    )
    settled: Optional[bool] = Field(
        None,
        description="For a motion row: done at the target within the time allowed.",
    )
    message: str = Field("", description="Why it ended so.")


class RestoreAbort(Contract):
    """``POST /api/snapshots/restore/abort``: stop the running restore before its next row, naming who.

    The row in progress ends (a motion's wait is abandoned: its axis keeps
    going until the operator presses the motor's Stop); every later row is
    ``not_run``.

    Examples
    --------
    {"operator": "S. Barber"}
    """

    operator: str = Field(
        min_length=1, max_length=60, description="Who aborts, logged."
    )


class RestoreAbortAck(Contract):
    """The answer to an abort: whether a restore was running, and of which snapshot.

    Examples
    --------
    {"running": true, "id": "s19a1b2c3d4e-0f1e2d3c"}
    """

    running: bool = Field(description="A restore was running and will stop.")
    id: Optional[str] = Field(None, description="The snapshot it was restoring.")


class RestoreResult(Contract):
    """``POST /api/snapshots/{id}/restore``'s answer: every ticked row, in the snapshot's order.

    Examples
    --------
    {"id": "s19a1b2c3d4e-0f1e2d3c", "operator": "S. Barber", "stopped": true, "rows": [{"device": "U_ChicaneInner", "role": "output_sp", "value": 1, "status": "denied", "result": {"outcome": "denied", "value": 1, "message": "output_sp on U_ChicaneInner needs the hazard confirm (to 1)", "journal_id": 60}, "message": "output_sp on U_ChicaneInner needs the hazard confirm (to 1)"}, {"device": "U_ChicaneInner", "role": "current_sp", "value": 0.25, "status": "not_run", "message": "not run: an earlier row was refused"}]}
    """

    id: str = Field(description="The snapshot.")
    operator: str = Field(description="Who restored.")
    stopped: bool = Field(
        description=(
            "The run stopped early: a refusal (stop_on_refusal), an unsettled "
            "motion (always), or an abort; the rows after it are not_run."
        )
    )
    aborted_by: Optional[str] = Field(
        None, description="Who aborted it, when an abort stopped it."
    )
    unreachable: list[str] = Field(
        default_factory=list, description="The devices skipped as unreachable."
    )
    rows: list[RestoreRow] = Field(description="Each ticked row and how it ended.")


#: Every published contract: name → model.
CONTRACTS: dict[str, type[Contract]] = {
    "device_frame": DeviceFrame,
    "heartbeat": Heartbeat,
    "device_states": DeviceStates,
    "put_request": PutRequest,
    "put_result": PutResult,
    "journal_entry": JournalEntry,
    "lock_release": LockRelease,
    "snapshot_capture": SnapshotCapture,
    "snapshot_summary": SnapshotSummary,
    "saved_snapshot": SavedSnapshot,
    "snapshot_diff": SnapshotDiff,
    "snapshot_restore": SnapshotRestore,
    "restore_result": RestoreResult,
    "restore_abort": RestoreAbort,
    "restore_abort_ack": RestoreAbortAck,
    "subscription_request": SubscriptionRequest,
    "subscription_ack": SubscriptionAck,
    "pv_frame": PvFrame,
}

#: The bridge's SSE ``event:`` names and the model each frame's ``data:`` is.
EVENT_MODELS: dict[str, type[Contract]] = {
    "device": DeviceFrame,
    "heartbeat": Heartbeat,
    "pv": PvFrame,
}

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def json_schemas() -> dict[str, dict[str, Any]]:  # Any: JSON Schema documents
    """Every contract's JSON Schema, by name."""
    return {name: model.model_json_schema() for name, model in CONTRACTS.items()}


def events_document() -> dict[str, Any]:  # Any: a JSON Schema document
    """The bridge's events document, in the shape ``scripts/export_openapi.py`` commits.

    ``properties`` maps each SSE event name to its frame's schema, as the
    exporter's ``events_document`` builds it for every service; ``$defs``
    also carries the request and response contracts, so the web repo
    generates them before the bridge's routes (and their OpenAPI document)
    exist.
    """
    keys, schema = models_json_schema(
        [(model, "serialization") for model in CONTRACTS.values()],
        ref_template="#/$defs/{model}",
    )
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "title": "GEECS PV bridge events",
        "description": (
            "Each property is one Server-Sent-Events `event:` name; its schema "
            "is the JSON object that frame's `data:` carries. A map of the "
            "stream's frames, not a payload. $defs also holds the bridge's "
            "request and response contracts."
        ),
        "type": "object",
        "properties": {
            name: keys[(model, "serialization")] for name, model in EVENT_MODELS.items()
        },
        "required": sorted(EVENT_MODELS),
        "additionalProperties": False,
        "$defs": schema.get("$defs", {}),
    }


def main() -> None:
    """Print the events document (one JSON Schema) for ``scripts/export_openapi.py``."""
    print(json.dumps(events_document(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
