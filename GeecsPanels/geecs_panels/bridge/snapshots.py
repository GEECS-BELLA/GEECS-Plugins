"""Snapshots v0: capture, compare against live, restore by row (``Planning/epics_estate.md`` §2.6).

A :class:`~geecs_schemas.epics.snapshot.Snapshot` is a named set of
setpoints across devices with the readbacks beside them. The bridge keeps
**one authoritative copy**, a JSON file per snapshot in its
``StateDirectory`` (:class:`SnapshotStore`). Not in v0: the
``{experiment}/snapshots/`` mirror on the data tree, golden snapshots
promoted into the configs repository by PR, snapshot-against-snapshot
compare (two documents, no live read: the page's), deleting one.

- **Capture** (``POST /api/snapshots``) takes every writable ``setpoint``,
  ``enum`` and ``bit`` role the device permits (:func:`snapshot_roles`), per
  axis, from the **live feeds**: a device whose stream is not open and
  ticking, or a role not connected, refuses the whole capture, naming them.
- **Diff** (``GET /api/snapshots/{id}/diff``) judges each row against live:
  ``within`` its tolerance (:func:`tolerance`: the family's ``tolerance``,
  else half the last digit its ``PREC`` shows, else exact), ``differs``, or
  ``unreachable``. It reads the devices' feeds, opening one for a device no
  page streams (evicted after the grace period like any other) and waiting
  up to :data:`CONNECT_S` for its records.
- **Active** (``GET /api/snapshots/active``): the snapshots whose every row
  is within tolerance of live, the experiment home's "which preset are we
  in".
- **Restore** (``POST /api/snapshots/{id}/restore``) writes the rows the
  operator ticked, **through the put gates** (:class:`~geecs_panels.bridge.puts.Puts`,
  never around them), one at a time in the snapshot's order, each journaled
  with the snapshot's id. A row the stale-page gate denies (409: the
  device's stream closed, stale, or its record disconnected) is
  ``unreachable``: skipped, grouped, and the run goes on. A **motion row**
  (a role ``settled_by`` another, the motor's ``val`` by ``dmov``) needs the
  operator's ``confirm`` like a hazard role and is waited on until its done
  flag reads done and its setpoint still reads the target (a Stop moves a
  motor record's ``VAL`` to where it stopped: ``unsettled``). A refused,
  denied or unsettled row ends the run when ``stop_on_refusal``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from geecs_schemas.epics.family import Role
from geecs_schemas.epics.snapshot import Snapshot, SnapshotRow, SnapshotValue
from pydantic import ValidationError

from geecs_panels.bridge.catalog import BridgeDevice
from geecs_panels.bridge.feed import Feed
from geecs_panels.bridge.puts import Denied, Puts, permitted
from geecs_panels.bridge.stream import DeviceStreams, UnknownDevice
from geecs_panels.contracts import (
    PutRequest,
    RestoreResult,
    RestoreRow,
    RowDiff,
    SavedSnapshot,
    SnapshotCapture,
    SnapshotDiff,
    SnapshotRestore,
    SnapshotSummary,
)

logger = logging.getLogger(__name__)

#: The snapshots' folder in the state directory.
SNAPSHOTS_FOLDER = "snapshots"
#: Seconds a diff waits for a device's records to connect (a feed it opened).
CONNECT_S = 2.0
#: Seconds a restore waits for a motion row to settle.
MOTION_TIMEOUT_S = 120.0
#: Seconds a motion has to be seen starting before a done flag that never
#: dropped counts as done (a target the axis already sits at).
MOTION_START_S = 1.0
#: How often the waits look again.
POLL_S = 0.05

#: Role types a snapshot keeps: the settings, never a press, an array or a readback.
SNAPSHOT_TYPES = frozenset({"setpoint", "enum", "bit"})

_ID = re.compile(r"^s[0-9a-f]+-[0-9a-f]{8}$")


class SnapshotError(Exception):
    """A capture, a diff or a restore that cannot run; the status is the HTTP answer."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def default_folder() -> Optional[Path]:
    """``$STATE_DIRECTORY/snapshots`` when systemd gives the unit one, else ``None`` (in memory)."""
    state = os.environ.get("STATE_DIRECTORY")
    if not state:
        return None
    return Path(state.split(":")[0]) / SNAPSHOTS_FOLDER


def snapshot_roles(device: BridgeDevice) -> list[Role]:
    """The roles a snapshot keeps: writable, monitored setpoints, enums and bits the device permits.

    A role settled by itself (a camera's or a scope's ``acquire``) is a
    start, whose state is a run, not a setting: never kept.
    """
    return [
        role
        for role in device.roles
        if role.type in SNAPSHOT_TYPES
        and role.writable
        and role.read_mode == "monitor"
        and role.settled_by != role.name
        and permitted(device, role)
    ]


def is_motion(role: Role) -> bool:
    """A write that completes when another role says so (the motor's ``val``, done by ``dmov``)."""
    return role.settled_by is not None and role.settled_by != role.name


def tolerance(role: Role, precision: object) -> float:
    """How far live may sit from a saved value and still be it: the family's, else half the last shown digit, else 0."""
    if role.tolerance is not None:
        return role.tolerance
    if (
        isinstance(precision, int)
        and not isinstance(precision, bool)
        and precision >= 0
    ):
        return 0.5 * 10.0**-precision
    return 0.0


def within(saved: SnapshotValue, live: object, tol: float) -> bool:
    """Whether *live* is *saved*: numbers within *tol* (and a relative 1e-9), anything else equal."""
    numbers = (int, float)
    if (
        isinstance(saved, numbers)
        and isinstance(live, numbers)
        and not isinstance(saved, bool)
        and not isinstance(live, bool)
    ):
        return math.isclose(float(live), float(saved), rel_tol=1e-9, abs_tol=tol)
    return live == saved


def _kept(value: object) -> Optional[SnapshotValue]:
    """A live value as a row keeps it; ``None`` for nothing read or not finite."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return None


def summary(snapshot_id: str, snapshot: Snapshot) -> SnapshotSummary:
    """A snapshot's list line."""
    return SnapshotSummary(
        id=snapshot_id,
        name=snapshot.name,
        created=snapshot.created,
        author=snapshot.author,
        note=snapshot.note,
        golden=snapshot.golden,
        scope=snapshot.scope,
        devices=list(dict.fromkeys(r.device for r in snapshot.rows)),
        rows=len(snapshot.rows),
    )


class SnapshotStore:
    """The snapshots' one authoritative copy: ``<folder>/<id>.json`` each, or memory without a folder."""

    def __init__(self, folder: Optional[Path] = None, *, warn: bool = True) -> None:
        if folder is None and warn:
            logger.warning(
                "snapshots are in memory: no --snapshots and no $STATE_DIRECTORY"
            )
        self.folder = folder
        self._memory: dict[str, Snapshot] = {}
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)

    def save(self, snapshot: Snapshot) -> str:
        """Keep *snapshot*; its new id (time-ordered, one URL segment, never 'active')."""
        snapshot_id = f"s{int(snapshot.created * 1000):x}-{uuid.uuid4().hex[:8]}"
        if self.folder is None:
            self._memory[snapshot_id] = snapshot
            return snapshot_id
        path = self.folder / f"{snapshot_id}.json"
        partial = path.with_suffix(".json.partial")
        partial.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        os.replace(partial, path)  # a reader never sees half a file
        return snapshot_id

    def get(self, snapshot_id: str) -> Optional[Snapshot]:
        """The snapshot of that id; ``None`` if there is none (or the id is not one of ours)."""
        if not _ID.match(snapshot_id):
            return None
        if self.folder is None:
            return self._memory.get(snapshot_id)
        path = self.folder / f"{snapshot_id}.json"
        if not path.is_file():
            return None
        return self._read(path)

    def all(self) -> list[tuple[str, Snapshot]]:
        """Every snapshot, newest first; a file that does not read is logged and left out."""
        if self.folder is None:
            found = list(self._memory.items())
        else:
            found = []
            for path in self.folder.glob("*.json"):
                if _ID.match(path.stem):
                    snapshot = self._read(path)
                    if snapshot is not None:
                        found.append((path.stem, snapshot))
        return sorted(found, key=lambda item: item[1].created, reverse=True)

    @staticmethod
    def _read(path: Path) -> Optional[Snapshot]:
        try:
            return Snapshot.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("snapshot %s does not read: %s", path.name, exc)
            return None


#: A composed panel's name -> the devices it embeds; ``None``: no such panel.
PanelDevicesOf = Callable[[str], Optional[list[str]]]


@dataclass
class Snapshots:
    """The bridge's snapshots: capture from the live feeds, diff, the active verdict, restore through the put gates."""

    streams: DeviceStreams
    puts: Puts
    store: SnapshotStore = field(default_factory=lambda: SnapshotStore(warn=False))
    panel_devices: PanelDevicesOf = lambda _name: None
    connect_s: float = CONNECT_S
    motion_timeout_s: float = MOTION_TIMEOUT_S
    motion_start_s: float = MOTION_START_S
    poll_s: float = POLL_S
    wall: Callable[[], float] = time.time

    # ---- capture ----------------------------------------------------------------------

    def capture(self, request: SnapshotCapture) -> SavedSnapshot:
        """Take a snapshot of the scope's devices from their live streams, and keep it.

        Raises
        ------
        SnapshotError
            404: an unknown device or panel; 409: a device's stream is not
            open and ticking, or a kept role is not connected (every one
            named); 422: the scope keeps no role.
        """
        names = self._scope_devices(request)
        rows: list[SnapshotRow] = []
        problems: list[str] = []
        for name in names:
            device = self.streams.devices[name]
            for axis in device.channel_axes:
                for role in snapshot_roles(device):
                    try:
                        hub = self.puts.live(device, role, axis)
                    except Denied as exc:
                        problems.append(str(exc))
                        continue
                    row = self._row(hub.feed, device, role, axis)
                    if row is None:
                        problems.append(f"{_where(name, axis, role.name)}: not read")
                    else:
                        rows.append(row)
        if problems:
            raise SnapshotError(
                "cannot snapshot: " + "; ".join(dict.fromkeys(problems)), 409
            )
        if not rows:
            raise SnapshotError("the scope's devices have no role to keep", 422)
        snapshot = Snapshot(
            name=request.name,
            created=self.wall(),
            author=request.operator,
            note=request.note,
            golden=request.golden,
            scope=request.scope,
            rows=rows,
        )
        return SavedSnapshot(id=self.store.save(snapshot), snapshot=snapshot)

    def _scope_devices(self, request: SnapshotCapture) -> list[str]:
        scope = request.scope
        if scope.panel is not None:
            names = self.panel_devices(scope.panel)
            if names is None:
                raise SnapshotError(f"no composed panel {scope.panel!r}", 404)
        else:
            names = list(scope.devices or [])
        unknown = [n for n in names if n not in self.streams.devices]
        if unknown:
            raise SnapshotError(f"no device named {', '.join(unknown)}", 404)
        return names

    @staticmethod
    def _row(
        feed: Feed, device: BridgeDevice, role: Role, axis: str
    ) -> Optional[SnapshotRow]:
        value = _kept(feed.value(role.name, axis))
        if value is None:
            return None
        readback = role.readback
        return SnapshotRow(
            device=device.name,
            axis=axis or None,
            role=role.name,
            setpoint=value,
            readback=(
                _kept(feed.value(readback, axis))
                if readback is not None and device.has_role(readback)
                else None
            ),
        )

    # ---- reads ------------------------------------------------------------------------

    def get(self, snapshot_id: str) -> SavedSnapshot:
        """The snapshot of that id (:class:`SnapshotError` 404 if none)."""
        snapshot = self.store.get(snapshot_id)
        if snapshot is None:
            raise SnapshotError(f"no snapshot {snapshot_id!r}", 404)
        return SavedSnapshot(id=snapshot_id, snapshot=snapshot)

    def summaries(self) -> list[SnapshotSummary]:
        """Every snapshot's summary, newest first."""
        return [summary(i, s) for i, s in self.store.all()]

    # ---- compare ----------------------------------------------------------------------

    async def diff(self, snapshot_id: str) -> SnapshotDiff:
        """Each row of the snapshot against live: within tolerance, differs, or unreachable."""
        snapshot = self.get(snapshot_id).snapshot
        await self._connect(snapshot.rows)
        return self._compare(snapshot_id, snapshot)

    async def active(self) -> list[SnapshotSummary]:
        """The snapshots whose every row is within tolerance of live, newest first."""
        found = self.store.all()
        await self._connect([row for _, s in found for row in s.rows])
        return [summary(i, s) for i, s in found if self._compare(i, s).within]

    async def _connect(self, rows: list[SnapshotRow]) -> None:
        """Open the rows' devices' feeds and wait (at most ``connect_s``) for their records."""
        wanted: list[tuple[Feed, str, str]] = []
        for row in rows:
            device = self.streams.devices.get(row.device)
            if device is None or not device.has_role(row.role):
                continue
            feed = self.streams.hub(row.device).feed
            wanted.append((feed, row.role, row.axis or ""))
        deadline = asyncio.get_running_loop().time() + self.connect_s
        while not all(f.connected(r, a) for f, r, a in wanted):
            if asyncio.get_running_loop().time() >= deadline:
                return
            await asyncio.sleep(self.poll_s)

    def _compare(self, snapshot_id: str, snapshot: Snapshot) -> SnapshotDiff:
        rows = [self._row_diff(row) for row in snapshot.rows]
        return SnapshotDiff(
            id=snapshot_id,
            server_time=self.wall(),
            within=all(r.state == "within" for r in rows),
            rows=rows,
            unreachable=list(
                dict.fromkeys(r.device for r in rows if r.state == "unreachable")
            ),
        )

    def _row_diff(self, row: SnapshotRow) -> RowDiff:
        base = {
            "device": row.device,
            "axis": row.axis,
            "role": row.role,
            "saved": row.setpoint,
            "readback_saved": row.readback,
        }
        device = self.streams.devices.get(row.device)
        if device is None or not device.has_role(row.role):
            why = "not served" if device is None else f"has no role {row.role!r}"
            return RowDiff(**base, state="unreachable", message=why)
        hub = self.streams.hubs.get(row.device)
        axis = row.axis or ""
        role = device.role(row.role)
        if hub is None or not hub.feed.connected(role.name, axis):
            return RowDiff(**base, state="unreachable", message="not connected")
        feed = hub.feed
        reading = feed.reading(role, axis, hub.stale_after)
        tol = tolerance(role, reading.fields.get("PREC"))
        live = reading.value
        readback = role.readback
        return RowDiff(
            **base,
            live=live,
            readback_live=(
                _kept(feed.value(readback, axis))
                if readback is not None and device.has_role(readback)
                else None
            ),
            tolerance=tol,
            state="within" if within(row.setpoint, live, tol) else "differs",
        )

    # ---- restore ----------------------------------------------------------------------

    async def restore(
        self, snapshot_id: str, request: SnapshotRestore
    ) -> RestoreResult:
        """Write the ticked rows back through the put gates, in the snapshot's order.

        Raises
        ------
        SnapshotError
            404: no such snapshot; 422: a ticked row the snapshot has not.
        """
        snapshot = self.get(snapshot_id).snapshot
        keys = {(r.device, r.axis, r.role) for r in request.rows}
        rows = [r for r in snapshot.rows if (r.device, r.axis, r.role) in keys]
        missing = keys - {(r.device, r.axis, r.role) for r in rows}
        if missing:
            named = ", ".join(
                _where(d, a or "", r) for d, a, r in sorted(missing, key=str)
            )
            raise SnapshotError(f"the snapshot has no row {named}", 422)
        done: list[RestoreRow] = []
        stopped = False
        for row in rows:
            if stopped:
                done.append(_not_run(row))
                continue
            line = await self._restore_row(snapshot_id, row, request)
            done.append(line)
            if request.stop_on_refusal and line.status in (
                "refused",
                "denied",
                "unsettled",
            ):
                stopped = True
        return RestoreResult(
            id=snapshot_id,
            operator=request.operator,
            stopped=stopped,
            unreachable=list(
                dict.fromkeys(r.device for r in done if r.status == "unreachable")
            ),
            rows=done,
        )

    async def _restore_row(
        self, snapshot_id: str, row: SnapshotRow, request: SnapshotRestore
    ) -> RestoreRow:
        base = {
            "device": row.device,
            "axis": row.axis,
            "role": row.role,
            "value": row.setpoint,
        }
        device = self.streams.devices.get(row.device)
        if device is None or not device.has_role(row.role):
            why = "not served" if device is None else f"has no role {row.role!r}"
            return RestoreRow(**base, status="unreachable", message=why)
        role = device.role(row.role)
        motion = is_motion(role)
        put = PutRequest(
            device=row.device,
            role=row.role,
            axis=row.axis,
            value=row.setpoint,
            operator=request.operator,
            confirm=request.confirm,
        )
        try:
            status, result = await self.puts.put(
                put, snapshot=snapshot_id, hazard=motion
            )
        except UnknownDevice:  # pragma: no cover - checked above
            return RestoreRow(**base, status="unreachable", message="not served")
        if result.outcome == "denied" and status == 409:
            return RestoreRow(
                **base, status="unreachable", result=result, message=result.message
            )
        if result.outcome != "accepted":
            return RestoreRow(
                **base, status=result.outcome, result=result, message=result.message
            )
        if not motion:
            return RestoreRow(**base, status="accepted", result=result)
        settled, message = await self._settle(device, role, row)
        return RestoreRow(
            **base,
            status="accepted" if settled else "unsettled",
            result=result,
            settled=settled,
            message=message,
        )

    async def _settle(
        self, device: BridgeDevice, role: Role, row: SnapshotRow
    ) -> tuple[bool, str]:
        """Wait for a motion: its done flag down then up (or never down within ``motion_start_s``), the target kept."""
        hub = self.streams.hubs.get(device.name)
        if hub is None:  # pragma: no cover - the put gate needed it
            return False, "its stream closed"
        feed, axis, flag = hub.feed, row.axis or "", role.settled_by
        assert flag is not None
        loop = asyncio.get_running_loop()
        start = loop.time()
        moving = False
        while True:
            elapsed = loop.time() - start
            done = feed.value(flag, axis)
            if feed.connected(flag, axis) and not done:
                moving = True
            elif done and (moving or elapsed >= self.motion_start_s):
                break
            if elapsed >= self.motion_timeout_s:
                return (
                    False,
                    f"{flag} did not read done within {self.motion_timeout_s:g} s",
                )
            await asyncio.sleep(self.poll_s)
        reading = feed.reading(role, axis, hub.stale_after)
        tol = tolerance(role, reading.fields.get("PREC"))
        if not within(row.setpoint, reading.value, tol):
            return False, f"stopped at {reading.value!r}, not {row.setpoint!r}"
        return True, "settled"


def _not_run(row: SnapshotRow) -> RestoreRow:
    return RestoreRow(
        device=row.device,
        axis=row.axis,
        role=row.role,
        value=row.setpoint,
        status="not_run",
        message="not run: an earlier row was refused",
    )


def _where(device: str, axis: str, role: str) -> str:
    return f"{device}.{axis}.{role}" if axis else f"{device}.{role}"
