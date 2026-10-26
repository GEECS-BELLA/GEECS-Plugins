"""Snapshot v0: capture from live streams, diff, the active verdict, restore through the put gates."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from geecs_schemas.epics.snapshot import Snapshot

from geecs_panels.app import create_app
from geecs_panels.bridge.journal import Journal
from geecs_panels.bridge.puts import Puts
from geecs_panels.bridge.routes import Bridge
from geecs_panels.bridge.snapshots import (
    SnapshotError,
    Snapshots,
    SnapshotStore,
    snapshot_roles,
    tolerance,
)
from geecs_panels.bridge.stream import DeviceStreams
from geecs_panels.contracts import RowRef, SnapshotCapture, SnapshotRestore
from geecs_panels.registry import Device
from tests._bridge import (
    CAM,
    MOTOR,
    SUPPLY,
    Clock,
    FakeCA,
    devices,
    healthy_axis,
    healthy_supply,
)
from tests.test_bridge_puts import FakeIO

OPERATOR = "S. Barber"


class Rig:
    """The supply and the stage streamed (each hub with a listener), over fakes; the camera not."""

    def __init__(self, store: Optional[SnapshotStore] = None) -> None:
        self.ca, self.clock, self.io = FakeCA(), Clock(), FakeIO()
        self.streams = DeviceStreams(
            devices(), self.ca.subscribe, clock=self.clock.mono, wall=self.clock.wall
        )
        for name in ("PS1", "Stage"):
            self.streams.hub(name, run=False).listeners.add(asyncio.Queue())
        healthy_supply(self.ca)
        healthy_axis(self.ca, "m1")
        healthy_axis(self.ca, "m2")
        self.tick()
        self.journal = Journal(wall=self.clock.wall, warn=False)
        self.puts = Puts(
            self.streams,
            self.io,
            self.journal,
            settle_s=0.3,
            poll_s=0.01,
            clock=self.clock.mono,
        )
        self.snapshots = Snapshots(
            self.streams,
            self.puts,
            store or SnapshotStore(warn=False),
            panel_devices=lambda name: ["PS1"] if name == "chicane" else None,
            connect_s=0.1,
            motion_timeout_s=1.0,
            motion_start_s=0.2,
            poll_s=0.01,
            wall=self.clock.wall,
        )

    def tick(self) -> None:
        for hub in self.streams.hubs.values():
            hub.tick()

    def capture(self, scope: Optional[dict] = None, name: str = "tuned") -> str:
        request = SnapshotCapture(
            name=name,
            operator=OPERATOR,
            scope=scope or {"devices": ["PS1", "Stage"]},
        )
        return self.snapshots.capture(request).id

    def restore(self, snapshot_id: str, rows: list[tuple], **kw: object):
        request = SnapshotRestore(
            rows=[RowRef(device=d, axis=a, role=r) for d, a, r in rows],
            operator=OPERATOR,
            **{"stop_on_refusal": False, **kw},
        )
        return asyncio.run(self.snapshots.restore(snapshot_id, request))


# ---- capture -------------------------------------------------------------------------------


def test_a_capture_keeps_every_writable_setting_with_its_readback() -> None:
    rig = Rig()
    snapshot = rig.snapshots.get(rig.capture()).snapshot
    keys = [(r.device, r.axis, r.role) for r in snapshot.rows]
    assert keys == [
        ("PS1", None, "current_sp"),
        ("PS1", None, "output_sp"),
        *[("Stage", a, r) for a in ("m1", "m2") for r in ("val", "velo", "twv")],
    ]
    current = snapshot.rows[0]
    assert current.setpoint == 0.0 and current.readback == 0.0
    val = snapshot.rows[2]
    assert val.setpoint == 1.0 and val.readback == 1.0  # .RBV beside .VAL
    assert snapshot.rows[3].readback is None  # velo has no readback
    assert snapshot.author == OPERATOR and snapshot.created == rig.clock.wall()


def test_a_start_and_a_command_are_never_kept() -> None:
    cam = devices()["Cam"]
    kept = {r.name for r in snapshot_roles(cam)}
    assert "acquire" not in kept  # settled by itself: a run, not a setting
    assert {"exposure", "trigger_mode", "crosshair1_x"} <= kept
    assert all(r.type in ("setpoint", "enum", "bit") for r in snapshot_roles(cam))


def test_a_role_the_device_does_not_permit_is_not_kept() -> None:
    allowed = devices(
        {"PS1": {"family": "fastps", "prefix": SUPPLY, "allow": ["current"]}}
    )
    assert [r.name for r in snapshot_roles(allowed["PS1"])] == ["current_sp"]


def test_a_device_whose_stream_is_not_live_refuses_the_capture() -> None:
    rig = Rig()
    with pytest.raises(SnapshotError) as exc:
        rig.capture({"devices": ["PS1", "Cam"]})
    assert exc.value.status == 409 and "no stream of Cam is open" in str(exc.value)
    rig.ca.drop(SUPPLY + "current_ramp:SP")
    with pytest.raises(SnapshotError, match="current_sp on PS1 is not connected"):
        rig.capture({"devices": ["PS1"]})
    assert rig.snapshots.summaries() == []


def test_a_panel_scope_takes_the_panels_devices() -> None:
    rig = Rig()
    snapshot = rig.snapshots.get(rig.capture({"panel": "chicane"})).snapshot
    assert {r.device for r in snapshot.rows} == {"PS1"}
    assert snapshot.scope.panel == "chicane"
    with pytest.raises(SnapshotError) as exc:
        rig.capture({"panel": "nope"})
    assert exc.value.status == 404


# ---- diff and the active verdict -------------------------------------------------------------


def test_a_diff_says_within_differs_and_unreachable() -> None:
    rig = Rig()
    sid = rig.capture()
    rig.ca.post(SUPPLY + "current_ramp:SP", 0.0004)  # PREC 3: within half a digit
    rig.ca.post(MOTOR + "m1.VELO", 1.5)
    rig.ca.drop(MOTOR + "m2.VAL")
    diff = asyncio.run(rig.snapshots.diff(sid))
    rows = {(r.device, r.axis, r.role): r for r in diff.rows}
    current = rows[("PS1", None, "current_sp")]
    assert current.state == "within" and current.tolerance == pytest.approx(0.0005)
    assert current.live == 0.0004 and current.readback_live == 0.0
    velo = rows[("Stage", "m1", "velo")]
    assert velo.state == "differs" and velo.saved == 1.0 and velo.live == 1.5
    assert rows[("Stage", "m2", "val")].state == "unreachable"
    assert rows[("Stage", "m1", "val")].state == "within"
    assert not diff.within and diff.unreachable == ["Stage"]


def test_the_active_snapshots_are_those_every_row_of_which_is_within() -> None:
    rig = Rig()
    first = rig.capture(name="first")
    rig.ca.post(SUPPLY + "current_ramp:SP", 0.5)
    rig.clock.advance(1.0)
    second = rig.capture(name="second")
    active = asyncio.run(rig.snapshots.active())
    assert [s.id for s in active] == [second]
    rig.ca.post(SUPPLY + "current_ramp:SP", 0.0)
    assert [s.id for s in asyncio.run(rig.snapshots.active())] == [first]


def test_tolerance_is_the_familys_else_half_the_last_digit_else_exact() -> None:
    role = devices()["PS1"].role("current_sp")
    assert tolerance(role, 2) == pytest.approx(0.005)
    assert tolerance(role, None) == 0.0
    assert tolerance(role.model_copy(update={"tolerance": 0.1}), 2) == 0.1


# ---- restore -----------------------------------------------------------------------------------


def test_a_restore_skips_an_unreachable_device_and_continues() -> None:
    rig = Rig()
    sid = rig.capture()
    rig.ca.post(SUPPLY + "current_ramp:SP", 0.5)
    rig.streams.hubs["Stage"].listeners.clear()  # its page closed
    result = rig.restore(
        sid,
        [("Stage", "m1", "velo"), ("PS1", None, "current_sp")],
        stop_on_refusal=True,
    )
    supply, stage = result.rows  # the snapshot's order, not the ticks'
    assert stage.status == "unreachable" and "no stream of Stage" in stage.message
    assert supply.status == "accepted"
    assert not result.stopped and result.unreachable == ["Stage"]
    assert (SUPPLY + "current_ramp:SP", 0.0, True) in rig.io.puts
    # every row's put is journaled with the snapshot's id
    assert [(e.role, e.outcome, e.snapshot) for e in rig.journal.last(10)] == [
        ("velo", "denied", sid),
        ("current_sp", "accepted", sid),
    ]


def test_a_restore_stops_on_the_first_refusal_when_asked() -> None:
    rig = Rig()
    rig.ca.post(SUPPLY + "current_ramp:SP", 0.9)
    rig.tick()
    sid = rig.capture()
    rig.ca.post(SUPPLY + "current_ramp:SP.DRVH", 0.5)  # 0.9 is now past the drive limit
    rig.tick()
    rows = [("PS1", None, "current_sp"), ("Stage", "m1", "velo")]
    stopped = rig.restore(sid, rows, stop_on_refusal=True)
    assert [r.status for r in stopped.rows] == ["refused", "not_run"]
    assert stopped.stopped and stopped.rows[1].result is None
    assert rig.io.puts == []
    going = rig.restore(sid, rows, stop_on_refusal=False)
    assert [r.status for r in going.rows] == ["refused", "accepted"]


def test_a_hazard_row_without_the_confirm_is_denied() -> None:
    rig = Rig()
    sid = rig.capture()
    result = rig.restore(sid, [("PS1", None, "output_sp")])
    (row,) = result.rows
    assert row.status == "denied" and "hazard confirm" in row.message
    assert row.result is not None and row.result.outcome == "denied"
    assert rig.journal.last(1)[0].snapshot == sid
    assert rig.restore(sid, [("PS1", None, "output_sp")], confirm=True).rows[
        0
    ].status == ("accepted")


def _moves(rig: Rig, to: float, stop_at: Optional[float] = None):
    """The record's posts for a commanded move: DMOV down, the target, then DMOV up."""

    async def run() -> None:
        while not any(pv.endswith("m1.VAL") for pv, _, _ in rig.io.puts):
            await asyncio.sleep(0.005)
        rig.ca.post(MOTOR + "m1.DMOV", 0)
        rig.ca.post(MOTOR + "m1.VAL", to)
        await asyncio.sleep(0.05)
        if stop_at is not None:  # a Stop: the record moves VAL to where it stopped
            rig.ca.post(MOTOR + "m1.VAL", stop_at)
        rig.ca.post(MOTOR + "m1.DMOV", 1)
        rig.events.append("done")

    return run


def _restore_with(rig: Rig, sid: str, mover) -> object:
    async def run() -> object:
        task = asyncio.create_task(mover())
        request = SnapshotRestore(
            rows=[RowRef(device="Stage", axis="m1", role="val")],
            operator=OPERATOR,
            confirm=True,
            stop_on_refusal=True,
        )
        result = await rig.snapshots.restore(sid, request)
        rig.events.append("answered")
        await task
        return result

    return asyncio.run(run())


def test_a_motion_row_is_waited_on_until_it_settles() -> None:
    rig = Rig()
    sid = rig.capture()
    rig.ca.post(MOTOR + "m1.VAL", 4.0)  # moved since
    rig.tick()
    rig.events = []
    result = _restore_with(rig, sid, _moves(rig, 1.0))
    (row,) = result.rows
    assert row.status == "accepted" and row.settled is True
    assert rig.events == ["done", "answered"]  # answered only after DMOV
    assert (MOTOR + "m1.VAL", 1.0, False) in rig.io.puts  # dispatched without waiting


def test_a_motion_stopped_short_is_unsettled() -> None:
    rig = Rig()
    sid = rig.capture()
    rig.ca.post(MOTOR + "m1.VAL", 4.0)
    rig.tick()
    rig.events = []
    (row,) = _restore_with(rig, sid, _moves(rig, 1.0, stop_at=2.5)).rows
    assert row.status == "unsettled" and row.settled is False
    assert "stopped at 2.5" in row.message


def test_a_motion_row_needs_the_confirm() -> None:
    rig = Rig()
    sid = rig.capture()
    (row,) = rig.restore(sid, [("Stage", "m1", "val")]).rows
    assert row.status == "denied" and "confirm" in row.message
    assert rig.io.puts == []


def test_a_restore_names_rows_the_snapshot_has_not() -> None:
    rig = Rig()
    sid = rig.capture()
    with pytest.raises(SnapshotError) as exc:
        rig.restore(sid, [("PS1", None, "loop_mode")])
    assert exc.value.status == 422 and "PS1.loop_mode" in str(exc.value)


# ---- the store and the journal -------------------------------------------------------------------


def test_the_store_keeps_one_json_file_per_snapshot(tmp_path: Path) -> None:
    rig = Rig(SnapshotStore(tmp_path))
    sid = rig.capture()
    assert [p.name for p in tmp_path.iterdir()] == [f"{sid}.json"]
    again = SnapshotStore(tmp_path)  # a restarted bridge reads them back
    assert again.get(sid) == rig.snapshots.get(sid).snapshot
    assert Snapshot.model_validate_json((tmp_path / f"{sid}.json").read_text())
    for bad in ("../etc", "active", sid + ".json", ""):
        assert again.get(bad) is None
    (tmp_path / "s1-00000000.json").write_text("{not json")
    assert [i for i, _ in again.all()] == [sid]  # an unreadable file is left out


def test_an_older_journal_gains_the_snapshot_column(tmp_path: Path) -> None:
    path = tmp_path / "journal.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE puts (id INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL,"
        " operator TEXT NOT NULL, device TEXT NOT NULL, role TEXT NOT NULL, axis TEXT,"
        " value TEXT, outcome TEXT NOT NULL, severity TEXT, message TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO puts (time, operator, device, role, value, outcome, message)"
        " VALUES (1.0, 'a', 'PS1', 'current_sp', '0.5', 'accepted', '')"
    )
    db.commit()
    db.close()
    journal = Journal(path)
    journal.record(
        operator="b", device="PS1", role="current_sp", axis=None, asked=0.1,
        outcome="accepted", snapshot="s1-00000000",
    )  # fmt: skip
    new, old = journal.last(2)
    assert new.snapshot == "s1-00000000" and old.snapshot is None


# ---- the routes ------------------------------------------------------------------------------------


def test_the_snapshot_routes() -> None:
    rig = Rig()
    bridge = Bridge(
        "expt",
        list(rig.streams.devices.values()),
        [],
        rig.streams,
        puts=rig.puts,
        snapshots=rig.snapshots,
    )
    camera = Device("Cam", "aravis_camera", CAM)
    with TestClient(create_app("expt", [camera], object(), bridge=bridge)) as client:
        body = {"name": "tuned", "operator": OPERATOR, "scope": {"devices": ["PS1"]}}
        saved = client.post("/api/snapshots", json=body).json()
        sid = saved["id"]
        assert saved["snapshot"]["rows"][0]["role"] == "current_sp"
        listing = client.get("/api/snapshots").json()
        assert [s["id"] for s in listing] == [sid] and listing[0]["rows"] == 2
        assert client.get(f"/api/snapshots/{sid}").json() == saved
        assert [s["id"] for s in client.get("/api/snapshots/active").json()] == [sid]
        assert client.get(f"/api/snapshots/{sid}/diff").json()["within"] is True
        restore = {
            "rows": [{"device": "PS1", "role": "current_sp"}],
            "operator": OPERATOR,
            "stop_on_refusal": True,
        }
        done = client.post(f"/api/snapshots/{sid}/restore", json=restore).json()
        assert done["rows"][0]["status"] == "accepted"
        assert client.get("/api/journal").json()[0]["snapshot"] == sid
        assert client.get("/api/snapshots/s0-00000000").status_code == 404
        cam = {**body, "scope": {"devices": ["Cam"]}}
        assert client.post("/api/snapshots", json=cam).status_code == 409
        del restore["stop_on_refusal"]  # chosen before the run: no default
        assert (
            client.post(f"/api/snapshots/{sid}/restore", json=restore).status_code
            == 422
        )
