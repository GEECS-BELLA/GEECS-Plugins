"""BackgroundSnapshot — the run's soft background telemetry (#1016, #929).

The object alone: probe, exclusion, the soft read, the lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import math
from functools import partial

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from ophyd_async.core import NotConnectedError, set_mock_value  # noqa: E402

from geecs_bluesky.devices.background import (  # noqa: E402
    BackgroundSnapshot,
    _blank,
    _invalid,
)
from geecs_bluesky.devices.ca import CaMotor, CaSnapshotReadable  # noqa: E402
from geecs_bluesky.devices.detector import GeecsDetector  # noqa: E402
from tests.ca_mock_helpers import DocCollector, connect_mock  # noqa: E402


class Fake:
    """A member with scripted answers: it connects, stages, reads as told."""

    parent = None

    def __init__(
        self,
        name: str,
        *,
        geecs: str = "U_Fake",
        connectable: bool = True,
        hang: bool = False,
        value: float = 1.0,
        severity: int = 0,
    ) -> None:
        self.name = name
        self._geecs_device_name = geecs
        self.connectable = connectable
        self.hang = hang
        self.value = value
        self.severity = severity
        self.fail_read = False
        self.staged = 0
        self.unstaged = 0
        self._mock = None  # what ``is_connected`` reads
        self._column_headers = {f"{name}-x": f"{geecs} X"}

    async def connect(self, mock=False, timeout=10.0, force_reconnect=False):
        if self.hang:
            await asyncio.sleep(timeout + 5.0)
        if not self.connectable:
            raise NotConnectedError(f"ca://{self.name}")
        self._mock = object()

    def stage(self):
        self.staged += 1

    def unstage(self):
        self.unstaged += 1

    async def describe(self):
        return {f"{self.name}-x": {"source": "fake", "dtype": "number", "shape": []}}

    async def read(self):
        if self.fail_read:
            raise RuntimeError("gone")
        return {
            f"{self.name}-x": {
                "value": self.value,
                "timestamp": 1.0,
                "alarm_severity": self.severity,
            }
        }


def _magnet(RE: RunEngine) -> CaSnapshotReadable:
    """A scalar device with a scanned settable child, as the namespace builds one."""
    magnet = CaSnapshotReadable(
        "U_S1H", ["Voltage"], experiment="TestExp", name="u_s1h"
    )
    magnet.current = CaMotor("U_S1H", "Current", experiment="TestExp", tolerance=0.01)
    magnet.add_readables([magnet.current])  # subscribed settable: its readback logs
    connect_mock(RE, magnet)
    return magnet


def _gauge(RE: RunEngine, name: str = "U_Gauge", value: float = 2.5):
    gauge = CaSnapshotReadable(
        name, ["Pressure"], experiment="TestExp", name=name.lower()
    )
    connect_mock(RE, gauge)
    set_mock_value(gauge.pressure, value)
    return gauge


def _cam(RE: RunEngine, name: str = "UC_Cam"):
    cam = GeecsDetector(name, ["MeanCounts"], experiment="TestExp", name=name.lower())
    connect_mock(RE, cam)
    set_mock_value(cam.meancounts, 7.0)
    set_mock_value(cam.acq_timestamp, 1000.0)
    return cam


def run(RE: RunEngine, coro_factory):
    """Run *coro_factory()* on the RunEngine's loop (the probe is a coroutine)."""
    return asyncio.run_coroutine_threadsafe(coro_factory(), RE._loop).result(10.0)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


# ------------------------------------------------------------- the object
def test_probe_leaves_the_runs_own_devices_to_the_run(RE) -> None:
    """Exclusion is by root device: a detector's signals, a scalar device, a motor's owner."""
    gauge = _gauge(RE)
    cam = _cam(RE)
    magnet = _magnet(RE)
    other = _gauge(RE, "U_Other", 1.0)
    candidates = [gauge, *cam._scalar_signals(), magnet, other]
    snapshot = BackgroundSnapshot(candidates, probe_timeout=0.5)
    # The plan stages roots: the camera (its signals go), the magnet whose
    # child is scanned (the whole device goes), and a view's owner.
    run(RE, lambda: snapshot.probe(staged=[cam, magnet.current, other.scalars]))
    assert snapshot.members == [gauge]
    assert snapshot.dropped == []
    keys = run(RE, snapshot.describe)
    assert set(keys) == {"u_gauge-pressure"}
    assert snapshot._column_headers == {"u_gauge-pressure": "U_Gauge Pressure"}
    # Nothing staged: everything is background, the detector's signals
    # included (read from its monitor cache: the last frame's scalars).
    run(RE, lambda: snapshot.probe(staged=[]))
    assert set(run(RE, snapshot.describe)) == {
        "u_gauge-pressure",
        "uc_cam-meancounts",
        "uc_cam-acq_timestamp",
        "u_s1h-voltage",
        "u_s1h-current-position",
        "u_other-pressure",
    }
    assert snapshot._column_headers["uc_cam-meancounts"] == "UC_Cam MeanCounts"


def test_probe_drops_what_does_not_answer_and_keeps_the_rest(RE, caplog) -> None:
    """An unserved PV and a hanging connect cost one bounded probe, never the run."""
    gauge = _gauge(RE)
    dead = Fake("u_dead", geecs="U_Dead", connectable=False)
    slow = Fake("u_slow", geecs="U_Slow", hang=True)
    snapshot = BackgroundSnapshot([dead, gauge, slow], probe_timeout=0.2)
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.background"):
        run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [gauge]
    assert snapshot.dropped == ["U_Dead", "U_Slow"]
    assert "left out of this run" in caplog.text
    assert "U_Dead (NotConnectedError: ca://u_dead)" in caplog.text
    assert "U_Slow (no answer within 0.2 s)" in caplog.text
    # A dropped member's half-staged cache is released.
    assert slow.unstaged == 1 and dead.unstaged == 1
    # Probed again at the next run: the device is back.
    dead.connectable = False
    slow.hang = False
    run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [gauge, slow] and snapshot.dropped == ["U_Dead"]


def test_probe_connects_a_member_only_once(RE) -> None:
    """A member connected already is not reconnected (a mock's callbacks would be lost)."""
    member = Fake("u_fake")
    snapshot = BackgroundSnapshot([member], probe_timeout=0.5)
    run(RE, lambda: snapshot.probe(staged=[]))
    token = member._mock
    run(RE, lambda: snapshot.probe(staged=[]))
    assert member._mock is token and member.staged == 2


def test_read_fills_every_declared_key_and_nans_the_dead(RE, caplog) -> None:
    """INVALID → NaN, MAJOR keeps its value, a failing member → NaN and one warning."""
    invalid = Fake("u_invalid", geecs="U_Invalid", value=3.0, severity=-1)
    major = Fake("u_major", geecs="U_Major", value=4.0, severity=2)
    live = Fake("u_live", geecs="U_Live", value=5.0)
    snapshot = BackgroundSnapshot([invalid, major, live], read_timeout=0.2)
    run(RE, lambda: snapshot.probe(staged=[]))
    row = run(RE, snapshot.read)
    assert math.isnan(row["u_invalid-x"]["value"])
    assert row["u_major-x"]["value"] == 4.0 and row["u_live-x"]["value"] == 5.0
    live.fail_read = True
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.background"):
        first = run(RE, snapshot.read)
        second = run(RE, snapshot.read)
    assert math.isnan(first["u_live-x"]["value"])
    assert math.isnan(second["u_live-x"]["value"])
    assert caplog.text.count("U_Live stopped answering") == 1
    assert set(second) == {"u_invalid-x", "u_major-x", "u_live-x"}


def test_read_outlasting_its_budget_is_nan_not_a_wait(RE) -> None:
    class Hanging(Fake):
        async def read(self):
            await asyncio.sleep(5.0)
            return await super().read()

    hanging = Hanging("u_hang", geecs="U_Hang")
    snapshot = BackgroundSnapshot([hanging], probe_timeout=0.2, read_timeout=0.2)
    run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == []  # the probe's first read did not land either
    snapshot._active = [hanging]  # as if it had: the per-shot backstop
    snapshot._datakeys = {id(hanging): {"u_hang-x": {"dtype": "number"}}}
    row = run(RE, snapshot.read)
    assert math.isnan(row["u_hang-x"]["value"])


def test_blank_and_invalid_rules() -> None:
    assert math.isnan(_blank({"dtype": "number"})["value"])
    assert math.isnan(_blank({"dtype": "integer"})["value"])
    assert _blank({"dtype": "string"})["value"] == ""
    assert _blank({"dtype": "boolean"})["value"] is False
    assert not _invalid({"value": 1.0, "timestamp": 0.0})  # no severity: live
    assert not _invalid({"value": 1.0, "timestamp": 0.0, "alarm_severity": 2})
    assert _invalid({"value": 1.0, "timestamp": 0.0, "alarm_severity": -1})
    assert _invalid({"value": 1.0, "timestamp": 0.0, "alarm_severity": 3})


def test_stage_resets_and_unstage_releases_the_members(RE) -> None:
    member = Fake("u_fake")
    snapshot = BackgroundSnapshot([member])
    run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [member] and member.staged == 1
    RE(bps.unstage(snapshot, wait=True))
    assert member.unstaged == 1 and snapshot.members == []
    RE(bps.stage(snapshot, wait=True))
    assert snapshot.members == [] and run(RE, snapshot.read) == {}


def test_a_run_reads_the_snapshot_per_row(RE) -> None:
    """Through the RunEngine: stage, the probe before open_run, one read per row."""
    gauge = _gauge(RE, value=2.5)
    dead = Fake("u_dead", geecs="U_Dead", connectable=False)
    snapshot = BackgroundSnapshot([gauge, dead], probe_timeout=0.2)
    col = DocCollector()
    RE.subscribe(col)

    def plan():
        yield from bps.stage(snapshot, wait=True)
        yield from bps.wait_for([partial(snapshot.probe, [])])
        yield from bps.open_run(md={"background_dropped": snapshot.dropped})
        for value in (2.5, 3.5):
            set_mock_value(gauge.pressure, value)
            yield from bps.create("primary")
            yield from bps.read(snapshot)
            yield from bps.save()
        yield from bps.close_run()
        yield from bps.unstage(snapshot, wait=True)

    RE(plan())
    events = col.primary_events()
    assert [e["data"]["u_gauge-pressure"] for e in events] == [2.5, 3.5]
    assert col.docs["start"][0]["background_dropped"] == ["U_Dead"]
    descriptor = col.docs["descriptor"][0]
    assert set(descriptor["data_keys"]) == {"u_gauge-pressure"}
    assert descriptor["object_keys"] == {"background": ["u_gauge-pressure"]}
