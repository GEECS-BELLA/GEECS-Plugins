"""Stock plans run strict GEECS scans through ``take_reading`` (phase 0, #807).

``bp.count`` / ``bp.scan`` from ``bluesky.plans`` over a
:class:`GeecsDetector` and a :class:`CaMotor`, with
:func:`geecs_per_shot` / :func:`geecs_per_step` supplying the one GEECS
difference — the fire between trigger and wait — on mock CA backends.  The
fake trigger box advances the camera's stamp **synchronously** inside the
SINGLESHOT put, the worst case for the baseline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps  # noqa: E402
import bluesky.plans as bp  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from bluesky.utils import FailedStatus  # noqa: E402
from ophyd_async.core import (  # noqa: E402
    StaticFilenameProvider,
    StaticPathProvider,
    set_mock_value,
)

from geecs_bluesky.devices.ca import CaMotor  # noqa: E402
from geecs_bluesky.devices.detector import GeecsDetector  # noqa: E402
from geecs_bluesky.devices.shot_control import ShotControl  # noqa: E402
from geecs_bluesky.exceptions import GeecsDeviceDownError  # noqa: E402
from geecs_bluesky.models.shot_control import ShotControlWrites  # noqa: E402
from geecs_bluesky.plans.strict import geecs_per_shot, geecs_per_step  # noqa: E402
from tests.ca_mock_helpers import DocCollector, connect_mock, follow_setpoint  # noqa: E402

WRITES = ShotControlWrites(
    name="test",
    states={
        "ARMED": [("DG", "Trigger.Source", "single")],
        "STANDBY": [("DG", "Trigger.Source", "edges")],
        "SINGLESHOT": [("DG", "Trigger.ExecuteSingleShot", "on")],
    },
)


class FakeBox:
    """Setter factory: the SINGLESHOT put is the shot, landing on every camera now.

    ``drop`` names (camera, attempt) pairs whose frame is lost — a dropped
    shot advances every other camera's stamp but not that one's.
    """

    def __init__(self) -> None:
        self.cameras: list[GeecsDetector] = []
        self.stamp = 1000.0
        self.fires = 0
        self.puts: list[tuple[str, str, str]] = []
        self.drop: set[tuple[str, int]] = set()

    def __call__(self, device: str, variable: str):
        box = self

        class Setter:
            async def put(self, value: str) -> None:
                box.puts.append((device, variable, value))
                if variable == "Trigger.ExecuteSingleShot":
                    box.fires += 1
                    box.stamp += 1.0
                    for cam in box.cameras:
                        if (cam.name, box.fires) in box.drop:
                            continue
                        set_mock_value(cam.acq_timestamp, box.stamp)

        return Setter()


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def box() -> FakeBox:
    return FakeBox()


@pytest.fixture
def shot_control(RE: RunEngine, box: FakeBox) -> ShotControl:
    sc = ShotControl(
        WRITES, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    RE(bps.mv(sc, "ARMED"))
    return sc


def _camera(RE: RunEngine, box: FakeBox, name: str, tmp_path: Path | None = None, **kw):
    provider = None
    if tmp_path is not None:
        provider = StaticPathProvider(
            StaticFilenameProvider("f"), tmp_path / "Scan001" / name
        )
    cam = GeecsDetector(
        name,
        ["MeanCounts"],
        experiment="TestExp",
        name=name.lower(),
        path_provider=provider,
        **kw,
    )
    connect_mock(RE, cam)
    set_mock_value(cam.acq_timestamp, box.stamp)
    box.cameras.append(cam)
    return cam


def test_count_fires_once_per_shot(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    set_mock_value(cam.meancounts, 3.0)
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([cam], num=4, per_shot=geecs_per_shot(shot_control)))
    events = col.primary_events()
    assert box.fires == 4
    assert [e["data"]["uc_cam-acq_timestamp"] for e in events] == [
        1001.0,
        1002.0,
        1003.0,
        1004.0,
    ]
    assert [e["data"]["uc_cam-meancounts"] for e in events] == [3.0] * 4
    start = col.docs["start"][0]
    assert start["detectors"] == ["uc_cam"] and start["plan_name"] == "count"
    assert shot_control.standing_state == "ARMED"


def test_scan_moves_then_fires(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    magnet = CaMotor(
        "U_S1H", "Current", experiment="TestExp", tolerance=0.01, name="u_s1h-current"
    )
    connect_mock(RE, magnet)
    follow_setpoint(magnet)
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.scan([cam], magnet, -1.0, 1.0, 5, per_step=geecs_per_step(shot_control)))
    events = col.primary_events()
    assert box.fires == 5
    assert [e["data"]["u_s1h-current-position"] for e in events] == pytest.approx(
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    )
    assert [e["data"]["uc_cam-acq_timestamp"] for e in events] == [
        1001.0 + i for i in range(5)
    ]
    assert list(col.docs["start"][0]["motors"]) == ["u_s1h-current"]


def test_two_cameras_share_one_fire(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    a = _camera(RE, box, "UC_A")
    b = _camera(RE, box, "UC_B")
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([a, b], num=3, per_shot=geecs_per_shot(shot_control)))
    assert box.fires == 3
    for event in col.primary_events():
        assert (
            event["data"]["uc_a-acq_timestamp"] == event["data"]["uc_b-acq_timestamp"]
        )


def test_dropped_frame_is_refired_and_rows_stay_one_shot(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    """Camera A misses fire 2: the shot is re-fired for everyone; B's orphan is not a row."""
    a = _camera(RE, box, "UC_A", shot_timeout=0.3)
    b = _camera(RE, box, "UC_B", shot_timeout=0.3)
    box.drop = {("uc_a", 2)}
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([a, b], num=3, per_shot=geecs_per_shot(shot_control)))
    events = col.primary_events()
    assert len(events) == 3
    assert box.fires == 4  # three shots, one refire
    stamps = [
        (e["data"]["uc_a-acq_timestamp"], e["data"]["uc_b-acq_timestamp"])
        for e in events
    ]
    assert stamps == [(1001.0, 1001.0), (1003.0, 1003.0), (1004.0, 1004.0)]


def test_refires_are_bounded(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    cam = _camera(RE, box, "UC_A", shot_timeout=0.2)
    box.drop = {("uc_a", n) for n in range(1, 10)}
    with pytest.raises(FailedStatus):
        RE(bp.count([cam], num=1, per_shot=geecs_per_shot(shot_control, max_refires=1)))
    assert box.fires == 2


def test_a_dead_device_is_not_refired(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    """CONNECTED says Disconnected → GeecsDeviceDownError, no refire burnt."""
    cam = _camera(RE, box, "UC_A", shot_timeout=0.2)
    set_mock_value(cam.connected_status, "Disconnected")
    box.drop = {("uc_a", n) for n in range(1, 10)}
    with pytest.raises(GeecsDeviceDownError, match="UC_A"):
        RE(bp.count([cam], num=1, per_shot=geecs_per_shot(shot_control)))
    assert box.fires == 1


def test_native_saving_brackets_the_run(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """save=on from the first prepare, the directory in every row, save=off at unstage."""
    (tmp_path / "Scan001").mkdir()
    cam = _camera(RE, box, "UC_Cam", tmp_path)
    saves: list[str] = []

    async def watch() -> None:
        cam.save.subscribe(
            lambda reading: saves.append(reading[cam.save.name]["value"])
        )

    asyncio.run_coroutine_threadsafe(watch(), RE._loop).result(5)
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([cam], num=2, per_shot=geecs_per_shot(shot_control)))
    directory = str(tmp_path / "Scan001" / "UC_Cam")
    assert [e["data"]["uc_cam-save_path"] for e in col.primary_events()] == [
        directory
    ] * 2
    assert Path(directory).is_dir()
    assert saves[-1] == "off" and "on" in saves
    assert (
        asyncio.run_coroutine_threadsafe(cam.save.get_value(), RE._loop).result(5)
        == "off"
    )


def test_plain_count_is_refused_clearly(RE: RunEngine, box: FakeBox) -> None:
    """Without the GEECS fire a camera cannot count: the stock plan fails at prepare."""
    cam = _camera(RE, box, "UC_Cam")
    with pytest.raises(FailedStatus) as info:
        RE(bp.count([cam], num=1))
    assert "EXTERNAL_EDGE" in str(info.value)
