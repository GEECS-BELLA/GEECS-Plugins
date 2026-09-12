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
import math
from pathlib import Path
from typing import Any

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
from geecs_bluesky.exceptions import (  # noqa: E402
    GeecsDeviceDownError,
    GeecsTriggerTimeoutError,
)
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
        self.counts: dict[str, int] = {}  # plugin-backed cameras' frame counts

    def __call__(self, device: str, variable: str):
        box = self

        class Setter:
            async def put(self, value: str) -> None:
                box.puts.append((device, variable, value))
                if variable == "Trigger.ExecuteSingleShot":
                    # The real put takes ~100 ms and the frame lands ≥ 1 s
                    # later; the trigger coroutine's count baseline (a PVA
                    # get) is long done by then.  Give it the loop once.
                    await asyncio.sleep(0.02)
                    box.fires += 1
                    box.stamp += 1.0
                    for cam in box.cameras:
                        if (cam.name, box.fires) in box.drop:
                            continue
                        hdf = getattr(cam, "hdf", None)
                        if hdf is not None:  # a plugin-backed camera counts first
                            box.counts[cam.name] = box.counts.get(cam.name, 0) + 1
                            set_mock_value(hdf.num_captured, box.counts[cam.name])
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


def _camera(
    RE: RunEngine,
    box: FakeBox,
    name: str,
    tmp_path: Path | None = None,
    provider: Any = None,
    **kw,
):
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


def test_missed_frame_keeps_the_row_and_adds_a_shot(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    """Camera A misses fire 2: the row is kept with A empty, and one more shot is taken.

    Design §2.1 (Sam, 2026-09-11): B's frame from shot 2 is data, not an
    orphan; A's columns for that row read NaN (its monitor cache would
    otherwise carry shot 1's values); the step then gets its complete row.
    """
    a = _camera(RE, box, "UC_A", shot_timeout=0.3)
    b = _camera(RE, box, "UC_B", shot_timeout=0.3)
    set_mock_value(a.meancounts, 7.0)
    box.drop = {("uc_a", 2)}
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([a, b], num=3, per_shot=geecs_per_shot(shot_control)))
    events = col.primary_events()
    assert len(events) == 4  # three complete rows and the partial one
    assert box.fires == 4
    stamps = [
        (e["data"]["uc_a-acq_timestamp"], e["data"]["uc_b-acq_timestamp"])
        for e in events
    ]
    assert stamps[0] == (1001.0, 1001.0)
    assert math.isnan(stamps[1][0]) and stamps[1][1] == 1002.0
    assert stamps[2:] == [(1003.0, 1003.0), (1004.0, 1004.0)]
    counts = [e["data"]["uc_a-meancounts"] for e in events]
    assert counts[0] == 7.0 and math.isnan(counts[1]) and counts[2] == 7.0


def test_extra_shots_are_bounded(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    """A device that never delivers: the partial rows stay, the step fails loudly."""
    cam = _camera(RE, box, "UC_A", shot_timeout=0.2)
    box.drop = {("uc_a", n) for n in range(1, 10)}
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(GeecsTriggerTimeoutError, match="no complete row after 2"):
        RE(bp.count([cam], num=1, per_shot=geecs_per_shot(shot_control, max_refires=1)))
    assert box.fires == 2
    assert len(col.primary_events()) == 2  # both partial rows are data


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
    assert [e["data"]["uc_cam-nonscalar_save_path"] for e in col.primary_events()] == [
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


def test_a_failed_fire_is_not_a_dropped_frame(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    """Codex review of #811: a refused SINGLESHOT put re-raises; no refire, no extra shot."""
    cam = _camera(RE, box, "UC_Cam", shot_timeout=0.5)

    class RefusingBox(FakeBox):
        def __call__(self, device: str, variable: str):
            inner = super().__call__(device, variable)
            outer = self

            class Setter:
                async def put(self, value: str) -> None:
                    if variable == "Trigger.ExecuteSingleShot":
                        outer.fires += 1
                        raise RuntimeError("GEECS refused the set")
                    await inner.put(value)

            return Setter()

    refusing = RefusingBox()
    sc = ShotControl(WRITES, experiment="TestExp", name="sc2", setter_factory=refusing)
    connect_mock(RE, sc)
    RE(bps.mv(sc, "ARMED"))
    with pytest.raises(FailedStatus) as info:
        RE(bp.count([cam], num=1, per_shot=geecs_per_shot(sc, max_refires=2)))
    assert isinstance(info.value.__cause__, RuntimeError)
    assert refusing.fires == 1  # no refire on a failed fire


def _plugin_camera(RE: RunEngine, box: FakeBox, name: str, tmp_path: Path, **kw):
    """A plugin-backed camera on mocks; the box advances its count on every fire."""
    from ophyd_async.core import callback_on_mock_put

    provider = StaticPathProvider(
        StaticFilenameProvider(name), tmp_path / "Scan001" / name
    )
    cam = GeecsDetector(
        name,
        ["MeanCounts"],
        experiment="TestExp",
        name=name.lower(),
        hdf_plugins=[("image", provider)],
        **kw,
    )
    connect_mock(RE, cam)
    set_mock_value(cam.acq_timestamp, box.stamp)
    set_mock_value(cam.hdf.file_path_exists, True)
    set_mock_value(cam.hdf.data_type, "UInt16")
    set_mock_value(cam.hdf.color_mode, "Mono")
    rewinds: list[int] = []

    def plugin_rewinds(value, **_) -> None:
        rewinds.append(value)
        box.counts[cam.name] = value
        set_mock_value(cam.hdf.num_captured, value)

    callback_on_mock_put(cam.hdf.rewind, plugin_rewinds)
    box.cameras.append(cam)
    return cam, rewinds


def test_two_plugin_cameras_in_one_run_keep_distinct_attribute_keys(
    RE: RunEngine, box: FakeBox, tmp_path: Path
) -> None:
    """The plugin's attribute keys carry the device, so two cameras coexist (#829).

    The XML here is the shape ``geecs_pva_gateway.file_plugin.attributes_xml``
    serves: ``<ophyd name>-<suffix>`` under the shared naming contract.
    """
    a, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    b, _ = _plugin_camera(RE, box, "UC_B", tmp_path)
    for cam in (a, b):
        set_mock_value(
            cam.hdf.nd_attributes_file,
            "<Attributes>"
            f'<Attribute name="{cam.name}-acq_timestamp" type="PARAM" '
            f'source="{cam.name}-acq_timestamp" datatype="DOUBLE" description="s"/>'
            f'<Attribute name="{cam.name}-recv_timestamp" type="PARAM" '
            f'source="{cam.name}-recv_timestamp" datatype="DOUBLE" description="r"/>'
            "</Attributes>",
        )
    sc = ShotControl(WRITES, experiment="TestExp", name="htu", setter_factory=box)
    connect_mock(RE, sc)
    docs = DocCollector()
    RE.subscribe(docs)
    RE(bp.count([a, b], 2, per_shot=geecs_per_shot(sc)))
    assert docs.docs["stop"][-1]["exit_status"] == "success"
    keys = set(docs.docs["descriptor"][0]["data_keys"])
    assert {"uc_a-acq_timestamp", "uc_b-acq_timestamp", "uc_a", "uc_b"} <= keys
    assert sorted(d["data_key"] for d in docs.docs["stream_resource"]) == [
        "uc_a",
        "uc_a-acq_timestamp",
        "uc_a-recv_timestamp",
        "uc_b",
        "uc_b-acq_timestamp",
        "uc_b-recv_timestamp",
    ]


def test_missed_frame_on_plugin_cameras_rewinds_the_partial_row(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """A partial row keeps its scalars and no frames: every plugin camera rewinds to its last datum.

    The bundler wants one same-width datum per external key per event, so
    B's frame from the shot A missed cannot be referenced by that row; the
    rewind (the late-frame guard for A) drops B's uncollected frame too.
    """
    a, a_rewinds = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=0.3)
    b, b_rewinds = _plugin_camera(RE, box, "UC_B", tmp_path, shot_timeout=0.3)
    box.drop = {("uc_a", 2)}
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([a, b], num=3, per_shot=geecs_per_shot(shot_control)))
    events = col.primary_events()
    assert len(events) == 4 and box.fires == 4
    # Both rewind to 1 (the frame row 1 referenced) before the retake.
    assert a_rewinds == [1] and b_rewinds == [1]
    datums = [d for d in col.docs["stream_datum"]]
    by_key: dict[str, list[dict]] = {}
    resources = {r["uid"]: r["data_key"] for r in col.docs["stream_resource"]}
    for d in datums:
        by_key.setdefault(resources[d["stream_resource"]], []).append(d["indices"])
    assert by_key["uc_a"] == [
        {"start": 0, "stop": 1},
        {"start": 1, "stop": 2},
        {"start": 2, "stop": 3},
    ]
    assert by_key["uc_b"] == by_key["uc_a"]
    assert math.isnan(events[1]["data"]["uc_a-acq_timestamp"])
    assert events[1]["data"]["uc_b-acq_timestamp"] == 1002.0
