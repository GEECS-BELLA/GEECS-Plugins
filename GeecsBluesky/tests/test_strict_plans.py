"""Stock plans run strict GEECS scans through the on-demand slot.

``bp.count`` / ``bp.scan`` from ``bluesky.plans`` over a
:class:`GeecsDetector` and a :class:`CaMotor`, with the on-demand hooks
(:mod:`geecs_bluesky.plans.on_demand`) and the device streams
(:class:`~geecs_bluesky.plans.streams.DeviceStreams`) bound as the
registry binds them (:func:`strict_count`, :func:`strict_scan`), on mock CA
backends.  ``primary`` is the shot table; each device records into its own
``<name>_stream``.  The fake trigger box advances the camera's stamp
**synchronously** inside the SINGLESHOT put, the worst case for the
baseline.
"""

from __future__ import annotations

import asyncio
import time
import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import (
    StaticFilenameProvider,
    StaticPathProvider,
    set_mock_value,
)

from geecs_bluesky.devices.ca import CaMotor
from geecs_bluesky.devices.detector import GeecsDetector
from geecs_bluesky.devices.shot_control import ShotControl, ShotControlWrites
from geecs_bluesky.plans.on_demand import (
    OnDemand,
    on_demand_per_shot,
    on_demand_per_step,
    start_keys,
)
from geecs_bluesky.plans.streams import DeviceStreams, is_per_shot
from tests.ca_mock_helpers import DocCollector, connect_mock, follow_setpoint

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
    shot advances every other camera's stamp but not that one's.  ``clock``
    stamps each shot with the worker's time (LabVIEW epoch) instead of a
    count from 1000, so the readers' fire-interval join can place it.
    """

    def __init__(self) -> None:
        self.cameras: list[GeecsDetector] = []
        self.stamp = 1000.0
        self.fires = 0
        self.puts: list[tuple[str, str, str]] = []
        self.drop: set[tuple[str, int]] = set()
        self.counts: dict[str, int] = {}  # plugin-backed cameras' frame counts
        self.clock = False

    def advance(self, step: float = 1.0) -> None:
        """The next shot's stamp."""
        from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET

        self.stamp = (
            time.time() + LABVIEW_EPOCH_OFFSET if self.clock else self.stamp + step
        )
        # Frames a free-running camera's plugin counts per fire beyond the shot's.
        self.extra: dict[str, int] = {}

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
                    box.advance()
                    for cam in box.cameras:
                        if (cam.name, box.fires) in box.drop:
                            continue
                        hdf = getattr(cam, "hdf", None)
                        if hdf is not None:  # a plugin-backed camera counts first
                            box.counts[cam.name] = (
                                box.counts.get(cam.name, 0)
                                + 1
                                + box.extra.get(cam.name, 0)
                            )
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


def _on_demand(sc: Any, devices: list, required: Any, scan_mode: Any, kw: dict):
    per_shot = [d for d in devices if is_per_shot(d)]
    streams = DeviceStreams(per_shot, dual_write=True)
    required = per_shot if required is None else required
    run = OnDemand(sc, streams, required, scan_mode=scan_mode, **kw)
    md = start_keys(streams, required, witnessed=False)
    return streams, run, [d for d in devices if not is_per_shot(d)], md


def strict_count(
    sc: Any, devices: list, num: int, *, required=None, scan_mode=None, md=None, **kw
):
    """The bound count's strict path over *devices* (no claim, no liveness gate)."""
    streams, run, rows, keys = _on_demand(sc, devices, required, scan_mode, kw)
    hook = on_demand_per_shot(run, scan_mode=scan_mode)
    return streams.wrap(
        bp.count(rows, num=num, per_shot=hook, md={**keys, **(md or {})})
    )


def strict_scan(
    sc: Any,
    devices: list,
    *args,
    shots_per_step: int = 1,
    required=None,
    plan: Any = bp.scan,
    md=None,
    **kw,
):
    """*plan* (``bp.scan``) over the rows with the on-demand step (*shots_per_step* slots)."""
    streams, run, rows, keys = _on_demand(sc, devices, required, None, kw)
    hook = on_demand_per_step(run, shots_per_step=shots_per_step)
    return streams.wrap(plan(rows, *args, per_step=hook, md={**keys, **(md or {})}))


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
    RE(strict_count(shot_control, [cam], 4))
    events = col.stream_events("uc_cam_stream")
    assert box.fires == 4
    assert [e["data"]["uc_cam-acq_timestamp"] for e in events] == [
        1001.0,
        1002.0,
        1003.0,
        1004.0,
    ]
    assert [e["data"]["uc_cam-meancounts"] for e in events] == [3.0] * 4
    rows = [e["data"] for e in col.primary_events()]
    assert [r["shot_index"] for r in rows] == [1, 2, 3, 4]
    assert [r["uc_cam-baseline"] for r in rows] == [1000.0, 1001.0, 1002.0, 1003.0]
    assert not any(k.startswith("uc_cam-acq") for k in rows[0])  # no device data
    start = col.docs["start"][0]
    assert start["per_shot_streams"] == ["uc_cam_stream"]
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
    RE(strict_scan(shot_control, [cam], magnet, -1.0, 1.0, 5))
    events = col.primary_events()
    assert box.fires == 5
    assert [e["data"]["u_s1h-current-position"] for e in events] == pytest.approx(
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    )
    assert [
        e["data"]["uc_cam-acq_timestamp"] for e in col.stream_events("uc_cam_stream")
    ] == [1001.0 + i for i in range(5)]
    assert list(col.docs["start"][0]["motors"]) == ["u_s1h-current"]


def test_deferred_pause_lands_between_the_shots_of_a_step(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    """The scanner's Pause in a strict step scan lands after the shot in progress.

    Five shots per step: the pause requested after the second shot lands
    long before the step's fifth, and the resume finishes both steps with
    every shot recorded once.
    """
    import threading

    from bluesky.utils import RunEngineInterrupted

    cam = _camera(RE, box, "UC_Cam")
    magnet = CaMotor(
        "U_S1H", "Current", experiment="TestExp", tolerance=0.01, name="u_s1h-current"
    )
    connect_mock(RE, magnet)
    follow_setpoint(magnet)
    col = DocCollector()
    RE.subscribe(col)

    def pause_soon() -> None:
        while box.fires < 2:
            time.sleep(0.005)
        RE.request_pause(defer=True)

    threading.Thread(target=pause_soon, daemon=True).start()
    with pytest.raises(RunEngineInterrupted):
        RE(strict_scan(shot_control, [cam], magnet, -1.0, 1.0, 2, shots_per_step=5))
    assert box.fires < 5, box.fires  # inside the first step
    RE.resume()
    assert col.docs["stop"][-1]["exit_status"] == "success"
    events = col.primary_events()
    assert box.fires == 10 and len(events) == 10
    assert [e["data"]["bin_number"] for e in events] == [1] * 5 + [2] * 5


def test_two_cameras_share_one_fire(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    a = _camera(RE, box, "UC_A")
    b = _camera(RE, box, "UC_B")
    col = DocCollector()
    RE.subscribe(col)
    RE(strict_count(shot_control, [a, b], 3))
    assert box.fires == 3
    stamps = [
        [e["data"][f"{n}-acq_timestamp"] for e in col.stream_events(f"{n}_stream")]
        for n in ("uc_a", "uc_b")
    ]
    assert stamps[0] == stamps[1] == [1001.0, 1002.0, 1003.0]


def test_missed_frame_keeps_the_row_and_adds_a_shot(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl
) -> None:
    """Camera A misses fire 2: the row is kept, flagged, and one more shot is taken.

    Sam, 2026-09-11: B's frame from shot 2 is data, not an orphan — it is
    in B's stream; A's stream simply has no record for that fire.
    """
    a = _camera(RE, box, "UC_A", shot_timeout=0.3)
    b = _camera(RE, box, "UC_B", shot_timeout=0.3)
    set_mock_value(a.meancounts, 7.0)
    box.drop = {("uc_a", 2)}
    col = DocCollector()
    RE.subscribe(col)
    RE(strict_count(shot_control, [a, b], 3))
    events = [e["data"] for e in col.primary_events()]
    assert len(events) == 4  # three complete rows and the partial one
    assert box.fires == 4
    assert [(r["complete"], r["missing"], r["attempt"]) for r in events] == [
        (True, "", 1),
        (False, "uc_a", 1),
        (True, "", 2),
        (True, "", 1),
    ]

    def stamps(name: str) -> list[float]:
        return [
            e["data"][f"{name}-acq_timestamp"]
            for e in col.stream_events(f"{name}_stream")
        ]

    assert stamps("uc_a") == [1001.0, 1003.0, 1004.0]
    assert stamps("uc_b") == [1001.0, 1002.0, 1003.0, 1004.0]
    a_counts = [e["data"]["uc_a-meancounts"] for e in col.stream_events("uc_a_stream")]
    assert a_counts == [7.0] * 3


def test_extra_shots_are_bounded_and_the_step_flagged(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, caplog
) -> None:
    """A device that never delivers: the refires stop, the rows stay, the scan continues."""
    cam = _camera(RE, box, "UC_A", shot_timeout=0.2)
    box.drop = {("uc_a", n) for n in range(1, 10)}
    col = DocCollector()
    RE.subscribe(col)
    RE(strict_count(shot_control, [cam], 1, max_refires=1))
    assert box.fires == 2
    assert [e["data"]["complete"] for e in col.primary_events()] == [False, False]
    assert col.docs["stop"][-1]["exit_status"] == "success"
    # No plugin to ask: the camera itself is the suspect.
    assert any(
        "no record from UC_A — the last row is flagged incomplete" in r.getMessage()
        and "known camera frame-drop" in r.getMessage()
        for r in caplog.records
    )


def test_a_dead_required_device_pauses_the_run_with_the_reason(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, caplog
) -> None:
    """CONNECTED says Disconnected → the run pauses naming it, no refire burnt."""
    from bluesky.utils import RunEngineInterrupted

    cam = _camera(RE, box, "UC_A", shot_timeout=0.2)
    set_mock_value(cam.connected_status, "Disconnected")
    box.drop = {("uc_a", n) for n in range(1, 10)}
    with pytest.raises(RunEngineInterrupted):
        RE(strict_count(shot_control, [cam], 1))
    assert box.fires == 1 and RE.state == "paused"
    assert any("UC_A report down" in r.getMessage() for r in caplog.records)
    RE.stop()


def test_native_saving_brackets_the_run(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """save=on from the stream's prepare, the directory in every record, save=off at unstage."""
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
    RE(strict_count(shot_control, [cam], 2))
    directory = str(tmp_path / "Scan001" / "UC_Cam")
    events = col.stream_events("uc_cam_stream")
    assert [e["data"]["uc_cam-nonscalar_save_path"] for e in events] == [directory] * 2
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


class _RefusedPut(RuntimeError):
    """A failed CA put's shape: ``aioca.CANothing`` is *falsy* when the put failed.

    Its ``repr`` is the bare error code; only ``str`` carries the PV and the
    CA message.  A truthy stand-in cannot see the ``exc.__cause__ or exc``
    bug (#817, caught on hardware 2026-09-10).
    """

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "_RefusedPut(ECA_BADTYPE)"


def test_a_failed_fire_is_not_a_dropped_frame(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, caplog
) -> None:
    """Codex review of #811: a refused SINGLESHOT put re-raises; no refire, no extra shot.

    And #817: the ERROR line names the cause by ``str`` — the PV and the CA
    message — even though a failed put is falsy.
    """
    cam = _camera(RE, box, "UC_Cam", shot_timeout=0.5)

    class RefusingBox(FakeBox):
        def __call__(self, device: str, variable: str):
            inner = super().__call__(device, variable)
            outer = self

            class Setter:
                async def put(self, value: str) -> None:
                    if variable == "Trigger.ExecuteSingleShot":
                        outer.fires += 1
                        raise _RefusedPut("TestExp:sc2:SP: GEECS refused the set")
                    await inner.put(value)

            return Setter()

    refusing = RefusingBox()
    sc = ShotControl(WRITES, experiment="TestExp", name="sc2", setter_factory=refusing)
    connect_mock(RE, sc)
    RE(bps.mv(sc, "ARMED"))
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.plans.strict"):
        with pytest.raises(FailedStatus) as info:
            RE(strict_count(sc, [cam], 1, max_refires=2))
    assert isinstance(info.value.__cause__, _RefusedPut)
    assert refusing.fires == 1  # no refire on a failed fire
    errors = [
        r.getMessage()
        for r in caplog.records
        if r.name == "geecs_bluesky.plans.strict" and r.levelno == logging.ERROR
    ]
    assert len(errors) == 1
    assert "_RefusedPut: TestExp:sc2:SP: GEECS refused the set" in errors[0]
    assert "ECA_BADTYPE" not in errors[0]  # str, not repr


def _attributes_xml(ophyd_name: str, *names: str) -> str:
    """The ``NDAttributesFile`` shape the gateway's file plugin serves.

    ``<device>-hdf-<variable>-frame_acq_timestamp`` / ``-frame_recv_timestamp``
    (``geecs_pva_gateway.file_plugin.attributes_xml``): unique per camera and
    never an event column's name.  Explicit *names* override, to pin what the
    bare names of #829 did.
    """
    names = names or (
        f"{ophyd_name}-hdf-image-frame_acq_timestamp",
        f"{ophyd_name}-hdf-image-frame_recv_timestamp",
    )
    return (
        "<Attributes>"
        + "".join(
            f'<Attribute name="{n}" type="PARAM" source="{n}" datatype="DOUBLE" '
            'description="x"/>'
            for n in names
        )
        + "</Attributes>"
    )


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
    set_mock_value(cam.hdf.nd_attributes_file, _attributes_xml(cam.name))
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
    """The plugin's attribute keys are unique per camera and disjoint from its columns (#829).

    Pinned through the consumer that broke: bluesky's ``RunNormalizer`` (the
    front of ``TiledWriter``) must keep the camera's own CA stamp column
    ``<name>-acq_timestamp`` as an internal event column and emit the stop
    document — a stream key of the same name overwrote the column and made
    it raise "Cannot emit StreamDatum … Datum document is missing".
    """
    from bluesky.callbacks.tiled_writer import RunNormalizer

    a, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    b, _ = _plugin_camera(RE, box, "UC_B", tmp_path)
    sc = ShotControl(WRITES, experiment="TestExp", name="htu", setter_factory=box)
    connect_mock(RE, sc)
    docs = DocCollector()
    RE.subscribe(docs)
    RE(strict_count(sc, [a, b], 2))
    assert docs.docs["stop"][-1]["exit_status"] == "success"
    for cam in (a, b):
        descriptor = next(
            d for d in docs.docs["descriptor"] if d["name"] == f"{cam.name}_stream"
        )
        data_keys = descriptor["data_keys"]
        assert data_keys[f"{cam.name}-hdf-image-frame_acq_timestamp"]["external"]
    assert sorted(d["data_key"] for d in docs.docs["stream_resource"]) == [
        "uc_a",
        "uc_a-hdf-image-frame_acq_timestamp",
        "uc_a-hdf-image-frame_recv_timestamp",
        "uc_b",
        "uc_b-hdf-image-frame_acq_timestamp",
        "uc_b-hdf-image-frame_recv_timestamp",
    ]

    class Collecting(RunNormalizer):
        def __init__(self) -> None:
            super().__init__()
            self.emitted: list[tuple[str, dict]] = []

        def emit(self, name, doc):
            self.emitted.append((str(getattr(name, "value", name)), doc))

    normalizer = Collecting()
    for name, doc in docs.ordered:
        normalizer(name, doc)
    names = [n for n, _ in normalizer.emitted]
    assert names[-1] == "stop"
    # Two shots, three keys per camera (the frame, two attributes), two cameras.
    assert names.count("stream_datum") == 12


def test_first_strict_shot_zeroes_a_plugins_stale_count(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """GEECS-Plugins#853, found on hardware (2b A8): the plugin's NumCaptured still
    reads the previous session's total at the next arm; the first shot must
    not wait for N+1 while its frame posts 1."""
    cam, rewinds = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=0.5)
    box.counts["uc_a"] = 5
    set_mock_value(cam.hdf.num_captured, 5)  # stale, from a closed session
    col = DocCollector()
    RE.subscribe(col)
    RE(strict_count(shot_control, [cam], 3))
    assert box.fires == 3  # no refire: the first shot counted from 0
    assert rewinds == [0]  # zeroed at the first arm, never rewound after
    assert col.datums("uc_a") == [
        {"start": 0, "stop": 1},
        {"start": 1, "stop": 2},
        {"start": 2, "stop": 3},
    ]
    assert (
        cam.count_zeroed is False
    )  # cleared by the unstage: the next run zeroes again


def test_no_frame_from_a_plugin_camera_names_the_plugins_reason(
    RE: RunEngine,
    box: FakeBox,
    shot_control: ShotControl,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """GEECS-Plugins#1023: a plugin refusing every frame is its WriteMessage, not frame-drop intermittency.

    The camera pushes every shot; the plugin drops each as a frame of
    another shape than the one it declared at the arm, so the count never
    advances and the device yields no frame.  The reason reaches only the
    plugin's ``WriteMessage``: the incomplete-shot warning and the flagged
    step's warning carry it beside the device's name, and stop blaming the
    camera.
    """
    cam, _ = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=0.3)
    refusal = (
        "frame shape (301, 2) != the shape declared at the arm (304, 2) "
        "(held frame from before a shape change); re-run the scan"
    )
    set_mock_value(cam.hdf.write_message, refusal)
    box.drop = {("uc_a", n) for n in range(1, 10)}
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.plans.on_demand"):
        RE(strict_count(shot_control, [cam], 1, max_refires=1))
    assert box.fires == 2
    messages = [r.getMessage() for r in caplog.records]
    flagged = [m for m in messages if "flagged incomplete" in m]
    assert len(flagged) == 1
    assert f"no record from UC_A (file plugin uc_a-hdf: {refusal})" in flagged[0]
    assert "a file plugin refused" in flagged[0]
    assert "frame-drop intermittency" not in flagged[0]
    incomplete = [m for m in messages if "shot 1 of 2 incomplete" in m]
    assert len(incomplete) == 1
    assert f"no record from UC_A (file plugin uc_a-hdf: {refusal})" in incomplete[0]


def test_a_plugin_that_never_answers_does_not_hang_the_miss_path(
    RE: RunEngine,
    box: FakeBox,
    shot_control: ShotControl,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason read is bounded: a hung WriteMessage costs the budget and the miss keeps its wording."""
    from geecs_bluesky.devices import detector as detector_module

    cam, _ = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=0.3)
    monkeypatch.setattr(detector_module, "PLUGIN_REASON_TIMEOUT_S", 0.2)

    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(cam.hdf.write_message, "get_value", hang)
    box.drop = {("uc_a", n) for n in range(1, 10)}
    started = time.monotonic()
    RE(strict_count(shot_control, [cam], 1, max_refires=1))
    # Two dropped shots cost the shot timeout each plus the 0.2 s reason
    # budget; a hung read would never return at all.
    assert time.monotonic() - started < 10.0
    assert box.fires == 2


def test_missed_frame_on_plugin_cameras_is_a_gap_in_its_stream(
    RE: RunEngine, box: FakeBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """A misses fire 2: no rewind anywhere — B's frame of that fire is in B's stream, A's stream skips it."""
    a, a_rewinds = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=0.3)
    b, b_rewinds = _plugin_camera(RE, box, "UC_B", tmp_path, shot_timeout=0.3)
    box.drop = {("uc_a", 2)}
    col = DocCollector()
    RE.subscribe(col)
    RE(strict_count(shot_control, [a, b], 3))
    assert len(col.primary_events()) == 4 and box.fires == 4
    assert a_rewinds == [0] and b_rewinds == [0]  # zeroed at the arm only

    def frames(key: str) -> int:
        return sum(d["stop"] - d["start"] for d in col.datums(key))

    assert frames("uc_a") == 3 and frames("uc_b") == 4
