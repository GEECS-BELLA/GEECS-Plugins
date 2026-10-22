"""NativeCamera — a native EPICS areaDetector camera as a strict GEECS detector, on mocks.

The IOC side was verified on a bench camera (geecs-epics#2: the stack
layout; a software trigger gives one frame per press and none while
armed and idle).  Pinned here: the camera is stopped and put in trigger
mode *before* the HDF capture starts and armed once per run, the software
source presses ``TriggerSoftware`` per shot and Line1 does not, a missed
frame is the GEECS timeout and fails the step instead of refiring, the
writer keeps the IOC's layout with SWMR off, the row stamp is the frame's
EPICS timestamp in the LabVIEW epoch through the reader's own conversion,
``unstage`` restores the camera, and fly prepares are refused.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path, PurePosixPath

import pytest

pytest.importorskip("aioca")

from bluesky import RunEngine
from ophyd_async.core import (
    OnOff,
    StaticFilenameProvider,
    StaticPathProvider,
    callback_on_mock_put,
    get_mock_execute,
    set_mock_value,
)
from ophyd_async.epics.adcore import ADImageMode

from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET, epics_timestamp_to_unix

from geecs_bluesky.devices.detector import (
    STRICT_TRIGGER_INFO,
    UNBOUNDED_TRIGGER_INFO,
    gated_trigger_info,
)
from geecs_bluesky.devices.native import (
    AsynConnect,
    NativeCamera,
    NativeTriggerSource,
    liveness_word,
)
from geecs_bluesky.exceptions import GeecsConfigurationError, GeecsTriggerTimeoutError
from tests.ca_mock_helpers import connect_mock

LAYOUT = "/epics/ioc/config/hdf5_layout.xml"
#: A real bench frame's EPICS timestamp (ADCore 3.14, 2026-10-22).
SEC, NSEC = 1161484182, 872442764


def _run(RE: RunEngine, make_awaitable):
    async def call():
        return await make_awaitable()

    return asyncio.run_coroutine_threadsafe(call(), RE._loop).result(timeout=10.0)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


def _camera(
    RE: RunEngine, tmp_path: Path, source=NativeTriggerSource.SOFTWARE, **kwargs
) -> tuple[NativeCamera, list[str]]:
    """A connected mock camera, free-running before the run, and a log of its puts."""
    provider = StaticPathProvider(
        StaticFilenameProvider("ADTEST_CAM1"),
        PurePosixPath(tmp_path / "Scan001" / "ADTEST_CAM1"),
    )
    cam = NativeCamera(
        "ADTEST_CAM1",
        "undulator:adtest_cam1:",
        trigger_source=source,
        path_provider=provider,
        name="adtest_cam1",
        **kwargs,
    )
    connect_mock(RE, cam)
    d, h = cam.driver, cam.hdf
    set_mock_value(d.acquire, True)  # free-running live view
    set_mock_value(d.trigger_mode, OnOff.OFF)
    set_mock_value(d.trigger_source, NativeTriggerSource.LINE1)
    set_mock_value(d.image_mode, ADImageMode.CONTINUOUS)
    set_mock_value(d.asyn_connect, AsynConnect.CONNECT)
    set_mock_value(d.data_type, "UInt8")
    set_mock_value(d.color_mode, "Mono")
    set_mock_value(d.array_size_x, 1280)
    set_mock_value(d.array_size_y, 1024)
    set_mock_value(h.file_path_exists, True)
    set_mock_value(h.xml_file_name, LAYOUT)
    set_mock_value(h.epics_ts_sec, SEC)
    set_mock_value(h.epics_ts_nsec, NSEC)
    log: list[str] = []
    for label, signal in (
        ("acquire", d.acquire),
        ("trigger_mode", d.trigger_mode),
        ("trigger_source", d.trigger_source),
        ("image_mode", d.image_mode),
        ("capture", h.capture),
    ):
        callback_on_mock_put(
            signal, lambda value, label=label, **_: log.append(f"{label}={value}")
        )
    return cam, log


def _frames_on_software_trigger(cam: NativeCamera) -> list[int]:
    """Each TriggerSoftware press writes one frame; returns the press counter."""
    presses: list[int] = []

    def press(*_, **__):
        presses.append(1)
        set_mock_value(cam.hdf.num_captured, len(presses))

    get_mock_execute(cam.driver.trigger_software).side_effect = press
    return presses


def test_pvs_live_under_the_prefix(RE: RunEngine, tmp_path: Path) -> None:
    cam, _ = _camera(RE, tmp_path)
    assert cam.driver.trigger_mode.source.endswith(
        "undulator:adtest_cam1:cam1:TriggerMode_RBV"
    )
    assert cam.hdf.capture.source.endswith("undulator:adtest_cam1:HDF1:Capture_RBV")
    assert cam.plugin_backed and not cam.native_save and cam.count_zeroed


def test_camera_is_stopped_and_set_to_trigger_before_the_capture_starts(
    RE: RunEngine, tmp_path: Path
) -> None:
    """No free-run frame can reach the stack: stop + trigger mode, then capture, then start."""
    cam, log = _camera(RE, tmp_path)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    capture = log.index("capture=True")
    assert log.index("acquire=False") < capture
    assert log.index(f"trigger_mode={OnOff.ON}") < capture
    assert log.index(f"trigger_source={NativeTriggerSource.SOFTWARE}") < capture
    assert log.index(f"image_mode={ADImageMode.CONTINUOUS}") < capture
    assert log.index("acquire=True") > capture
    # Strict prepares every shot: the camera is not touched again.
    before = list(log)
    for _ in range(3):
        _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    assert [
        entry for entry in log[len(before) :] if not entry.startswith("capture")
    ] == []


def test_the_writer_keeps_the_ioc_layout_and_swmr_off(
    RE: RunEngine, tmp_path: Path
) -> None:
    """The stock prepare clears XMLFileName and turns SWMR on; both come back."""
    cam, _ = _camera(RE, tmp_path)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    assert _run(RE, lambda: cam.hdf.xml_file_name.get_value()) == LAYOUT
    assert _run(RE, lambda: cam.hdf.swmr_mode.get_value()) is False
    assert str(_run(RE, lambda: cam.hdf.file_path.get_value())).endswith("ADTEST_CAM1/")


def test_software_shot_presses_once_and_stamps_the_row(
    RE: RunEngine, tmp_path: Path
) -> None:
    cam, _ = _camera(RE, tmp_path)
    presses = _frames_on_software_trigger(cam)
    _run(RE, lambda: cam.stage())

    async def shots(n):
        out = []
        for _ in range(n):
            await cam.prepare(STRICT_TRIGGER_INFO)
            await cam.trigger()
            out.append(await cam.read())
        docs = [doc async for doc in cam.collect_asset_docs()]
        return out, docs

    readings, docs = _run(RE, lambda: shots(3))
    assert len(presses) == 3
    expected = float(epics_timestamp_to_unix(SEC, NSEC)) + LABVIEW_EPOCH_OFFSET
    assert readings[0]["adtest_cam1-acq_timestamp"]["value"] == expected
    datums = [doc for name, doc in docs if name == "stream_datum"]
    # One frame per shot, contiguous from frame 0 (what Tiled's consolidator reads).
    assert sum(d["indices"]["stop"] - d["indices"]["start"] for d in datums) == 3
    assert datums[0]["indices"]["start"] == 0
    assert not cam.missed_shot


def test_line1_waits_for_the_edge_and_never_presses(
    RE: RunEngine, tmp_path: Path
) -> None:
    cam, log = _camera(RE, tmp_path, source=NativeTriggerSource.LINE1)
    presses = _frames_on_software_trigger(cam)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    assert f"trigger_source={NativeTriggerSource.LINE1}" in log

    async def shot():
        status = cam.trigger()
        await asyncio.sleep(0.05)
        assert not status.done
        set_mock_value(cam.hdf.num_captured, 1)  # the timing system's edge
        await status

    _run(RE, shot)
    assert presses == []


def test_a_missed_frame_is_the_geecs_timeout_and_fails_the_step(
    RE: RunEngine, tmp_path: Path
) -> None:
    """No frame: GeecsTriggerTimeoutError, an empty stamp, and no refire (discard raises)."""
    cam, _ = _camera(RE, tmp_path)  # software source, but no frame ever lands
    set_mock_value(cam.hdf.write_message, "Error opening file")
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    with pytest.raises(GeecsTriggerTimeoutError, match="Error opening file"):
        _run(RE, lambda: cam.trigger())
    assert cam.missed_shot
    reading = _run(RE, lambda: cam.read())
    assert math.isnan(reading["adtest_cam1-acq_timestamp"]["value"])
    with pytest.raises(GeecsTriggerTimeoutError, match="cannot refire"):
        _run(RE, lambda: cam.discard_uncollected())


def test_a_frame_that_arrived_survives_another_devices_miss(
    RE: RunEngine, tmp_path: Path
) -> None:
    """Strict calls discard on every plugin-backed device when any device missed."""
    cam, _ = _camera(RE, tmp_path)
    _frames_on_software_trigger(cam)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    _run(RE, lambda: cam.trigger())
    _run(RE, lambda: cam.discard_uncollected())  # no-op: its own frame stays
    assert not cam.missed_shot


def test_unstage_restores_the_camera_and_closes_the_file(
    RE: RunEngine, tmp_path: Path
) -> None:
    cam, log = _camera(RE, tmp_path)
    _frames_on_software_trigger(cam)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    _run(RE, lambda: cam.trigger())
    _run(RE, lambda: cam.unstage())
    d = cam.driver
    assert _run(RE, lambda: d.trigger_mode.get_value()) == OnOff.OFF
    assert _run(RE, lambda: d.trigger_source.get_value()) == NativeTriggerSource.LINE1
    assert _run(RE, lambda: d.acquire.get_value()) is True  # live view again
    assert _run(RE, lambda: cam.hdf.capture.get_value()) is False
    # A second run programs the camera again from scratch.
    log.clear()
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    assert f"trigger_mode={OnOff.ON}" in log


@pytest.mark.parametrize(
    "info",
    [UNBOUNDED_TRIGGER_INFO, gated_trigger_info(5)],
    ids=["non-essential", "gated"],
)
def test_fly_prepares_are_refused(RE: RunEngine, tmp_path: Path, info) -> None:
    cam, log = _camera(RE, tmp_path)
    _run(RE, lambda: cam.stage())
    log.clear()  # stage itself stops the writer (capture=False)
    with pytest.raises(GeecsConfigurationError, match="strict scans only"):
        _run(RE, lambda: cam.prepare(info))
    assert log == []  # refused before anything touched the camera
    with pytest.raises(GeecsConfigurationError):
        _run(RE, lambda: cam.kickoff())


def test_liveness_reads_the_asyn_connection(RE: RunEngine, tmp_path: Path) -> None:
    cam, _ = _camera(RE, tmp_path)
    assert _run(RE, lambda: cam.connected_status.get_value()) == "Connected"
    set_mock_value(cam.driver.asyn_connect, AsynConnect.DISCONNECT)
    assert _run(RE, lambda: cam.connected_status.get_value()) == "Disconnected"
    assert liveness_word(AsynConnect.CONNECT) == "Connected"


def test_column_header_is_the_device_stamp(RE: RunEngine, tmp_path: Path) -> None:
    cam, _ = _camera(RE, tmp_path)
    assert cam._column_headers == {
        "adtest_cam1-acq_timestamp": "ADTEST_CAM1 acq_timestamp"
    }


def test_a_pause_mid_shot_never_replays_the_shot(RE: RunEngine, tmp_path: Path) -> None:
    """The strict shot's rewind rule, on the real RunEngine (``plans/strict.py``).

    An immediate pause lands after the frame was written but before the row
    is saved.  A rewindable shot would replay trigger + fire on resume and
    the append-only writer would hold two frames for one row; the native
    camera is not rewindable, so the resume continues and one frame stays.
    """
    import bluesky.plan_stubs as bps
    from bluesky.preprocessors import rewindable_wrapper, run_wrapper, stage_wrapper
    from bluesky.utils import all_safe_rewind

    cam, _ = _camera(RE, tmp_path)
    presses = _frames_on_software_trigger(cam)
    paused = []

    def press_then_pause(*_, **__):
        presses.append(1)
        set_mock_value(cam.hdf.num_captured, len(presses))
        if not paused:
            paused.append(1)
            # From another thread, as an operator's pause arrives (the RE's
            # own loop thread would deadlock on it).
            import threading

            threading.Thread(target=RE.request_pause, kwargs={"defer": False}).start()

    get_mock_execute(cam.driver.trigger_software).side_effect = press_then_pause
    assert all_safe_rewind([cam]) is False

    def shot():
        yield from bps.prepare(cam, STRICT_TRIGGER_INFO, wait=True)
        yield from bps.checkpoint()
        yield from rewindable_wrapper(
            bps.trigger_and_read([cam]), all_safe_rewind([cam])
        )

    from bluesky.utils import RunEngineInterrupted

    docs: list[tuple[str, dict]] = []
    with pytest.raises(RunEngineInterrupted):
        RE(
            stage_wrapper(run_wrapper(shot()), [cam]),
            lambda name, doc: docs.append((name, doc)),
        )
    assert RE.state == "paused" and paused
    RE.resume()
    assert len(presses) == 1  # the shot was not fired again
    events = [doc for name, doc in docs if name == "event"]
    datums = [doc for name, doc in docs if name == "stream_datum"]
    assert len(events) == 1
    assert sum(d["indices"]["stop"] - d["indices"]["start"] for d in datums) == 1


def test_the_path_provider_hands_a_linux_ioc_a_posix_path(tmp_path: Path) -> None:
    """Same provider and scan-folder rules as the gateway's plugin; POSIX for the IOC."""
    from geecs_bluesky.data_paths import native_ioc_save_path
    from geecs_bluesky.devices.hdf_plugin import PluginPathProvider
    from geecs_bluesky.plans.claim_scan import GeecsScanPathProvider

    shared = GeecsScanPathProvider()
    shared.point_at(tmp_path / "Scan009")
    provider = PluginPathProvider(
        shared, "ADTEST_CAM1", plugin_path=native_ioc_save_path, path_type=PurePosixPath
    )
    with pytest.raises(FileNotFoundError, match="claimed by the scanner"):
        provider("adtest_cam1")
    assert not (tmp_path / "Scan009").exists()  # never a scan folder
    (tmp_path / "Scan009").mkdir()
    info = provider("adtest_cam1")
    assert isinstance(info.directory_path, PurePosixPath)
    assert str(info.directory_path).endswith("/Scan009/ADTEST_CAM1")
    assert info.filename == "ADTEST_CAM1"


def test_native_ioc_save_path_is_the_identity_unless_configured(monkeypatch) -> None:
    """Same-path mount by default; a configured root rebases the scan path onto it."""
    from geecs_bluesky import data_paths

    monkeypatch.setattr(data_paths, "_local_base_path", lambda: "/mnt/hdna2/data")
    monkeypatch.setattr(data_paths, "_read_paths_entry", lambda key: None)
    scan = "/mnt/hdna2/data/Undulator/Y2026/10-Oct/26_1022/scans/Scan001/ADTEST_CAM1"
    assert data_paths.native_ioc_save_path(scan) == scan
    monkeypatch.setattr(data_paths, "_read_paths_entry", lambda key: "/data")
    assert (
        data_paths.native_ioc_save_path(scan)
        == "/data/Undulator/Y2026/10-Oct/26_1022/scans/Scan001/ADTEST_CAM1"
    )
    with pytest.raises(ValueError, match="not under the local data root"):
        data_paths.native_ioc_save_path("/elsewhere/x")


def test_a_refused_trigger_setup_fails_the_prepare(
    RE: RunEngine, tmp_path: Path
) -> None:
    """A refused GenICam write completes its put: the readback check catches it."""
    cam, _ = _camera(RE, tmp_path)
    # A mock put callback's return value is what the record ends up holding.
    callback_on_mock_put(cam.driver.trigger_mode, lambda value, **_: OnOff.OFF)
    _run(RE, lambda: cam.stage())
    with pytest.raises(GeecsConfigurationError, match="refused its trigger setup"):
        _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    assert (
        _run(RE, lambda: cam.hdf.capture.get_value()) is not True
    )  # no capture opened


def test_an_unclean_end_never_becomes_the_pre_run_state(
    RE: RunEngine, tmp_path: Path
) -> None:
    """A run without a completed unstage (halt, killed worker) leaves the camera armed;
    the next run keeps the original pre-run state and restores live view."""
    cam, _ = _camera(RE, tmp_path)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))  # armed; no unstage follows
    _run(RE, lambda: cam.stage())  # the next run
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    _run(RE, lambda: cam.unstage())
    d = cam.driver
    assert _run(RE, lambda: d.trigger_mode.get_value()) == OnOff.OFF
    assert _run(RE, lambda: d.acquire.get_value()) is True  # live view again
