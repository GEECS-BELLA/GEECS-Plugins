"""GeecsDetector — the GEECS acquirer as a stock StandardDetector (phase 0, #807).

Device-level contracts on mock CA backends: the synchronous baseline that
makes the shot wait exact, the timeout, the native-saving data logic's
lifecycle and its scan-folder invariant, and the configuration the detector
reports.  The plan-level behaviour is in ``test_strict_plans.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("aioca")  # CA backend needs the `ca` extra

from bluesky import RunEngine  # noqa: E402
from bluesky.utils import FailedStatus  # noqa: E402
from ophyd_async.core import (  # noqa: E402
    DetectorTrigger,
    StaticFilenameProvider,
    StaticPathProvider,
    TriggerInfo,
    set_mock_value,
)

from geecs_bluesky.devices.detector import (  # noqa: E402
    STRICT_TRIGGER_INFO,
    GeecsDetector,
)
from geecs_bluesky.exceptions import GeecsTriggerTimeoutError  # noqa: E402
from tests.ca_mock_helpers import connect_mock  # noqa: E402


async def _watch(signal, sink: list) -> None:
    """Subscribe (on the loop) and collect every value the signal reports."""
    signal.subscribe(lambda reading: sink.append(reading[signal.name]["value"]))


def _run(RE: RunEngine, make_awaitable):
    """Call *make_awaitable* inside the RE loop and await what it returns.

    Device methods wrapped in ``AsyncStatus`` create their task at call time,
    so the call itself must happen on the running loop.
    """

    async def call():
        return await make_awaitable()

    return asyncio.run_coroutine_threadsafe(call(), RE._loop).result(timeout=10.0)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


def _camera(RE: RunEngine, tmp_path: Path | None = None, **kwargs) -> GeecsDetector:
    path_provider = None
    if tmp_path is not None:
        path_provider = StaticPathProvider(
            StaticFilenameProvider("frame"), tmp_path / "Scan001" / "UC_TestCam"
        )
    cam = GeecsDetector(
        "UC_TestCam",
        ["MeanCounts", "MaxCounts", "acq_timestamp"],
        experiment="TestExp",
        name="uc_testcam",
        path_provider=path_provider,
        **kwargs,
    )
    connect_mock(RE, cam)
    set_mock_value(cam.acq_timestamp, 1000.0)
    return cam


def test_children_and_headers(RE: RunEngine) -> None:
    """Scalars, the stamp and liveness are children; headers map back to GEECS names."""
    cam = _camera(RE)
    assert cam.meancounts.name == "uc_testcam-meancounts"
    assert cam.acq_timestamp.name == "uc_testcam-acq_timestamp"
    assert cam.connected_status.name == "uc_testcam-connected_status"
    assert cam._column_headers == {
        "uc_testcam-meancounts": "UC_TestCam MeanCounts",
        "uc_testcam-maxcounts": "UC_TestCam MaxCounts",
        "uc_testcam-acq_timestamp": "UC_TestCam acq_timestamp",
    }
    # No path provider → no save controls, scalars only.
    assert not hasattr(cam, "save")


def test_only_external_edges_are_supported(RE: RunEngine) -> None:
    """A GEECS camera cannot self-trigger: INTERNAL is refused at prepare."""
    cam = _camera(RE)
    assert cam._supported_triggers == {DetectorTrigger.EXTERNAL_EDGE}
    with pytest.raises(ValueError, match="not supported"):
        _run(RE, lambda: cam.prepare(TriggerInfo(trigger=DetectorTrigger.INTERNAL)))
    with pytest.raises(ValueError, match="exposure"):
        _run(
            RE,
            lambda: cam.prepare(
                TriggerInfo(trigger=DetectorTrigger.EXTERNAL_EDGE, livetime=0.5)
            ),
        )


def test_trigger_waits_for_the_stamp_to_advance(RE: RunEngine) -> None:
    """trigger() completes only when acq_timestamp changes; the row carries it."""
    cam = _camera(RE)
    set_mock_value(cam.meancounts, 7.0)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))

    async def shot():
        status = cam.trigger()
        await asyncio.sleep(0.05)
        assert not status.done  # nothing fired yet
        set_mock_value(cam.acq_timestamp, 1001.0)  # the shot
        await status
        return await cam.read()

    reading = _run(RE, lambda: shot())
    assert reading["uc_testcam-acq_timestamp"]["value"] == 1001.0
    assert reading["uc_testcam-meancounts"]["value"] == 7.0
    _run(RE, lambda: cam.unstage())


def test_baseline_is_taken_synchronously_in_trigger(RE: RunEngine) -> None:
    """A shot landing right after trigger() returns is the awaited shot (no blind window).

    The fire is the plan's very next message; with an asynchronous baseline a
    stamp arriving before the coroutine's first step would be read as the
    pre-shot value and the real shot missed.  Pinned here by firing
    synchronously, before the event loop runs the trigger task at all.
    """
    cam = _camera(RE)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))

    async def race():
        status = cam.trigger()
        set_mock_value(cam.acq_timestamp, 1001.0)  # fires before any await
        await asyncio.wait_for(status, timeout=1.0)

    _run(RE, lambda: race())


def test_no_shot_raises_the_attributable_timeout(RE: RunEngine) -> None:
    """No stamp within shot_timeout → GeecsTriggerTimeoutError naming the device."""
    cam = _camera(RE, shot_timeout=0.2)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    with pytest.raises(GeecsTriggerTimeoutError) as info:
        _run(RE, lambda: cam.trigger())
    assert info.value.device_name == "UC_TestCam"
    assert "FailedStatus" not in type(info.value).__name__


def test_stale_updates_before_trigger_are_not_a_shot(RE: RunEngine) -> None:
    """Stamps that arrived before trigger() are drained; only a newer one counts."""
    cam = _camera(RE)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    set_mock_value(cam.acq_timestamp, 1001.0)
    set_mock_value(cam.acq_timestamp, 1002.0)

    async def shot():
        status = cam.trigger()  # baseline = 1002.0, queue drained
        await asyncio.sleep(0.05)
        assert not status.done
        set_mock_value(cam.acq_timestamp, 1003.0)
        await status

    _run(RE, lambda: shot())


def test_native_saving_lifecycle(RE: RunEngine, tmp_path: Path) -> None:
    """prepare → dir created, path + save=on; stop/unstage → save=off."""
    (tmp_path / "Scan001").mkdir()
    cam = _camera(RE, tmp_path)
    seen: list[str] = []
    _run(RE, lambda: _watch(cam.save, seen))

    _run(RE, lambda: cam.stage())  # eager save-off: a stale flag is cleared first
    assert _run(RE, lambda: cam.save.get_value()) == "off"
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    directory = tmp_path / "Scan001" / "UC_TestCam"
    assert directory.is_dir()
    assert _run(RE, lambda: cam.save.get_value()) == "on"
    # The device-side path: the config.ini mapping is not loaded under test,
    # so the worker path passes through unchanged.
    assert _run(RE, lambda: cam.localsavingpath.get_value()) == str(directory)
    describe = _run(RE, lambda: cam.describe())
    assert describe["uc_testcam-nonscalar_save_path"]["dtype"] == "string"
    _run(RE, lambda: cam.unstage())
    assert _run(RE, lambda: cam.save.get_value()) == "off"
    assert seen[-1] == "off"


def test_native_saving_never_creates_the_scan_folder(
    RE: RunEngine, tmp_path: Path
) -> None:
    """A missing scan folder is an error, never a mkdir (the analysis-side invariant)."""
    cam = _camera(RE, tmp_path)  # tmp_path/Scan001 does NOT exist
    _run(RE, lambda: cam.stage())
    with pytest.raises(FileNotFoundError, match="claimed by the scanner"):
        _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    assert not (tmp_path / "Scan001").exists()


def test_drain_offset_is_configuration(RE: RunEngine) -> None:
    """The calibrated drain offset rides in read_configuration."""
    cam = _camera(RE)
    _run(RE, lambda: cam.drain_offset.set(0.066))
    config = _run(RE, lambda: cam.read_configuration())
    assert config["uc_testcam-drain_offset"]["value"] == pytest.approx(0.066)
    triggers, deadtime = _run(RE, lambda: cam.get_trigger_deadtime())
    assert triggers == {DetectorTrigger.EXTERNAL_EDGE}
    assert deadtime == pytest.approx(0.066)


def test_failed_status_carries_the_geecs_error(RE: RunEngine) -> None:
    """Through the RunEngine a no-frame wait surfaces as FailedStatus ← GeecsTriggerTimeoutError."""
    import bluesky.plan_stubs as bps

    cam = _camera(RE, shot_timeout=0.2)

    def plan():
        # stage() is asynchronous and resets the prepare context: wait for
        # it, as the stock stage_wrapper does, or the prepare races it.
        yield from bps.stage(cam, wait=True)
        yield from bps.prepare(cam, STRICT_TRIGGER_INFO, wait=True)
        yield from bps.trigger(cam, wait=True)

    with pytest.raises(FailedStatus) as info:
        RE(plan())
    assert isinstance(info.value.__cause__, GeecsTriggerTimeoutError)


def test_native_save_without_a_path_clears_a_stale_flag_and_adds_no_column(
    RE: RunEngine,
) -> None:
    """A camera whose frames are not wanted still owns ``save`` (found live 26_0828)."""
    cam = GeecsDetector(
        "UC_TestCam",
        ["MeanCounts"],
        experiment="TestExp",
        name="uc_testcam",
        native_save=True,
    )
    connect_mock(RE, cam)
    set_mock_value(cam.acq_timestamp, 1000.0)
    set_mock_value(cam.save, "on")  # left on by a crashed run
    _run(RE, lambda: cam.stage())
    assert _run(RE, lambda: cam.save.get_value()) == "off"
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    assert _run(RE, lambda: cam.save.get_value()) == "off"
    assert "uc_testcam-nonscalar_save_path" not in _run(RE, lambda: cam.describe())
    _run(RE, lambda: cam.unstage())
