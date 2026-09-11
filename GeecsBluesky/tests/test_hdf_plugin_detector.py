"""GeecsDetector on the PVA gateway's file plugin (#806) — the worker side on mocks.

The plugin itself is exercised for real in GeecsPvaGateway's
``tests/test_file_plugin.py`` (stock ``ADHDFDataLogic`` over ``pva://``);
here the plugin's PVs are mock signals and the contracts pinned are the
detector's: the stock lifecycle produces stream documents, the count wait
precedes the stamp wait and its timeout is the GEECS one, a missed shot
reads empty, ``discard_uncollected`` rewinds to the last referenced frame,
and the path provider hands the plugin and Tiled their two paths.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path, PureWindowsPath

import pytest

pytest.importorskip("aioca")
pytest.importorskip("p4p")

from bluesky import RunEngine  # noqa: E402
from ophyd_async.core import (  # noqa: E402
    DetectorTrigger,
    StaticFilenameProvider,
    StaticPathProvider,
    TriggerInfo,
    callback_on_mock_put,
    set_mock_value,
)

from geecs_bluesky.devices.detector import (  # noqa: E402
    DEFAULT_SHOT_TIMEOUT,
    STRICT_TRIGGER_INFO,
    GeecsDetector,
    mask_missed_shot,
)
from geecs_bluesky.devices.hdf_plugin import (  # noqa: E402
    GeecsHdfIO,
    PluginPathProvider,
    file_plugin_hosts,
)
from geecs_bluesky.exceptions import GeecsTriggerTimeoutError  # noqa: E402
from geecs_bluesky.plans.claim_scan import GeecsScanPathProvider  # noqa: E402
from tests.ca_mock_helpers import connect_mock  # noqa: E402


def _run(RE: RunEngine, make_awaitable):
    async def call():
        return await make_awaitable()

    return asyncio.run_coroutine_threadsafe(call(), RE._loop).result(timeout=10.0)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


def _camera(RE: RunEngine, tmp_path: Path, **kwargs) -> GeecsDetector:
    provider = StaticPathProvider(
        StaticFilenameProvider("UC_TestCam"), tmp_path / "Scan001" / "UC_TestCam"
    )
    cam = GeecsDetector(
        "UC_TestCam",
        ["MeanCounts"],
        experiment="TestExp",
        name="uc_testcam",
        hdf_plugins=[("image", provider)],
        **kwargs,
    )
    connect_mock(RE, cam)
    set_mock_value(cam.acq_timestamp, 1000.0)
    set_mock_value(cam.hdf.file_path_exists, True)
    set_mock_value(cam.hdf.data_type, "UInt16")
    set_mock_value(cam.hdf.color_mode, "Mono")
    set_mock_value(cam.hdf.array_size_x, 6)
    set_mock_value(cam.hdf.array_size_y, 4)
    return cam


def test_plugin_child_and_pv_prefix(RE: RunEngine, tmp_path: Path) -> None:
    """One GeecsHdfIO per image variable, under the naming contract's prefix."""
    cam = _camera(RE, tmp_path)
    assert cam.plugin_backed
    assert isinstance(cam.hdf, GeecsHdfIO)
    assert cam.hdf.capture.source.startswith(
        "mock+pva://testexp:uc_testcam:image:hdf1:"
    )
    assert cam.hdf.rewind.source.endswith(":hdf1:Rewind")
    assert not GeecsDetector("UC_Other", [], name="o").plugin_backed


def test_stock_lifecycle_emits_stream_documents(RE: RunEngine, tmp_path: Path) -> None:
    """stage → prepare → trigger → collect: capture on, one datum per shot, capture off."""
    cam = _camera(RE, tmp_path)
    set_mock_value(cam.meancounts, 5.0)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))
    assert _run(RE, lambda: cam.hdf.capture.get_value()) is True
    assert _run(RE, lambda: cam.hdf.num_capture.get_value()) == 0  # unbounded
    assert _run(RE, lambda: cam.hdf.file_path.get_value()).endswith("UC_TestCam/")
    assert _run(RE, lambda: cam.hdf.file_name.get_value()) == "UC_TestCam"
    described = _run(RE, lambda: cam.describe())
    assert described["uc_testcam"]["external"] == "STREAM:"
    assert described["uc_testcam"]["shape"] == [1, 4, 6]
    assert "uc_testcam-meancounts" in described

    async def shot():
        status = cam.trigger()
        await asyncio.sleep(0.05)
        assert not status.done
        set_mock_value(cam.hdf.num_captured, 1)  # the plugin wrote the frame
        await asyncio.sleep(0.05)
        assert not status.done  # the stamp wait comes second
        set_mock_value(cam.acq_timestamp, 1001.0)
        await status
        docs = [doc async for doc in cam.collect_asset_docs()]
        return await cam.read(), docs

    reading, docs = _run(RE, lambda: shot())
    assert reading["uc_testcam-acq_timestamp"]["value"] == 1001.0
    assert reading["uc_testcam-meancounts"]["value"] == 5.0
    names = [name for name, _ in docs]
    assert names.count("stream_resource") == 1 and names.count("stream_datum") == 1
    resource = docs[0][1]
    assert resource["mimetype"] == "application/x-hdf5"
    assert resource["parameters"]["dataset"] == "/entry/data/data"
    assert resource["uri"].endswith("Scan001/UC_TestCam/UC_TestCam.h5")
    assert docs[1][1]["indices"] == {"start": 0, "stop": 1}
    _run(RE, lambda: cam.unstage())
    assert _run(RE, lambda: cam.hdf.capture.get_value()) is False


def test_count_timeout_is_the_geecs_timeout_and_the_row_reads_empty(
    RE: RunEngine, tmp_path: Path
) -> None:
    """No frame counted within exposure_timeout → GeecsTriggerTimeoutError, missed, NaN."""
    cam = _camera(RE, tmp_path)
    set_mock_value(cam.meancounts, 5.0)
    assert STRICT_TRIGGER_INFO.exposure_timeout == DEFAULT_SHOT_TIMEOUT
    quick = TriggerInfo(
        trigger=DetectorTrigger.EXTERNAL_EDGE, number_of_events=1, exposure_timeout=0.2
    )
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(quick))

    async def missed():
        with pytest.raises(GeecsTriggerTimeoutError, match="UC_TestCam.*file plugin"):
            await cam.trigger()
        assert cam.missed_shot
        reading = await cam.read()
        # The next baseline clears the flag; the retake reads live values.
        status = cam.trigger()
        assert not cam.missed_shot
        await asyncio.sleep(0.05)  # the trigger re-baselines the count first
        set_mock_value(cam.hdf.num_captured, 1)
        set_mock_value(cam.acq_timestamp, 1002.0)
        await status
        return reading, await cam.read()

    empty, live = _run(RE, lambda: missed())
    assert math.isnan(empty["uc_testcam-meancounts"]["value"])
    assert math.isnan(empty["uc_testcam-acq_timestamp"]["value"])
    assert live["uc_testcam-meancounts"]["value"] == 5.0
    assert live["uc_testcam-acq_timestamp"]["value"] == 1002.0
    _run(RE, lambda: cam.unstage())


def test_scalars_view_reads_empty_after_a_missed_shot(
    RE: RunEngine, tmp_path: Path
) -> None:
    """``X.scalars`` shares the owner's flag: the same shot, the same empty row."""
    cam = _camera(RE, tmp_path, shot_timeout=0.2)
    set_mock_value(cam.meancounts, 5.0)

    async def missed():
        with pytest.raises(GeecsTriggerTimeoutError):
            await cam.scalars.trigger()
        assert cam.scalars.missed_shot
        return await cam.scalars.read()

    reading = _run(RE, lambda: missed())
    assert math.isnan(reading["uc_testcam-meancounts"]["value"])


def test_discard_uncollected_rewinds_to_the_last_referenced_frame(
    RE: RunEngine, tmp_path: Path
) -> None:
    """A late frame past the last datum is rewound; a delivered device is untouched."""
    cam = _camera(RE, tmp_path)
    rewinds: list[int] = []

    def plugin_rewinds(value, **_) -> None:
        rewinds.append(value)
        set_mock_value(cam.hdf.num_captured, value)

    callback_on_mock_put(cam.hdf.rewind, plugin_rewinds)
    _run(RE, lambda: cam.stage())
    _run(RE, lambda: cam.prepare(STRICT_TRIGGER_INFO))

    async def scenario():
        status = cam.trigger()
        await asyncio.sleep(0.05)  # the trigger re-baselines the count first
        set_mock_value(cam.hdf.num_captured, 1)
        set_mock_value(cam.acq_timestamp, 1001.0)
        await status
        _ = [doc async for doc in cam.collect_asset_docs()]  # frame 0 referenced
        set_mock_value(cam.hdf.num_captured, 2)  # a late frame nobody referenced
        await cam.discard_uncollected()
        return await cam.hdf.num_captured.get_value()

    assert _run(RE, lambda: scenario()) == 1
    assert rewinds == [1]
    # Outside prepare, and without a plugin, it is a no-op.
    _run(RE, lambda: cam.unstage())
    plain = GeecsDetector("UC_Plain", [], name="plain")
    connect_mock(RE, plain)
    _run(RE, lambda: plain.discard_uncollected())


def test_mask_missed_shot_blanks_by_type() -> None:
    masked = mask_missed_shot(
        {
            "a": {"value": 3.5, "timestamp": 1.0, "alarm_severity": 0},
            "b": {"value": 7, "timestamp": 1.0, "alarm_severity": 0},
            "c": {"value": "on", "timestamp": 1.0, "alarm_severity": 0},
        }
    )
    assert math.isnan(masked["a"]["value"]) and math.isnan(masked["b"]["value"])
    assert masked["c"]["value"] == ""


def test_plugin_path_provider_hands_out_both_paths(tmp_path: Path) -> None:
    """Windows path for the plugin's FilePath, the worker's file URI for Tiled."""
    shared = GeecsScanPathProvider()
    shared.point_at(tmp_path / "Scan007")
    provider = PluginPathProvider(
        shared,
        "UC_TestCam",
        plugin_path=lambda local: local.replace(str(tmp_path), r"\\nas\hdna2\data"),
    )
    info = provider("uc_testcam")  # the ophyd datakey is ignored
    assert isinstance(info.directory_path, PureWindowsPath)
    assert str(info.directory_path) == r"\\nas\hdna2\data\Scan007\UC_TestCam"
    assert info.filename == "UC_TestCam"
    assert info.directory_uri.startswith("file://localhost/")
    assert info.directory_uri.endswith("/Scan007/UC_TestCam/")


def test_file_plugin_hosts_reads_the_config_keys(tmp_path: Path) -> None:
    ini = tmp_path / "config.ini"
    ini.write_text("[pva]\naddr_list = 192.168.6.100 192.168.6.101\n")
    assert file_plugin_hosts(ini) == {"192.168.6.100", "192.168.6.101"}
    ini.write_text(
        "[pva]\naddr_list = 192.168.6.100 192.168.6.101\n"
        "file_plugin_addr_list = 192.168.6.101\n"
    )
    assert file_plugin_hosts(ini) == {"192.168.6.101"}
    ini.write_text("[Paths]\ngeecs_data = x\n")
    assert file_plugin_hosts(ini) is None
    assert file_plugin_hosts(tmp_path / "missing.ini") is None
