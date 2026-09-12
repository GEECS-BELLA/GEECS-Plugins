"""The file plugin: PV contract, session semantics, file layout.

Hermetic like ``test_server.py`` (a binary fake camera, ``isolate=True``).
The contract test drives the plugin exactly as the worker will: the stock
ophyd-async ``ADHDFDataLogic`` over a real ``NDFileHDF5IO`` connected with
``pva://`` — so every suffix, type and enum choice the worker expects is
pinned on both sides at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import time
from pathlib import Path, PurePath

import h5py
import numpy as np
import pytest
from p4p.client.thread import Context

from geecs_pva_gateway import file_plugin
from geecs_pva_gateway.config import CameraSpec, PvaGatewayConfig
from geecs_pva_gateway.file_plugin import (
    ATTRIBUTES_GROUP,
    FRAMES_DATASET,
    PLUGIN_SUFFIX,
    PV_TABLE,
    HdfFilePlugin,
    attribute_names,
    attributes_xml,
)
from geecs_pva_gateway.server import GeecsPvaGateway
from tests.test_server import DEVICE, IMG, _imaq_blob

pytestmark = pytest.mark.fake_server

LV_EPOCH = 2_082_844_800
PREFIX = "testexp:uc_testcam:image" + PLUGIN_SUFFIX


def _push(img: np.ndarray, unix_stamp: float) -> bytes:
    body = (
        DEVICE
        + b">>7>>image nval,"
        + _imaq_blob(img)
        + b" nvar,acq_timestamp nval,%.3f nvar" % (unix_stamp + LV_EPOCH)
    )
    return struct.pack(">i", len(body)) + body


class StampedCamera:
    """A push server whose frames (image, stamp) the test enqueues.

    Everything queued before a subscriber connects is pushed on connect —
    the way LabVIEW's idle re-push greets a fresh subscription.
    """

    def __init__(self) -> None:
        self.frames: asyncio.Queue[tuple[np.ndarray, float]] = asyncio.Queue()
        self.connections = 0
        self.connected = asyncio.Event()
        self.disconnected = asyncio.Event()
        self._server: asyncio.Server | None = None
        self.port = 0

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    def push(self, img: np.ndarray, unix_stamp: float) -> None:
        self.frames.put_nowait((img, unix_stamp))

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connections += 1
        self.connected.set()
        self.disconnected.clear()
        eof_task: asyncio.Task | None = None
        try:
            header = await reader.readexactly(4)
            await reader.readexactly(struct.unpack(">i", header)[0])  # Wait>>
            eof_task = asyncio.create_task(reader.read())
            while not eof_task.done():
                getter = asyncio.create_task(self.frames.get())
                done, _ = await asyncio.wait(
                    {getter, eof_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if getter not in done:
                    getter.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await getter
                    break
                img, stamp = getter.result()
                writer.write(_push(img, stamp))
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if eof_task is not None:
                eof_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, ConnectionError):
                    await eof_task
            self.connections -= 1
            self.disconnected.set()
            with contextlib.suppress(Exception):
                writer.close()


async def _start_gateway(cam: StampedCamera) -> tuple[GeecsPvaGateway, asyncio.Task]:
    spec = CameraSpec(
        device=DEVICE.decode(),
        host="127.0.0.1",
        port=cam.port,
        experiment="testexp",
        image_variables=["image"],
    )
    gateway = GeecsPvaGateway(PvaGatewayConfig(experiment="testexp", cameras=[spec]))
    task = asyncio.create_task(gateway.run(isolate=True))
    for _ in range(100):
        await asyncio.sleep(0.05)
        try:
            gateway.conf()
            break
        except AssertionError:
            continue
    return gateway, task


async def _shutdown(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _plugin(gateway: GeecsPvaGateway) -> HdfFilePlugin:
    return gateway._workers[0].plugins["image"]


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


# ------------------------------------------------------------------ table
def test_pv_table_covers_every_ndfilehdf5io_suffix():
    """Every suffix ophyd-async's ``NDFileHDF5IO`` connects is in the table."""
    from ophyd_async.epics.adcore import NDFileHDF5IO
    from ophyd_async.epics.core import PvSuffix

    served = set()
    for param in PV_TABLE:
        served.add(param.suffix)
        if param.rbv:
            served.add(param.suffix + "_RBV")
    expected = set()
    for klass in NDFileHDF5IO.__mro__:
        for annotation in getattr(klass, "__annotations__", {}).values():
            for meta in getattr(annotation, "__metadata__", ()):
                if isinstance(meta, PvSuffix):
                    expected.add(meta.read_suffix)
                    if meta.write_suffix:
                        expected.add(meta.write_suffix)
    missing = expected - served
    assert not missing, f"plugin does not serve {sorted(missing)}"


# --------------------------------------------------------------- contract
@pytest.mark.timeout(60)
async def test_stock_adhdf_data_logic_drives_the_plugin(tmp_path, monkeypatch):
    """The worker's exact sequence: stock ADHDFDataLogic over pva:// NDFileHDF5IO."""
    from ophyd_async.core import StaticPathProvider, init_devices
    from ophyd_async.epics.adcore import (
        ADHDFDataLogic,
        NDArrayDescription,
        NDFileHDF5IO,
    )
    from ophyd_async.epics.core import _p4p

    cam = StampedCamera()
    await cam.start()
    gateway, task = await _start_gateway(cam)
    # ophyd-async's PVA context reads the environment once; point it at the
    # isolated server before the first connect.
    for key, value in gateway.conf().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(_p4p, "_context", None)
    run_dir = tmp_path / "Scan001" / "UC_TestCam"
    run_dir.mkdir(parents=True)
    try:
        async with init_devices():
            hdf = NDFileHDF5IO(f"pva://{PREFIX}")
        logic = ADHDFDataLogic(
            array_description=NDArrayDescription(
                shape_signals=[hdf.array_size_z, hdf.array_size_y, hdf.array_size_x],
                data_type_signal=hdf.data_type,
                color_mode_signal=hdf.color_mode,
            ),
            path_provider=StaticPathProvider(
                lambda _=None: "UC_TestCam", directory_path=run_dir
            ),
            driver=hdf,
            writer=hdf,
        )
        # The idle re-push LabVIEW greets a subscription with: an old stamp.
        cam.push(IMG, time.time() - 5.0)
        provider = await logic.prepare_unbounded("cam")
        assert await hdf.capture.get_value() is True
        assert await hdf.file_path_exists.get_value() is True
        # Geometry came from the arming frame; the resource describes it.
        keys = await provider.make_datakeys(1)
        assert keys["cam"]["shape"] == [1, *IMG.shape]
        assert keys["cam"]["dtype_numpy"] == "<u2"
        assert keys["cam"]["external"] == "STREAM:"
        # The attribute keys carry the device (unique across cameras, #829)
        # and spell the worker's event column for the stamp.
        assert {
            "uc_testcam-hdf-image-frame_acq_timestamp",
            "uc_testcam-hdf-image-frame_recv_timestamp",
        } <= set(keys)
        assert not {"acq_timestamp", "uc_testcam-acq_timestamp"} & set(keys)
        assert provider.uri.endswith("UC_TestCam/UC_TestCam.h5")
        assert await provider.collections_written_signal.get_value() == 0
        # Two shots, then a re-push of the second (dedupe), then a stale one.
        t = time.time()
        cam.push(IMG, t)
        cam.push(IMG + 1, t + 1.0)
        cam.push(IMG + 1, t + 1.0)
        cam.push(IMG, t - 30.0)
        await _wait_until(lambda: _plugin(gateway).value("NumCaptured_RBV") == 2)
        await asyncio.sleep(0.2)
        assert await provider.collections_written_signal.get_value() == 2
        docs = [doc async for doc in provider.make_stream_docs(2, 1)]
        assert [name for name, _ in docs].count("stream_resource") == 3
        datum = next(doc for name, doc in docs if name == "stream_datum")
        assert datum["indices"] == {"start": 0, "stop": 2}
        await logic.stop()
        assert await hdf.capture.get_value() is False
    finally:
        await _shutdown(task)
        await cam.stop()
    with h5py.File(run_dir / "UC_TestCam.h5", "r", swmr=True, libver="latest") as f:
        frames = f[FRAMES_DATASET]
        assert frames.shape == (2, *IMG.shape)
        assert frames.chunks == (1, *IMG.shape)
        np.testing.assert_array_equal(frames[1], IMG + 1)
        stamps = f[f"{ATTRIBUTES_GROUP}/uc_testcam-hdf-image-frame_acq_timestamp"][:]
        assert stamps[1] == pytest.approx(t + 1.0, abs=0.002)
        assert "acq_timestamp" not in f[ATTRIBUTES_GROUP]
        assert f.attrs["finalized"]
        assert f.attrs["frames_written"] == 2
        assert f.attrs["duplicates_dropped"] == 1
        assert f.attrs["stale_skipped"] == 2  # the arming frame + the old one


# ---------------------------------------------------------------- session
@pytest.mark.timeout(60)
async def test_session_semantics_over_raw_pva(tmp_path):
    """Directory checks, rewind, counters and the no-file case, via plain puts."""
    cam = StampedCamera()
    await cam.start()
    gateway, task = await _start_gateway(cam)
    plugin = _plugin(gateway)
    ctx = Context("pva", conf=gateway.conf(), useenv=False)
    loop = asyncio.get_running_loop()

    async def put(suffix: str, value) -> None:
        await loop.run_in_executor(None, lambda: ctx.put(PREFIX + suffix, value))

    async def get(suffix: str):
        return await loop.run_in_executor(None, lambda: ctx.get(PREFIX + suffix))

    try:
        # A directory the plugin must never create.
        missing = tmp_path / "Scan002" / "UC_TestCam"
        await put("FilePath", str(missing) + os.sep)
        assert bool(await get("FilePathExists_RBV")) is False
        await put("FileName", "UC_TestCam")
        with pytest.raises(Exception, match="does not exist"):
            await put("Capture", True)
        assert not missing.exists()
        assert str(await get("WriteStatus")) == "Write Error"

        # A claimed directory: capture arms on the first frame.
        run_dir = tmp_path / "Scan002" / "UC_TestCam"
        run_dir.mkdir(parents=True)
        await put("FilePath", str(run_dir) + os.sep)
        assert bool(await get("FilePathExists_RBV")) is True
        cam.push(IMG, time.time() - 5.0)
        await put("Capture", True)
        assert bool(await get("Capture_RBV")) is True
        assert cam.connections == 1  # the plugin holds the subscription
        assert int(await get("ArraySizeX_RBV")) == IMG.shape[1]
        assert str(await get("DataType_RBV")) == "UInt16"
        # Capture=1 again is idempotent (the stock logic may repeat it).
        await put("Capture", True)

        t = time.time()
        cam.push(IMG, t)
        cam.push(IMG + 1, t + 1)
        cam.push(IMG + 2, t + 2)
        await _wait_until(lambda: plugin.value("NumCaptured_RBV") == 3)
        # Rewind to 2: the third frame goes; a late frame stamped before the
        # rewind (a shot from before the refire) is stale, a newer one counts.
        await put("Rewind", 2)
        assert int(await get("NumCaptured_RBV")) == 2
        cam.push(IMG + 3, time.time() - 0.5)  # late: stamped before the rewind
        cam.push(IMG + 4, time.time())  # the retake
        await _wait_until(lambda: plugin.value("NumCaptured_RBV") == 3)
        await asyncio.sleep(0.2)
        assert plugin.value("NumCaptured_RBV") == 3
        # Rewind past the count is refused.
        with pytest.raises(Exception, match="outside"):
            await put("Rewind", 7)
        await put("Capture", False)
        assert bool(await get("Capture_RBV")) is False
        await asyncio.wait_for(cam.disconnected.wait(), 10)  # subscription released

        with h5py.File(run_dir / "UC_TestCam.h5", "r") as f:
            frames = f[FRAMES_DATASET]
            assert frames.shape == (3, *IMG.shape)
            np.testing.assert_array_equal(frames[2], IMG + 4)
            assert f.attrs["rewound"] == 1
            assert f.attrs["stale_skipped"] == 2
            assert f.attrs["frames_written"] == 4
            assert f.attrs["finalized"]

        # A fresh arming frame whose stack cannot be opened: the put fails
        # with the reason, the session is gone and the subscription released.
        broken = tmp_path / "Scan005" / "UC_TestCam"
        broken.mkdir(parents=True)
        await put("FilePath", str(broken) + os.sep)
        real_open = plugin._open_file

        def refuse(session, frame):
            raise OSError("share refused the create")

        plugin._open_file = refuse
        cam.push(IMG, time.time())  # fresh: would be written
        with pytest.raises(Exception, match="share refused"):
            await put("Capture", True)
        plugin._open_file = real_open
        assert bool(await get("Capture_RBV")) is False
        assert not plugin.capturing
        await asyncio.wait_for(cam.disconnected.wait(), 10)
        assert cam.connections == 0
        assert list(broken.iterdir()) == []

        # Two Capture=1 puts racing during arming: one session, one
        # subscription, released once at Capture=0.
        again = tmp_path / "Scan004" / "UC_TestCam"
        again.mkdir(parents=True)
        await put("FilePath", str(again) + os.sep)
        first = loop.run_in_executor(None, lambda: ctx.put(PREFIX + "Capture", True))
        second = loop.run_in_executor(None, lambda: ctx.put(PREFIX + "Capture", True))
        await asyncio.sleep(0.3)
        cam.push(IMG, time.time() - 5.0)  # the arming frame
        await asyncio.gather(first, second)
        assert bool(await get("Capture_RBV")) is True
        assert cam.connections == 1
        await put("Capture", False)
        await asyncio.wait_for(cam.disconnected.wait(), 10)
        assert cam.connections == 0

        # A session that accepts nothing leaves no file.
        empty = tmp_path / "Scan003" / "UC_TestCam"
        empty.mkdir(parents=True)
        await put("FilePath", str(empty) + os.sep)
        cam.push(IMG, time.time() - 5.0)
        await put("Capture", True)
        await asyncio.sleep(0.3)
        await put("Capture", False)
        assert list(empty.iterdir()) == []

        # A camera that pushes nothing fails the capture put, loudly.
        file_plugin.ARM_TIMEOUT_S = 0.5
        try:
            with pytest.raises(Exception, match="no frame"):
                await put("Capture", True)
        finally:
            file_plugin.ARM_TIMEOUT_S = 8.0
        assert bool(await get("Capture_RBV")) is False
    finally:
        ctx.close()
        await _shutdown(task)
        await cam.stop()


def test_no_plugin_without_h5py(monkeypatch):
    """A box not re-bootstrapped serves no plugin PVs at all."""
    monkeypatch.setattr(file_plugin, "available", lambda: False)
    spec = CameraSpec(
        device="UC_TestCam", host="127.0.0.1", port=1, experiment="testexp"
    )
    from geecs_pva_gateway.server import _CameraWorker

    worker = _CameraWorker(spec, asyncio.new_event_loop())
    assert worker.plugins == {}
    assert [name for name, _, _ in worker.provider_entries()] == [
        "testexp:uc_testcam:image"
    ]


def test_stack_reads_back_through_data_utils(tmp_path):
    """The read side (geecs_data_utils.io.scan_stack) finds and joins the stack."""
    from geecs_data_utils.io.scan_stack import (
        find_stack_file,
        read_shot_for_acq_timestamp,
        read_stack_timestamps,
    )

    device_dir = tmp_path / "UC_TestCam"
    device_dir.mkdir()
    path = device_dir / "UC_TestCam.h5"
    stamps = np.array([100.0, 101.0, 102.0])
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(FRAMES_DATASET, data=np.stack([IMG, IMG + 1, IMG + 2]))
        f.create_dataset(f"{ATTRIBUTES_GROUP}/acq_timestamp", data=stamps)
    assert find_stack_file(device_dir) == path
    np.testing.assert_array_equal(read_stack_timestamps(path), stamps)
    index, frame = read_shot_for_acq_timestamp(path, 101.0, labview_epoch=False)
    assert index == 1
    np.testing.assert_array_equal(frame, IMG + 1)


def test_pathinfo_windows_and_uri_are_independent():
    """The seam the worker uses: a Windows directory for the plugin, a file URI for Tiled."""
    from ophyd_async.core import PathInfo

    info = PathInfo(
        directory_path=PurePath("//nas/hdna2/data/Scan001/Cam"),
        filename="Cam",
        directory_uri="file://localhost/mnt/hdna2/data/Scan001/Cam/",
    )
    assert info.directory_uri == "file://localhost/mnt/hdna2/data/Scan001/Cam/"
    assert Path(str(info.directory_path)).name == "Cam"


# ------------------------------------------------------------ attribute names
def test_attribute_names_carry_the_normalized_device() -> None:
    """``<device>-<suffix>`` under the shared naming contract (the worker's ophyd name)."""
    assert attribute_names("UC Test.Cam", "bakground image") == (
        "uc_test_cam-hdf-bakground_image-frame_acq_timestamp",
        "uc_test_cam-hdf-bakground_image-frame_recv_timestamp",
    )
    xml = attributes_xml("UC_TestCam", "image")
    assert 'name="uc_testcam-hdf-image-frame_acq_timestamp"' in xml
    assert 'name="uc_testcam-hdf-image-frame_recv_timestamp"' in xml
    # Never an event column's name (the camera's own stamp column is
    # uc_testcam-acq_timestamp) and never the bare name (#829).
    assert "uc_testcam-acq_timestamp" not in xml and 'name="acq_timestamp"' not in xml
    plugin = HdfFilePlugin(
        device="UC_TestCam",
        variable="image",
        experiment="TestExp",
        retain=lambda _v: None,
        release=lambda _v: None,
    )
    try:
        assert plugin.value("NDAttributesFile") == xml
        assert plugin.attributes == attribute_names("UC_TestCam", "image")
    finally:
        plugin.stop()
