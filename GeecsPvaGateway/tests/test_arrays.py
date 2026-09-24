"""Array variables end to end: the wire shapes → padded float64 NTNDArrays → 1-D/2-D stacks.

A fake device pushes one named variable with a payload the test chooses
(nested pairs, a LabVIEW waveform, an image), so the gateway's per-variable
decode, the ceiling padding, the NTNDArray attributes and the file plugin's
1-D and float paths are pinned over a real ``isolate=True`` PVA server —
the same harness shape as ``test_server.py`` / ``test_file_plugin.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("p4p")
h5py = pytest.importorskip("h5py")

from p4p.client.thread import Context  # noqa: E402

from geecs_pva_gateway import file_plugin  # noqa: E402
from geecs_pva_gateway.config import DeviceSpec, PvaGatewayConfig  # noqa: E402
from geecs_pva_gateway.server import GeecsPvaGateway  # noqa: E402
from geecs_pva_gateway.streams import ArrayTooLongError, pad_rows  # noqa: E402

pytestmark = pytest.mark.skipif(not file_plugin.available(), reason="h5py")

DEVICE = b"U_Spec"
LV_EPOCH = 2_082_844_800
WIRE = (
    Path(__file__).parent.parent.parent / "GEECS-Data-Utils" / "tests" / "data" / "wire"
)


def _pairs(n: int, start: float = 51.55832, de: float = 0.25) -> bytes:
    """``[[x,y], [x,y], ...]`` — the MagSpec lineout shape, *n* rows."""
    rows = ", ".join(f"[{start + i * de:.6E},{float(i * 10):.6E}]" for i in range(n))
    return f"[{rows}]".encode()


def _waveform(
    raw: list[int], *, dx: float = 4e-9, offset=-0.5, gain=1e-3, name="Ch0"
) -> bytes:
    header = (
        f"{len(raw)}.000000,0.000000000000,{dx:.12f},{offset:.12f},{gain:.12f},{name}|"
    )
    return (
        header.encode()
        + struct.pack(">I", len(raw))
        + struct.pack(f">{len(raw)}h", *raw)
    )


def _push(variable: bytes, payload: bytes, unix_stamp: float) -> bytes:
    body = (
        DEVICE
        + b">>7>>"
        + variable
        + b" nval,"
        + payload
        + b" nvar,acq_timestamp nval,%.3f nvar" % (unix_stamp + LV_EPOCH)
    )
    return struct.pack(">i", len(body)) + body


class ArrayDevice:
    """A push server for one variable whose payloads the test enqueues."""

    def __init__(self, variable: str) -> None:
        self.variable = variable.encode()
        self.frames: asyncio.Queue[tuple[bytes, float]] = asyncio.Queue()
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

    def push(self, payload: bytes, unix_stamp: float) -> None:
        self.frames.put_nowait((payload, unix_stamp))

    async def _handle(self, reader, writer) -> None:
        self.connections += 1
        self.connected.set()
        self.disconnected.clear()
        eof_task = None
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
                payload, stamp = getter.result()
                writer.write(_push(self.variable, payload, stamp))
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


async def _start_gateway(dev: ArrayDevice, variable: str, ceiling: int | None):
    spec = DeviceSpec(
        device=DEVICE.decode(),
        host="127.0.0.1",
        port=dev.port,
        experiment="testexp",
        devicetype="MagSpecCamera",
        array_variables=[variable],
        array_ceiling=ceiling,
    )
    gateway = GeecsPvaGateway(PvaGatewayConfig(experiment="testexp", devices=[spec]))
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


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.02)


# ------------------------------------------------------------------ unit
def test_pad_rows_fills_with_nan_and_refuses_to_truncate() -> None:
    padded = pad_rows(np.array([[1.0, 2.0]]), 4)
    assert padded.shape == (4, 2) and padded.dtype == np.float64
    assert padded[0].tolist() == [1.0, 2.0] and np.isnan(padded[1:]).all()
    assert pad_rows(np.arange(3.0), 3).tolist() == [
        0.0,
        1.0,
        2.0,
    ]  # exactly at the ceiling
    with pytest.raises(ArrayTooLongError, match="never truncated"):
        pad_rows(np.zeros((5, 2)), 4)


def test_array_plugins_declare_the_waveform_axis_after_the_scalars() -> None:
    names = file_plugin.attribute_names(
        "U_ICT", "scopeTrace.Channel0", ["MaxV"], is_array=True
    )
    assert names == (
        "u_ict-hdf-scopetrace_channel0-frame_acq_timestamp",
        "u_ict-hdf-scopetrace_channel0-frame_recv_timestamp",
        "u_ict-hdf-scopetrace_channel0-maxv",
        "u_ict-hdf-scopetrace_channel0-wave_x0",
        "u_ict-hdf-scopetrace_channel0-wave_dx",
        "u_ict-hdf-scopetrace_channel0-wave_samples",
    )
    # An image plugin's names are unchanged.
    assert file_plugin.attribute_names("UC_Cam", "image", ["MaxV"]) == (
        "uc_cam-hdf-image-frame_acq_timestamp",
        "uc_cam-hdf-image-frame_recv_timestamp",
        "uc_cam-hdf-image-maxv",
    )
    assert "wave_dx" in file_plugin.attributes_xml(
        "U_ICT", "scopeTrace.Channel0", is_array=True
    )
    assert "wave_dx" not in file_plugin.attributes_xml("UC_Cam", "image")


def test_device_spec_stream_variables_are_images_then_arrays() -> None:
    spec = DeviceSpec(
        device="D",
        host="h",
        port=1,
        experiment="e",
        image_variables=["Image"],
        array_variables=["interpSpec"],
    )
    assert spec.stream_variables == ["Image", "interpSpec"]
    assert spec.is_array("interpSpec") and not spec.is_array("Image")


def test_an_idle_instance_is_named_after_its_served_host_not_the_machine() -> None:
    """No device to serve: the identity PVs still carry the addr_list IP the fleet
    probe asks for (review of #946), never socket.gethostname()."""
    idle = GeecsPvaGateway(PvaGatewayConfig(experiment="e", host="192.168.7.168"))
    assert idle._instance_host() == "192.168.7.168"
    assert idle.pv_names == []
    # A config without a recorded host keeps the historic fallback order.
    spec = DeviceSpec(
        device="D",
        host="192.168.8.201",
        port=1,
        experiment="e",
        image_variables=["image"],
    )
    assert (
        GeecsPvaGateway(
            PvaGatewayConfig(experiment="e", devices=[spec])
        )._instance_host()
        == "192.168.8.201"
    )


# -------------------------------------------------------- end to end: PV
@pytest.mark.timeout(30)
async def test_pairs_are_posted_padded_to_the_ceiling_and_a_single_row_is_fine():
    """A 285-row lineout and a 1-row default both post as (8, 2) — the ceiling here."""
    dev = ArrayDevice("interpSpec")
    await dev.start()
    gateway, task = await _start_gateway(dev, "interpSpec", ceiling=8)
    try:
        received: list = []
        got = threading.Event()

        def on_update(value) -> None:
            received.append(value)
            if value.shape == (8, 2):
                got.set()

        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        loop = asyncio.get_running_loop()
        try:
            sub = ctx.monitor("testexp:u_spec:interpspec", on_update)
            await asyncio.wait_for(dev.connected.wait(), 5)
            dev.push(_pairs(5), time.time())
            await loop.run_in_executor(None, got.wait, 10)
            frame = np.asarray(received[-1])
            assert frame.shape == (8, 2) and frame.dtype == np.float64
            assert frame[0].tolist() == pytest.approx([51.55832, 0.0])
            assert frame[4, 0] == pytest.approx(51.55832 + 4 * 0.25)
            assert np.isnan(frame[5:]).all()
            # The 1 x 2 magnet-off default is an ordinary frame, same shape.
            got.clear()
            received.clear()
            dev.push(b"[[5.155832E+1,0.000000E+0]]", time.time())
            await loop.run_in_executor(None, got.wait, 10)
            frame = np.asarray(received[-1])
            assert frame.shape == (8, 2) and np.isnan(frame[1:]).all()
            sub.close()
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
        await dev.stop()


@pytest.mark.timeout(30)
async def test_a_waveform_posts_volts_at_native_length_with_its_axis_attributes():
    dev = ArrayDevice("scopeTrace.Channel0")
    await dev.start()
    gateway, task = await _start_gateway(dev, "scopeTrace.Channel0", ceiling=None)
    try:
        received: list = []
        got = threading.Event()

        def on_update(value) -> None:
            received.append(value)
            if value.shape == (3,):
                got.set()

        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        loop = asyncio.get_running_loop()
        try:
            sub = ctx.monitor("testexp:u_spec:scopetrace_channel0", on_update)
            await asyncio.wait_for(dev.connected.wait(), 5)
            dev.push(_waveform([0, 1000, -1000]), time.time())
            await loop.run_in_executor(None, got.wait, 10)
            frame = received[-1]
            np.testing.assert_allclose(np.asarray(frame), [-0.5, 0.5, -1.5])
            attrs = {a["name"]: a["value"] for a in frame.raw["attribute"]}
            assert attrs["dx"] == pytest.approx(4e-9)
            assert (
                attrs["x0"] == 0.0 and attrs["samples"] == 3 and attrs["name"] == "Ch0"
            )
            sub.close()
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
        await dev.stop()


# ---------------------------------------------------- end to end: stack
@pytest.mark.timeout(30)
async def test_the_plugin_writes_a_1d_float_stack_and_drops_over_ceiling_frames(
    tmp_path,
):
    """1-D float64 frames: ArraySizeY 0, DataType Float64, a (N, n) stack;
    an over-ceiling frame is counted as a shape error and never truncated."""
    dev = ArrayDevice("interpSpec")
    await dev.start()
    gateway, task = await _start_gateway(dev, "interpSpec", ceiling=4)
    plugin = gateway._workers[0].plugins["interpSpec"]
    prefix = "testexp:u_spec:interpspec" + file_plugin.PLUGIN_SUFFIX
    ctx = Context("pva", conf=gateway.conf(), useenv=False)
    loop = asyncio.get_running_loop()

    async def put(suffix: str, value) -> None:
        await loop.run_in_executor(None, lambda: ctx.put(prefix + suffix, value))

    async def get(suffix: str):
        return await loop.run_in_executor(None, lambda: ctx.get(prefix + suffix))

    try:
        run_dir = tmp_path / "Scan003" / "U_Spec-interpSpec"
        run_dir.mkdir(parents=True)
        await put("FilePath", str(run_dir) + os.sep)
        await put("FileName", "U_Spec-interpSpec")
        # A CSV payload is a 1-D array: padded to (4,), posted 1-D.
        dev.push(b"1.0,2.0\r\n", time.time() - 5.0)
        await put("Capture", True)
        assert bool(await get("Capture_RBV")) is True
        assert int(await get("ArraySizeX_RBV")) == 4
        assert int(await get("ArraySizeY_RBV")) == 0
        assert str(await get("DataType_RBV")) == "Float64"

        t = time.time()
        dev.push(b"1.0,2.0", t)
        dev.push(b"3.0", t + 1)
        dev.push(b"1,2,3,4,5", t + 2)  # five > ceiling 4: dropped, counted
        dev.push(b"4.0,5.0,6.0,7.0", t + 3)
        await _wait_until(lambda: plugin.value("NumCaptured_RBV") == 3)
        await asyncio.sleep(0.1)
        assert plugin.value("NumCaptured_RBV") == 3
        assert "ceiling" in str(plugin.value("WriteMessage"))
        await put("Capture", False)
        await asyncio.wait_for(dev.disconnected.wait(), 10)

        with h5py.File(run_dir / "U_Spec-interpSpec.h5", "r") as f:
            frames = f[file_plugin.FRAMES_DATASET]
            assert frames.shape == (3, 4) and frames.dtype == np.float64
            np.testing.assert_array_equal(frames[0][:2], [1.0, 2.0])
            assert np.isnan(frames[0][2:]).all()
            assert frames[1][0] == 3.0 and np.isnan(frames[1][1:]).all()
            np.testing.assert_array_equal(frames[2], [4.0, 5.0, 6.0, 7.0])
            assert f.attrs["shape_errors"] == 1
            assert f.attrs["frames_written"] == 3
            # An array stack always carries the axis attributes; a CSV
            # lineout has no axis, so they read NaN.
            dx = f[f"{file_plugin.ATTRIBUTES_GROUP}/u_spec-hdf-interpspec-wave_dx"][:]
            assert dx.shape == (3,) and np.isnan(dx).all()
    finally:
        ctx.close()
        await _shutdown(task)
        await dev.stop()


@pytest.mark.timeout(30)
async def test_a_waveform_stack_carries_its_time_axis_as_attributes(tmp_path):
    """A captured scope trace is never an axis-less array: x0, dx and the record
    length ride as per-frame attributes beside the stamps (review of #946)."""
    dev = ArrayDevice("scopeTrace.Channel0")
    await dev.start()
    gateway, task = await _start_gateway(dev, "scopeTrace.Channel0", ceiling=None)
    plugin = gateway._workers[0].plugins["scopeTrace.Channel0"]
    prefix = "testexp:u_spec:scopetrace_channel0" + file_plugin.PLUGIN_SUFFIX
    ctx = Context("pva", conf=gateway.conf(), useenv=False)
    loop = asyncio.get_running_loop()

    async def put(suffix: str, value) -> None:
        await loop.run_in_executor(None, lambda: ctx.put(prefix + suffix, value))

    try:
        run_dir = tmp_path / "Scan004" / "U_Spec"
        run_dir.mkdir(parents=True)
        await put("FilePath", str(run_dir) + os.sep)
        await put("FileName", "U_Spec")
        dev.push(_waveform([0, 1, 2], dx=4e-9), time.time() - 5.0)
        await put("Capture", True)
        t = time.time()
        dev.push(_waveform([0, 1000, -1000], dx=4e-9), t)
        dev.push(_waveform([5, 6, 7], dx=8e-9), t + 1)
        await _wait_until(lambda: plugin.value("NumCaptured_RBV") == 2)
        await put("Capture", False)
        await asyncio.wait_for(dev.disconnected.wait(), 10)
        with h5py.File(run_dir / "U_Spec.h5", "r") as f:
            group = f[file_plugin.ATTRIBUTES_GROUP]
            dx = group["u_spec-hdf-scopetrace_channel0-wave_dx"][:]
            np.testing.assert_allclose(dx, [4e-9, 8e-9])
            assert group["u_spec-hdf-scopetrace_channel0-wave_x0"][:].tolist() == [
                0.0,
                0.0,
            ]
            assert group["u_spec-hdf-scopetrace_channel0-wave_samples"][:].tolist() == [
                3.0,
                3.0,
            ]
            assert list(f.attrs["waveform_attributes"]) == [
                "u_spec-hdf-scopetrace_channel0-wave_x0",
                "u_spec-hdf-scopetrace_channel0-wave_dx",
                "u_spec-hdf-scopetrace_channel0-wave_samples",
            ]
            assert list(f.attrs["scalar_attributes"]) == []  # the axis is not a scalar
    finally:
        ctx.close()
        await _shutdown(task)
        await dev.stop()


# ---------------------------------------------------------------- config
def test_config_serves_arrays_minus_the_devicetype_exclusions(monkeypatch) -> None:
    """A MagSpec camera: images + interpSpec/interpDiv served, the axes excluded,
    the ceiling declared; a scope with only arrays is served; a timing box is not."""
    from geecs_core.db.geecs_db import GeecsDb

    endpoints = {
        "UC_MagCam": ("192.168.8.201", 65001),
        "U_ICT": ("192.168.8.201", 65002),
        "U_Timing": ("192.168.8.201", 65003),
    }
    arr = {"variabletype": "1darray", "choices": None}
    var_map = {
        "UC_MagCam": [
            {"name": "Image", "variabletype": "image", "choices": None},
            {"name": "ImageInterp", "variabletype": "image", "choices": None},
            {"name": "interpSpec", **arr},
            {"name": "interpDiv", **arr},
            {"name": "EnergyAxis", **arr},
            {"name": "AngleAxis", **arr},
        ],
        "U_ICT": [
            {"name": "scopeTrace.Channel0", **arr},
            {"name": "scopeTraceGUI.Channel0", **arr},
            {"name": "wfm", **arr},
        ],
        "U_Timing": [{"name": "delay", "variabletype": "numeric", "choices": None}],
    }
    types = {"UC_MagCam": "MagSpecCamera", "U_ICT": "PicoscopeV2", "U_Timing": "DG645"}
    monkeypatch.setattr(
        GeecsDb, "get_experiment_devices", classmethod(lambda cls, e, **kw: endpoints)
    )
    monkeypatch.setattr(
        GeecsDb,
        "get_experiment_device_variables",
        classmethod(lambda cls, e, **kw: var_map),
    )
    monkeypatch.setattr(
        GeecsDb, "get_experiment_device_types", classmethod(lambda cls, e, **kw: types)
    )
    monkeypatch.setattr(
        GeecsDb, "get_subscribed_variables", classmethod(lambda cls, e, **kw: {})
    )

    cfg = PvaGatewayConfig.from_geecs_experiment("TestExp", host="192.168.8.201")
    by_dev = {d.device: d for d in cfg.devices}
    assert set(by_dev) == {"UC_MagCam", "U_ICT"}
    cam = by_dev["UC_MagCam"]
    assert cam.image_variables == ["Image", "ImageInterp"]
    assert cam.array_variables == ["interpDiv", "interpSpec"]
    assert cam.array_ceiling == 16384 and cam.devicetype == "MagSpecCamera"
    ict = by_dev["U_ICT"]
    assert ict.image_variables == [] and ict.array_variables == ["scopeTrace.Channel0"]
    assert ict.array_ceiling is None
    gateway = GeecsPvaGateway(cfg)
    assert "testexp:u_ict:scopetrace_channel0" in gateway.pv_names
    assert "testexp:u_ict:scopetracegui_channel0" not in gateway.pv_names
    assert "testexp:uc_magcam:energyaxis" not in gateway.pv_names
