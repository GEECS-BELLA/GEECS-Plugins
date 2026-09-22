"""End-to-end server tests: fake GEECS camera -> gateway -> PVA client.

Hermetic: a hand-rolled push server speaks the GEECS wire format (the fake
server in geecs_core.testing is ASCII-only, and image payloads are
binary), and the PVA server runs with ``isolate=True`` (sandboxed ports).
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import struct
import threading

import numpy as np
import pytest
from p4p.client.thread import Context

from geecs_pva_gateway.config import DeviceSpec, PvaGatewayConfig
from geecs_pva_gateway.server import GeecsPvaGateway, __version__

pytestmark = pytest.mark.fake_server

DEVICE = b"UC_TestCam"
# A LabVIEW-epoch acquisition timestamp and its Unix equivalent.
LV_TIMESTAMP = 3_862_844_800.25
UNIX_TIMESTAMP = LV_TIMESTAMP - 2_082_844_800

IMG = (np.arange(4 * 6) % 200).astype("<u2").reshape(4, 6)


def _imaq_blob(img: np.ndarray) -> bytes:
    """Anchored IMAQ flatten wrapper around ``img`` (no border, no padding)."""
    height, width = img.shape
    sb = bytearray(64)
    struct.pack_into("<i", sb, 36, 7)  # Grayscale U16
    struct.pack_into("<i", sb, 40, width)
    struct.pack_into("<i", sb, 48, height)
    struct.pack_into("<i", sb, 56, 0)
    prefixed = struct.pack(">I", len(DEVICE)) + DEVICE
    wrapper = b"nivissvc.*" + b"LV_ImageDTClassInfo" + b"\x00" * 8 + bytes(sb)
    return prefixed + wrapper + prefixed + img.tobytes()


def _push_frame() -> bytes:
    body = (
        DEVICE
        + b">>7>>image nval,"
        + _imaq_blob(IMG)
        + b" nvar,acq_timestamp nval,%.2f nvar" % LV_TIMESTAMP
    )
    return struct.pack(">i", len(body)) + body


class FakeCamera:
    """Minimal GEECS push server: one Wait>> handshake, then framed pushes."""

    def __init__(self, drop_after: int | None = None) -> None:
        self.drop_after = drop_after
        self.connections = 0
        self.total_connections = 0
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

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connections += 1
        self.total_connections += 1
        self.connected.set()
        frame = _push_frame()
        sent = 0
        eof_task: asyncio.Task | None = None
        try:
            header = await reader.readexactly(4)
            await reader.readexactly(struct.unpack(">i", header)[0])  # Wait>>
            # The subscriber never writes again; read() returning marks its FIN.
            eof_task = asyncio.create_task(reader.read())
            while not eof_task.done() and (
                self.drop_after is None or sent < self.drop_after
            ):
                writer.write(frame)
                await writer.drain()
                sent += 1
                await asyncio.sleep(0.05)
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


async def _start_gateway(cam: FakeCamera) -> tuple[GeecsPvaGateway, asyncio.Task]:
    spec = DeviceSpec(
        device=DEVICE.decode(),
        host="127.0.0.1",
        port=cam.port,
        experiment="testexp",
        image_variables=["image"],
    )
    gateway = GeecsPvaGateway(PvaGatewayConfig(experiment="testexp", devices=[spec]))
    task = asyncio.create_task(gateway.run(isolate=True))
    for _ in range(100):  # wait for the server to come up
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


@pytest.mark.timeout(30)
async def test_frames_flow_and_subscription_is_gated():
    """Frames reach a PVA monitor; the GEECS subscription follows the clients."""
    cam = FakeCamera()
    await cam.start()
    gateway, task = await _start_gateway(cam)
    try:
        # No PVA clients yet: the camera must not be subscribed.
        assert cam.connections == 0

        received: list = []
        got_real_frame = threading.Event()

        def on_update(value) -> None:  # p4p worker thread
            received.append(value)
            if value.shape == IMG.shape:
                got_real_frame.set()

        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        loop = asyncio.get_running_loop()
        connected_pv = "testexp:uc_testcam:image:connected"
        try:
            # Gated off: the subscription state is Idle (nothing is known).
            idle = await loop.run_in_executor(None, ctx.get, connected_pv)
            assert str(idle) == "Idle"
            sub = ctx.monitor("testexp:uc_testcam:image", on_update)
            await asyncio.wait_for(cam.connected.wait(), 5)  # gating: started
            await loop.run_in_executor(None, got_real_frame.wait, 10)
            assert got_real_frame.is_set()
            live = await loop.run_in_executor(None, ctx.get, connected_pv)
            assert str(live) == "Connected" and live.raw["alarm.severity"] == 0

            frame = received[-1]
            assert frame.shape == IMG.shape
            np.testing.assert_array_equal(np.asarray(frame), IMG)
            # GEECS timestamp ladder: acq_timestamp, LabVIEW -> Unix epoch.
            assert frame.timestamp == pytest.approx(UNIX_TIMESTAMP, abs=0.01)

            sub.close()
        finally:
            # p4p's Context caches channels: the client-side channel (and with
            # it onLastDisconnect) closes with the context, not the monitor.
            ctx.close()
        await asyncio.wait_for(cam.disconnected.wait(), 10)  # gating: stopped
        ctx2 = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            for _ in range(50):  # the supervisor's cancel lands after the socket
                idle = await loop.run_in_executor(None, ctx2.get, connected_pv)
                if str(idle) == "Idle":
                    break
                await asyncio.sleep(0.1)
            assert str(idle) == "Idle"
        finally:
            ctx2.close()
    finally:
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(30)
async def test_dropped_device_connection_reconnects():
    """A socket drop while clients are attached triggers reconnect + frames."""
    cam = FakeCamera(drop_after=3)
    await cam.start()
    gateway, task = await _start_gateway(cam)
    try:
        count = [0]

        def on_update(value) -> None:
            if value.shape == IMG.shape:
                count[0] += 1

        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            # Bind the subscription: unbound, it is GC'd and the monitor closes.
            sub = ctx.monitor("testexp:uc_testcam:image", on_update)
            # Each connection drops itself after 3 frames; >= 3 connections
            # proves the supervisor reconnects repeatedly. (Frame counts are
            # unreliable here: p4p monitors squash rapid updates client-side.)
            for _ in range(200):
                if cam.total_connections >= 3:
                    break
                await asyncio.sleep(0.1)
            assert cam.total_connections >= 3
            assert count[0] >= 1  # real frames flowed across reconnects
            sub.close()
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(30)
async def test_unreachable_device_is_re_resolved_at_the_backoff_ceiling(monkeypatch):
    """A watched device that stays unreachable reads Disconnected (MAJOR), and
    once the backoff sits at its ceiling the endpoint is re-asked of the DB
    and redialed there — a camera app that came up on another port after the
    gateway started is found without a restart (#854)."""
    from geecs_pva_gateway import server as server_module

    monkeypatch.setattr(server_module, "_RECONNECT_MIN_S", 0.05)
    monkeypatch.setattr(server_module, "_RECONNECT_MAX_S", 0.1)
    monkeypatch.setattr(server_module, "_ENDPOINT_RESOLVE_HOLDOFF_CYCLES", 0)
    cam = FakeCamera()
    await cam.start()
    # The endpoint the DB held at startup: a port nobody listens on.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()
    resolved: list[str] = []

    def resolver(device: str) -> tuple[str, int]:
        resolved.append(device)
        return ("127.0.0.1", cam.port)

    spec = DeviceSpec(
        device=DEVICE.decode(),
        host="127.0.0.1",
        port=dead_port,
        experiment="testexp",
        image_variables=["image"],
    )
    gateway = GeecsPvaGateway(
        PvaGatewayConfig(experiment="testexp", devices=[spec]),
        endpoint_resolver=resolver,
    )
    task = asyncio.create_task(gateway.run(isolate=True))
    for _ in range(100):
        await asyncio.sleep(0.05)
        try:
            gateway.conf()
            break
        except AssertionError:
            continue
    try:
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            states: list[tuple[str, int]] = []
            state_sub = ctx.monitor(
                "testexp:uc_testcam:image:connected",
                lambda v: states.append((str(v), int(v.raw["alarm.severity"]))),
            )
            sub = ctx.monitor("testexp:uc_testcam:image", lambda v: None)
            for _ in range(100):
                if cam.total_connections >= 1:
                    break
                await asyncio.sleep(0.1)
            assert cam.total_connections >= 1  # redialed on the resolved port
            assert resolved == [DEVICE.decode()]  # once, at the ceiling
            await asyncio.sleep(0.3)
            assert ("Disconnected", 2) in states  # visible while unreachable
            assert states[-1] == ("Connected", 0)
            sub.close()
            state_sub.close()
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(30)
async def test_cross_camera_pv_name_collision_raises():
    """Two cameras normalizing to one PV name refuse to start, loudly."""
    specs = [
        DeviceSpec(
            device="UC_Cam-A",
            host="127.0.0.1",
            port=1,
            experiment="testexp",
            image_variables=["image"],
        ),
        DeviceSpec(
            device="UC_Cam_A",
            host="127.0.0.1",
            port=2,
            experiment="testexp",
            image_variables=["image"],
        ),
    ]
    gateway = GeecsPvaGateway(PvaGatewayConfig(experiment="testexp", devices=specs))
    with pytest.raises(ValueError, match="collision"):
        await gateway.run(isolate=True)


@pytest.mark.timeout(30)
async def test_within_camera_pv_name_collision_raises():
    """Two variables of ONE camera normalizing to one name refuse to start.

    These collapse in a per-camera dict before a cross-worker guard sees
    them, so the guard must inspect per-variable entries.
    """
    spec = DeviceSpec(
        device="UC_Cam",
        host="127.0.0.1",
        port=1,
        experiment="testexp",
        image_variables=["processed image", "processed_image"],
    )
    gateway = GeecsPvaGateway(PvaGatewayConfig(experiment="testexp", devices=[spec]))
    with pytest.raises(ValueError, match="collision"):
        await gateway.run(isolate=True)


@pytest.mark.timeout(30)
async def test_gating_is_per_variable():
    """Watching one variable subscribes only that variable's connection."""
    cam = FakeCamera()
    await cam.start()
    spec = DeviceSpec(
        device=DEVICE.decode(),
        host="127.0.0.1",
        port=cam.port,
        experiment="testexp",
        image_variables=["bakground image", "image"],
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
    try:
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            sub1 = ctx.monitor("testexp:uc_testcam:image", lambda v: None)
            await asyncio.wait_for(cam.connected.wait(), 5)
            await asyncio.sleep(0.3)  # settle: would a second conn appear?
            assert cam.connections == 1  # only the watched variable subscribes

            sub2 = ctx.monitor("testexp:uc_testcam:bakground_image", lambda v: None)
            for _ in range(100):
                if cam.connections == 2:
                    break
                await asyncio.sleep(0.05)
            assert cam.connections == 2  # each watched variable has its own
            sub1.close()
            sub2.close()
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(30)
async def test_instance_identity_pvs():
    """version/heartbeat PVs exist under the instance prefix."""
    cam = FakeCamera()
    await cam.start()
    gateway, task = await _start_gateway(cam)
    try:
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            loop = asyncio.get_running_loop()
            version = await loop.run_in_executor(
                None, lambda: ctx.get("testexp:pvagateway:127_0_0_1:version")
            )
            assert version == __version__  # ntstr subclasses str
            heartbeat = await loop.run_in_executor(
                None, lambda: ctx.get("testexp:pvagateway:127_0_0_1:heartbeat")
            )
            assert int(heartbeat) >= 0
        finally:
            ctx.close()
    finally:
        await _shutdown(task)
        await cam.stop()


@pytest.mark.timeout(30)
async def test_restart_pv_shuts_down_cleanly():
    """A put to the :restart PV ends run() cleanly with restart_requested set."""
    cam = FakeCamera()
    await cam.start()
    gateway, task = await _start_gateway(cam)
    try:
        assert not gateway.restart_requested
        ctx = Context("pva", conf=gateway.conf(), useenv=False)
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, lambda: ctx.put("testexp:pvagateway:127_0_0_1:restart", 1)
            )
        finally:
            ctx.close()
        await asyncio.wait_for(task, 10)  # run() returns on its own — no cancel
        assert gateway.restart_requested
    finally:
        if not task.done():
            await _shutdown(task)
        await cam.stop()


def test_subscription_carries_the_scalars_only_with_a_plugin(monkeypatch) -> None:
    """One TCP subscription: frame + stamps, plus the subscribed scalars when the
    file plugin serves the variable — never without it."""
    from geecs_pva_gateway import file_plugin
    from geecs_pva_gateway.server import _DeviceWorker

    spec = DeviceSpec(
        device="UC_TestCam",
        host="127.0.0.1",
        port=1,
        experiment="testexp",
        image_variables=["image"],
        scalar_variables=["MaxCounts", "acq_timestamp", "exposure"],
    )
    loop = asyncio.new_event_loop()
    try:
        worker = _DeviceWorker(spec, loop)
        try:
            assert worker.plugins  # h5py is installed here
            assert worker.subscription_variables("image") == [
                "image",
                "acq_timestamp",
                "systimestamp",
                "MaxCounts",
                "exposure",
            ]
            assert worker.plugins["image"].scalar_variables == (
                "MaxCounts",
                "acq_timestamp",
                "exposure",
            )
        finally:
            for plugin in worker.plugins.values():
                plugin.stop()
        monkeypatch.setattr(file_plugin, "available", lambda: False)
        bare = _DeviceWorker(spec, loop)
        assert bare.subscription_variables("image") == [
            "image",
            "acq_timestamp",
            "systimestamp",
        ]
    finally:
        loop.close()
