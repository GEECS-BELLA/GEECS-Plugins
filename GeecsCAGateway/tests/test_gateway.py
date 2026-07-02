"""Offline gateway tests against the in-process FakeGeecsServer.

These verify the two data paths without any CA client or lab network:

* stream → readback channel value
* setpoint channel write → GEECS device (caput semantics)
"""

from __future__ import annotations

import asyncio
import socket
import struct

import pytest
from caproto import AlarmSeverity
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer

from geecs_ca_gateway.config import DeviceSpec, GatewayConfig, VariableSpec
from geecs_ca_gateway.gateway import GeecsCaGateway, _extract_timestamp

pytestmark = pytest.mark.fake_server

DEVICE = "U_ESP_JetXYZ"
LABVIEW_OFFSET = 2_082_844_800  # LabVIEW epoch (1904) → Unix epoch (1970)


def test_extract_timestamp_converts_labview_to_unix() -> None:
    """`systimestamp` (LabVIEW epoch) converts to a Unix-epoch timestamp."""
    unix = _extract_timestamp(
        {"systimestamp": LABVIEW_OFFSET + 1000.5}, ["systimestamp"]
    )
    assert unix == pytest.approx(1000.5)


def test_extract_timestamp_ladder_prefers_first_present() -> None:
    """The ladder tries variables in order; missing rungs fall through."""
    frame = {
        "acq_timestamp": LABVIEW_OFFSET + 5.0,
        "systimestamp": LABVIEW_OFFSET + 9.0,
    }
    ladder = ["acq_timestamp", "systimestamp"]
    assert _extract_timestamp(frame, ladder) == pytest.approx(5.0)
    assert _extract_timestamp(
        {"systimestamp": LABVIEW_OFFSET + 9.0}, ladder
    ) == pytest.approx(9.0)


def test_extract_timestamp_none_when_absent_or_implausible() -> None:
    """No timestamp var, or a nonsense (pre-1970) value, yields None."""
    assert _extract_timestamp({"Position": 1.0}, ["systimestamp"]) is None
    assert _extract_timestamp({"systimestamp": 100.0}, ["systimestamp"]) is None


def _free_port() -> int:
    """Grab a currently-free localhost TCP port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _wait_until(predicate, timeout: float = 6.0, interval: float = 0.05) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses."""
    waited = 0.0
    while waited < timeout:
        if predicate():
            return True
        await asyncio.sleep(interval)
        waited += interval
    return predicate()


async def _start_on_port(device: FakeGeecsDevice, port: int) -> FakeGeecsServer:
    """Start a fake server on a specific port, retrying while it frees up."""
    last: OSError | None = None
    for _ in range(15):
        srv = FakeGeecsServer(device, port=port)
        try:
            await srv.start()
            return srv
        except OSError as exc:  # port not yet released after prior close
            last = exc
            await asyncio.sleep(0.1)
    raise last  # type: ignore[misc]


def _config(host: str, port: int) -> GatewayConfig:
    return GatewayConfig(
        devices=[
            DeviceSpec(
                name=DEVICE,
                host=host,
                port=port,
                variables=[
                    VariableSpec(
                        geecs_var="Position", dtype="float", settable=True, egu="mm"
                    ),
                    VariableSpec(geecs_var="acq_timestamp", dtype="float"),
                ],
            )
        ]
    )


def test_experiment_prefix_and_manifest() -> None:
    """Experiment prefix flows into PV names; the manifest maps PV → GEECS var."""
    cfg = GatewayConfig(
        devices=[
            DeviceSpec(
                name="U_S1H",
                host="h",
                port=1,
                experiment="Undulator",
                variables=[VariableSpec(geecs_var="Current", settable=True)],
            )
        ]
    )
    gw = GeecsCaGateway(cfg)
    assert "Undulator:U_S1H:Current" in gw.pvdb
    assert "Undulator:U_S1H:Current:SP" in gw.pvdb
    assert gw.manifest["Undulator:U_S1H:Current"] == ("U_S1H", "Current", "readback")
    assert gw.manifest["Undulator:U_S1H:Current:SP"] == ("U_S1H", "Current", "setpoint")


def test_pv_name_collision_raises() -> None:
    """Two GEECS variables mapping to one PV is a hard error, not silent clobber."""
    cfg = GatewayConfig(
        devices=[
            DeviceSpec(
                name="U_DG645",
                host="h",
                port=1,
                # "Trigger.Source" and "Trigger Source" both normalize identically.
                variables=[
                    VariableSpec(geecs_var="Trigger.Source"),
                    VariableSpec(geecs_var="Trigger Source"),
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="collision"):
        GeecsCaGateway(cfg)


async def test_pvdb_contains_readback_and_setpoint() -> None:
    """Settable variable yields both a readback and a ``:SP`` setpoint PV."""
    gw = GeecsCaGateway(_config("127.0.0.1", 1))
    assert f"{DEVICE}:Position" in gw.pvdb
    assert f"{DEVICE}:Position:SP" in gw.pvdb
    # non-settable variable has no setpoint
    assert f"{DEVICE}:acq_timestamp" in gw.pvdb
    assert f"{DEVICE}:acq_timestamp:SP" not in gw.pvdb


async def test_stream_updates_readback() -> None:
    """The 5 Hz subscription stream drives the readback channel value."""
    device = FakeGeecsDevice(
        DEVICE, variables={"Position": 7.5, "acq_timestamp": 1000.0}
    )
    async with FakeGeecsServer(device) as srv:
        gw = GeecsCaGateway(_config(srv.host, srv.port))
        await gw.connect()
        await gw.subscribe()
        try:
            await asyncio.sleep(0.4)  # a couple of 5 Hz frames
            assert gw.pvdb[f"{DEVICE}:Position"].value == pytest.approx(7.5)
        finally:
            await gw.close()


async def test_reconnect_and_validity() -> None:
    """On drop: readback goes INVALID; on reconnect: recovers and updates."""
    port = _free_port()
    device = FakeGeecsDevice(
        DEVICE, variables={"Position": 7.5, "acq_timestamp": 1000.0}
    )
    srv = await _start_on_port(device, port)
    gw = GeecsCaGateway(
        _config("127.0.0.1", port), reconnect_min_s=0.2, reconnect_max_s=0.4
    )
    await gw.connect()
    await gw.subscribe()
    rb = gw.pvdb[f"{DEVICE}:Position"]
    try:
        # live: value flows and severity is clear
        assert await _wait_until(lambda: rb.value == pytest.approx(7.5))
        assert int(rb.severity) == int(AlarmSeverity.NO_ALARM)

        # drop: the device goes away -> readback marked INVALID
        await srv.stop()
        assert await _wait_until(
            lambda: int(rb.severity) == int(AlarmSeverity.INVALID_ALARM)
        )

        # restore on the same port with a new value -> auto-recovers
        device2 = FakeGeecsDevice(
            DEVICE, variables={"Position": 9.0, "acq_timestamp": 1001.0}
        )
        srv2 = await _start_on_port(device2, port)
        try:
            assert await _wait_until(
                lambda: int(rb.severity) == int(AlarmSeverity.NO_ALARM)
                and rb.value == pytest.approx(9.0)
            )
        finally:
            await srv2.stop()
    finally:
        await gw.close()


async def _silent_after_frames_server(
    host: str, port: int, *, n_frames: int, position: float
) -> asyncio.AbstractServer:
    """A GEECS-like TCP server that pushes ``n_frames`` then goes silent.

    The socket stays open (no FIN), so only the stall watchdog — not
    socket-close detection — can notice the device has gone quiet.
    """

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            header = await reader.readexactly(4)
            await reader.readexactly(struct.unpack(">i", header)[0])  # "Wait>>..."
            for shot in range(1, n_frames + 1):
                msg = f"DEV>>{shot}>>Position nval,{position} nvar".encode("ascii")
                writer.write(struct.pack(">i", len(msg)) + msg)
                await writer.drain()
                await asyncio.sleep(0.1)
            # Go silent: push nothing, but exit cleanly when the client hangs up.
            while await reader.read(1):
                pass
        except (
            asyncio.IncompleteReadError,
            ConnectionResetError,
            asyncio.CancelledError,
        ):
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, host, port)


async def test_stall_watchdog_marks_invalid_on_silence() -> None:
    """A device that stops pushing (socket still open) is caught as stalled."""
    port = _free_port()
    server = await _silent_after_frames_server(
        "127.0.0.1", port, n_frames=5, position=7.5
    )
    gw = GeecsCaGateway(
        _config("127.0.0.1", port),
        reconnect_min_s=0.3,
        reconnect_max_s=0.5,
        stall_timeout_s=0.6,
    )
    await gw.connect()
    await gw.subscribe()
    rb = gw.pvdb[f"{DEVICE}:Position"]
    try:
        # frames flow -> valid
        assert await _wait_until(lambda: rb.value == pytest.approx(7.5))
        assert int(rb.severity) == int(AlarmSeverity.NO_ALARM)
        # server goes silent after 5 frames -> watchdog trips -> INVALID
        assert await _wait_until(
            lambda: int(rb.severity) == int(AlarmSeverity.INVALID_ALARM), timeout=4.0
        )
    finally:
        await gw.close()
        server.close()
        await server.wait_closed()


async def test_pv_timestamp_from_systimestamp() -> None:
    """A `systimestamp` frame stamps the readback PV with the converted time."""
    labview_val = LABVIEW_OFFSET + 1782949690.5
    device = FakeGeecsDevice(
        DEVICE, variables={"Position": 3.3, "systimestamp": labview_val}
    )
    async with FakeGeecsServer(device) as srv:
        cfg = GatewayConfig(
            devices=[
                DeviceSpec(
                    name=DEVICE,
                    host=srv.host,
                    port=srv.port,
                    variables=[VariableSpec(geecs_var="Position", dtype="float")],
                )
            ]
        )
        gw = GeecsCaGateway(cfg)
        await gw.connect()
        await gw.subscribe()
        rb = gw.pvdb[f"{DEVICE}:Position"]
        try:
            assert await _wait_until(lambda: rb.value == pytest.approx(3.3))
            assert rb.timestamp == pytest.approx(1782949690.5, abs=1e-3)
        finally:
            await gw.close()


async def test_setpoint_write_reaches_geecs() -> None:
    """Writing the setpoint channel forwards the value to the GEECS device."""
    device = FakeGeecsDevice(
        DEVICE, variables={"Position": 1.0, "acq_timestamp": 1000.0}
    )
    async with FakeGeecsServer(device) as srv:
        gw = GeecsCaGateway(_config(srv.host, srv.port))
        await gw.connect()
        await gw.subscribe()
        try:
            await gw.pvdb[f"{DEVICE}:Position:SP"].write(4.2)
            assert device.get("Position") == pytest.approx(4.2)
            # and the change streams back into the readback
            await asyncio.sleep(0.4)
            assert gw.pvdb[f"{DEVICE}:Position"].value == pytest.approx(4.2)
        finally:
            await gw.close()
