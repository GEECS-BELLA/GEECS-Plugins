"""Offline gateway tests against the in-process FakeGeecsServer.

These verify the two data paths without any CA client or lab network:

* stream → readback channel value
* setpoint channel write → GEECS device (caput semantics)
"""

from __future__ import annotations

import asyncio
import math
import socket

import pytest
from caproto import AlarmSeverity
from geecs_ca_gateway.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer

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


async def test_timestamp_vars_exposed_as_pvs_with_raw_value() -> None:
    """systimestamp/acq_timestamp become float PVs carrying the raw LabVIEW value."""
    labview = LABVIEW_OFFSET + 1000.0
    device = FakeGeecsDevice(
        DEVICE, variables={"Position": 1.0, "systimestamp": labview}
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
        # both intrinsic timestamp vars get PVs (acq even if this device won't push it)
        assert f"{DEVICE}:systimestamp" in gw.pvdb
        assert f"{DEVICE}:acq_timestamp" in gw.pvdb
        await gw.connect()
        await gw.subscribe()
        try:
            rb = gw.pvdb[f"{DEVICE}:systimestamp"]
            # RAW LabVIEW value, not unix-converted
            assert await _wait_until(lambda: rb.value == pytest.approx(labview))
        finally:
            await gw.close()


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


def _enum_label(channel) -> str:
    """The current option label of an enum channel, as a str."""
    v = channel.value
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


async def test_enum_readback_and_setpoint() -> None:
    """Enum: GEECS string ↔ enum index in both directions, over the gateway."""
    device = FakeGeecsDevice(DEVICE, variables={"Enable_Output": "off"})
    async with FakeGeecsServer(device) as srv:
        cfg = GatewayConfig(
            devices=[
                DeviceSpec(
                    name=DEVICE,
                    host=srv.host,
                    port=srv.port,
                    variables=[
                        VariableSpec(
                            geecs_var="Enable_Output",
                            dtype="enum",
                            settable=True,
                            choices=["on", "off"],
                        )
                    ],
                )
            ]
        )
        gw = GeecsCaGateway(cfg)
        await gw.connect()
        await gw.subscribe()
        rb = gw.pvdb[f"{DEVICE}:Enable_Output"]
        sp = gw.pvdb[f"{DEVICE}:Enable_Output:SP"]
        try:
            # readback: GEECS "off" string -> enum label "off"
            assert await _wait_until(lambda: _enum_label(rb) == "off")
            # setpoint: CA put index 0 -> GEECS "on"
            await sp.write(0)
            assert device.get("Enable_Output") == "on"
            # readback follows back to "on"
            assert await _wait_until(lambda: _enum_label(rb) == "on")
        finally:
            await gw.close()


async def test_deadband_suppresses_small_changes() -> None:
    """A float change within the deadband is not re-posted; a larger one is."""
    device = FakeGeecsDevice(DEVICE, variables={"Pos": 5.0})
    async with FakeGeecsServer(device) as srv:
        cfg = GatewayConfig(
            devices=[
                DeviceSpec(
                    name=DEVICE,
                    host=srv.host,
                    port=srv.port,
                    variables=[
                        VariableSpec(geecs_var="Pos", dtype="float", deadband=0.1)
                    ],
                )
            ]
        )
        gw = GeecsCaGateway(cfg)
        await gw.connect()
        await gw.subscribe()
        rb = gw.pvdb[f"{DEVICE}:Pos"]
        try:
            assert await _wait_until(lambda: rb.value == pytest.approx(5.0))
            # within deadband (0.05 <= 0.1): suppressed, value stays 5.0
            device.variables["Pos"] = 5.05
            await asyncio.sleep(0.5)
            assert rb.value == pytest.approx(5.0)
            # beyond deadband: posts
            device.variables["Pos"] = 5.5
            assert await _wait_until(lambda: rb.value == pytest.approx(5.5))
        finally:
            await gw.close()


async def test_nan_readback_accepted_despite_limits() -> None:
    """A NaN readback (e.g. failed analysis) is reported, not rejected by limits."""
    device = FakeGeecsDevice(DEVICE, variables={"Pos": "NaN"})
    async with FakeGeecsServer(device) as srv:
        cfg = GatewayConfig(
            devices=[
                DeviceSpec(
                    name=DEVICE,
                    host=srv.host,
                    port=srv.port,
                    variables=[
                        VariableSpec(geecs_var="Pos", dtype="float", lo=0.0, hi=100.0)
                    ],
                )
            ]
        )
        gw = GeecsCaGateway(cfg)
        await gw.connect()
        await gw.subscribe()
        rb = gw.pvdb[f"{DEVICE}:Pos"]
        try:
            await asyncio.sleep(0.4)
            value = rb.value
            if hasattr(value, "__len__") and not isinstance(value, str):
                value = value[0]
            assert math.isnan(value)  # accepted, not clamped/rejected to 0
        finally:
            await gw.close()


async def test_uncoercible_value_warns_once_and_skips() -> None:
    """A float PV fed a string (DB type mismatch) skips it and warns just once."""
    device = FakeGeecsDevice(
        DEVICE, variables={"Pos": "off"}
    )  # float var, string value
    async with FakeGeecsServer(device) as srv:
        cfg = GatewayConfig(
            devices=[
                DeviceSpec(
                    name=DEVICE,
                    host=srv.host,
                    port=srv.port,
                    variables=[VariableSpec(geecs_var="Pos", dtype="float")],
                )
            ]
        )
        gw = GeecsCaGateway(cfg)
        await gw.connect()
        await gw.subscribe()
        try:
            await asyncio.sleep(0.5)  # several 5 Hz frames of the bad value
            # value skipped (PV stays at initial), no crash, warned exactly once
            assert gw.pvdb[f"{DEVICE}:Pos"].value == pytest.approx(0.0)
            assert (DEVICE, "Pos") in gw._coerce_warned
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


# ---------------------------------------------------------------------------
# Self-diagnostics: read-only readbacks + status PVs
# ---------------------------------------------------------------------------


def test_readback_channels_deny_client_writes() -> None:
    """Readbacks report READ-only access; setpoints keep WRITE access.

    Regression: a mistaken caput to a readback used to *stick* (the deadband
    cache suppressed the next unchanged hardware frame).
    """
    from caproto import AccessRights

    from geecs_ca_gateway.channels import make_readback_channel, make_setpoint_channel
    from geecs_ca_gateway.config import VariableSpec

    for dtype in ("float", "int", "string", "path"):
        rb = make_readback_channel(VariableSpec(geecs_var="x", dtype=dtype))
        assert rb.check_access("h", "u") == AccessRights.READ, dtype
    rb = make_readback_channel(
        VariableSpec(geecs_var="x", dtype="enum", choices=["on", "off"])
    )
    assert rb.check_access("h", "u") == AccessRights.READ

    async def setter(value):
        return value

    sp = make_setpoint_channel(
        VariableSpec(geecs_var="x", dtype="float", settable=True), setter
    )
    assert AccessRights.WRITE in sp.check_access("h", "u")


def test_pvdb_has_connected_and_gateway_status_pvs() -> None:
    """Every device gets a CONNECTED PV; the gateway exposes devIocStats-style PVs."""
    spec = DeviceSpec(
        name="U_Dev",
        host="h",
        port=1,
        experiment="Test",
        variables=[VariableSpec(geecs_var="Val")],
    )
    gw = GeecsCaGateway(GatewayConfig(devices=[spec]))

    assert "Test:U_Dev:CONNECTED" in gw.pvdb
    assert str(gw.pvdb["Test:U_Dev:CONNECTED"].value) == "Disconnected"
    for suffix in ("UPTIME", "HEARTBEAT", "DEVICES_CONNECTED", "VERSION"):
        assert f"Test:CAGateway:{suffix}" in gw.pvdb
    assert gw.manifest["Test:U_Dev:CONNECTED"] == ("U_Dev", "CONNECTED", "status")


async def test_set_connected_updates_pv_severity_and_count() -> None:
    """CONNECTED transitions carry MAJOR severity when down, count when up."""
    from caproto import AlarmSeverity

    spec = DeviceSpec(
        name="U_Dev",
        host="h",
        port=1,
        experiment="Test",
        variables=[VariableSpec(geecs_var="Val")],
    )
    gw = GeecsCaGateway(GatewayConfig(devices=[spec]))

    await gw._set_connected("U_Dev", True)
    conn = gw.pvdb["Test:U_Dev:CONNECTED"]
    assert str(conn.value) == "Connected"
    assert conn.alarm.severity == AlarmSeverity.NO_ALARM
    assert gw.pvdb["Test:CAGateway:DEVICES_CONNECTED"].value in (1, [1])

    await gw._set_connected("U_Dev", False)
    assert str(conn.value) == "Disconnected"
    assert conn.alarm.severity == AlarmSeverity.MAJOR_ALARM
    assert gw.pvdb["Test:CAGateway:DEVICES_CONNECTED"].value in (0, [0])
