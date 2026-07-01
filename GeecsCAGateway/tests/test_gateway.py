"""Offline gateway tests against the in-process FakeGeecsServer.

These verify the two data paths without any CA client or lab network:

* stream → readback channel value
* setpoint channel write → GEECS device (caput semantics)
"""

from __future__ import annotations

import asyncio

import pytest
from geecs_bluesky.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer

from geecs_ca_gateway.config import DeviceSpec, GatewayConfig, VariableSpec
from geecs_ca_gateway.gateway import GeecsCaGateway

pytestmark = pytest.mark.fake_server

DEVICE = "U_ESP_JetXYZ"


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
