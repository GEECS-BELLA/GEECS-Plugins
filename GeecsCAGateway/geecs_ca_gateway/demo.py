"""Offline demo: serve a fake GEECS device as EPICS PVs, no hardware or network.

Run it::

    poetry run python -m geecs_ca_gateway.demo

It spins up an in-process ``FakeGeecsServer``, builds a gateway over it, performs
an in-process self-check (stream → readback, and caput → GEECS → readback), then
serves the PVs so you can poke them with real CA tools::

    caget  u_esp_jetxyz:position
    camonitor u_esp_jetxyz:position
    caput  u_esp_jetxyz:position:SP 4.2
"""

from __future__ import annotations

import asyncio
import logging

from geecs_ca_gateway.testing.fake_device_server import FakeGeecsDevice, FakeGeecsServer

from .config import DeviceSpec, GatewayConfig, VariableSpec
from .gateway import GeecsCaGateway
from .pv_naming import pv_name, setpoint_pv

DEVICE_NAME = "U_ESP_JetXYZ"


def _build_config(host: str, port: int) -> GatewayConfig:
    """Build a one-device demo config pointed at the fake server."""
    return GatewayConfig(
        devices=[
            DeviceSpec(
                name=DEVICE_NAME,
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


async def main() -> None:
    """Run the fake server + gateway, self-check, then serve until Ctrl-C."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    device = FakeGeecsDevice(
        DEVICE_NAME, variables={"Position": 1.23, "acq_timestamp": 1000.0}
    )
    async with FakeGeecsServer(device) as srv:
        gateway = GeecsCaGateway(_build_config(srv.host, srv.port))
        await gateway.connect()
        await gateway.subscribe()

        # ---- in-process self-check (proves the two data paths) --------------
        await asyncio.sleep(0.5)  # let a couple of 5 Hz stream frames land
        rb = gateway.pvdb[pv_name(DEVICE_NAME, "Position")]
        print(f"\n[self-check] stream → readback: {DEVICE_NAME}:Position = {rb.value}")

        sp = gateway.pvdb[setpoint_pv(pv_name(DEVICE_NAME, "Position"))]
        await sp.write(4.2)  # simulate a CA caput on the setpoint
        await asyncio.sleep(0.5)  # let the change stream back as a readback
        print(
            f"[self-check] caput 4.2 → GEECS device Position = {device.get('Position')}"
        )
        print(f"[self-check] readback followed: {DEVICE_NAME}:Position = {rb.value}\n")

        print("Serving PVs (Ctrl-C to stop):")
        for name in gateway.pvdb:
            print(f"  {name}")
        print()

        try:
            await gateway.serve()
        finally:
            await gateway.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
