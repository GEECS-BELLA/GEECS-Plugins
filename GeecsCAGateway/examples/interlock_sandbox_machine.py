"""Interlock sandbox, the machine half: fake GEECS devices behind the real CA gateway.

Offline, on this computer only.  Three ``FakeGeecsServer`` devices speak
the real GEECS wire protocol, and the real CA gateway serves them as PVs
under the ``Sandbox`` experiment prefix:

- ``U_FakeGauge`` — ``Pressure`` and a free-running ``acq_timestamp``
  (advanced twice a second while the gauge is alive): what the interlocks
  watch.
- ``U_FakeMagnet`` — a settable ``Current``: what the sandbox scan records
  and the interlocks' action plans set.
- ``U_FakeTrigger`` — a settable ``Trigger.Source``: the trigger box the
  scan drives (``scan`` while running, ``off`` while an interlock holds it).

Run it in one terminal, from the repository root, with the gateway's
environment::

    poetry -C GeecsCAGateway run python GeecsCAGateway/examples/interlock_sandbox_machine.py

then the worker half in a second terminal
(``GeecsBluesky/examples/interlock_sandbox_worker.py``).  Type commands
here to play the machine (``help`` lists them): change the pressure, freeze
the gauge's stamp, take the gauge down.  Every change the worker makes —
the magnet current, the trigger source — is printed as it lands.

The CA server binds to 127.0.0.1 only; set ``EPICS_CA_SERVER_PORT`` in
both terminals if port 5064 is taken here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time

# Serve on loopback only, before caproto reads its environment.
os.environ.setdefault("EPICS_CAS_INTF_ADDR_LIST", "127.0.0.1")
os.environ.setdefault("EPICS_CAS_AUTO_BEACON_ADDR_LIST", "NO")
os.environ.setdefault("EPICS_CAS_BEACON_ADDR_LIST", "127.0.0.1")

from geecs_core.testing.fake_device_server import (
    FakeGeecsDevice,
    FakeGeecsServer,
)

from geecs_ca_gateway.config import DeviceSpec, GatewayConfig, VariableSpec
from geecs_ca_gateway.gateway import GeecsCaGateway

EXPERIMENT = "Sandbox"
STAMP_PERIOD_S = 0.5

HELP = """\
commands:
  p <value>    set the gauge's Pressure (e.g. 'p 1e-3' trips the vacuum interlock)
  freeze       stop the gauge's acq_timestamp (the stale interlock trips)
  thaw         start it again
  down         take the gauge's server down (the gateway marks it Disconnected)
  up           bring it back
  show         print every device's values
  quit         stop the sandbox"""

# systimestamp is the gateway's timestamp fallback on every device (a
# static value is enough here); the gauge also has the acq_timestamp the
# stale interlock watches.
DEVICES = {
    "U_FakeGauge": {
        "Pressure": 1e-6,
        "acq_timestamp": time.time(),
        "systimestamp": 0.0,
    },
    "U_FakeMagnet": {"Current": 1.0, "systimestamp": 0.0},
    "U_FakeTrigger": {"Trigger.Source": "edges", "systimestamp": 0.0},
}
TIMESTAMP_VARS = {
    "U_FakeGauge": ["acq_timestamp", "systimestamp"],
    "U_FakeMagnet": ["systimestamp"],
    "U_FakeTrigger": ["systimestamp"],
}
VARIABLES = {
    "U_FakeGauge": [
        VariableSpec(geecs_var="Pressure", dtype="float", precision=8),
        VariableSpec(geecs_var="acq_timestamp", dtype="float", precision=6),
    ],
    "U_FakeMagnet": [
        VariableSpec(geecs_var="Current", dtype="float", settable=True, egu="A")
    ],
    "U_FakeTrigger": [
        VariableSpec(geecs_var="Trigger.Source", dtype="string", settable=True)
    ],
}


class Machine:
    """The fake devices, their servers and the gateway, on one asyncio loop."""

    def __init__(self) -> None:
        self.devices = {
            name: FakeGeecsDevice(name, variables=dict(values))
            for name, values in DEVICES.items()
        }
        self.servers: dict[str, FakeGeecsServer] = {}
        self.ports: dict[str, int] = {}
        self.ticking = True
        self.loop: asyncio.AbstractEventLoop | None = None
        self.gateway: GeecsCaGateway | None = None

    async def start(self) -> None:
        """Start one fake server per device, then the gateway over them."""
        self.loop = asyncio.get_running_loop()
        for name, device in self.devices.items():
            server = FakeGeecsServer(device)
            await server.start()
            self.servers[name] = server
            self.ports[name] = server.port
        self.gateway = GeecsCaGateway(
            GatewayConfig(
                devices=[
                    DeviceSpec(
                        name=name,
                        host="127.0.0.1",
                        port=self.ports[name],
                        experiment=EXPERIMENT,
                        variables=VARIABLES[name],
                        timestamp_vars=TIMESTAMP_VARS[name],
                    )
                    for name in self.devices
                ]
            )
        )
        await self.gateway.connect()
        await self.gateway.subscribe()

    async def gauge_down(self) -> None:
        """Stop the gauge's server (the gateway sees the connection drop)."""
        server = self.servers.pop("U_FakeGauge", None)
        if server is not None:
            await server.stop()

    async def gauge_up(self) -> None:
        """Restart the gauge's server on its old port (the gateway reconnects)."""
        if "U_FakeGauge" in self.servers:
            return
        server = FakeGeecsServer(
            self.devices["U_FakeGauge"], port=self.ports["U_FakeGauge"]
        )
        await server.start()
        self.servers["U_FakeGauge"] = server

    def run_on_loop(self, coroutine) -> None:
        """Run *coroutine* on the machine's loop from the prompt thread."""
        assert self.loop is not None
        asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(10)


def tick_stamp(machine: Machine, stop: threading.Event) -> None:
    """Advance the gauge's acq_timestamp while it is alive (a free-running device)."""
    gauge = machine.devices["U_FakeGauge"]
    while not stop.wait(STAMP_PERIOD_S):
        if machine.ticking and "U_FakeGauge" in machine.servers:
            gauge.set("acq_timestamp", time.time())


def report_changes(machine: Machine, stop: threading.Event) -> None:
    """Print every change the worker makes to the magnet and the trigger box."""
    watched = [("U_FakeMagnet", "Current"), ("U_FakeTrigger", "Trigger.Source")]
    last = {key: machine.devices[key[0]].get(key[1]) for key in watched}
    while not stop.wait(0.1):
        for device, variable in watched:
            value = machine.devices[device].get(variable)
            if value != last[(device, variable)]:
                last[(device, variable)] = value
                print(f"  [machine] {device} {variable} -> {value}", flush=True)


def _set_pressure(machine: Machine, argument: str) -> None:
    gauge = machine.devices["U_FakeGauge"]
    try:
        gauge.set("Pressure", float(argument))
    except ValueError:
        print("  usage: p <number>")
        return
    print(f"  Pressure = {gauge.get('Pressure')}")


def _set_ticking(machine: Machine, ticking: bool) -> None:
    machine.ticking = ticking
    print("  acq_timestamp ticking" if ticking else "  acq_timestamp frozen")


def _gauge_down(machine: Machine) -> None:
    machine.run_on_loop(machine.gauge_down())
    print("  U_FakeGauge down (the gateway notices the dropped connection)")


def _gauge_up(machine: Machine) -> None:
    machine.run_on_loop(machine.gauge_up())
    print("  U_FakeGauge up (the gateway reconnects with its backoff)")


def _show(machine: Machine) -> None:
    for name, device in machine.devices.items():
        print(f"  {name}: {device.variables}")


COMMANDS = {
    "p": _set_pressure,
    "freeze": lambda machine, _: _set_ticking(machine, False),
    "thaw": lambda machine, _: _set_ticking(machine, True),
    "down": lambda machine, _: _gauge_down(machine),
    "up": lambda machine, _: _gauge_up(machine),
    "show": lambda machine, _: _show(machine),
}


def prompt(machine: Machine, stop: threading.Event) -> None:
    """Read commands from the terminal until 'quit' (or end of input)."""
    print(HELP, flush=True)
    while not stop.is_set():
        try:
            line = input("machine> ").strip()
        except EOFError:
            break
        command, _, argument = line.partition(" ")
        if command == "quit":
            break
        if command in COMMANDS:
            COMMANDS[command](machine, argument)
        elif command:
            print(HELP)
    stop.set()


async def main() -> None:
    """Start the machine, serve the gateway until 'quit'."""
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    machine = Machine()
    await machine.start()
    assert machine.gateway is not None
    print("\nServing PVs on 127.0.0.1:")
    for pv in sorted(machine.gateway.pvdb):
        if pv.lower().startswith(EXPERIMENT.lower() + ":u_"):
            print(f"  {pv}")
    print()
    stop = threading.Event()
    for target in (tick_stamp, report_changes, prompt):
        threading.Thread(target=target, args=(machine, stop), daemon=True).start()
    serving = asyncio.create_task(machine.gateway.serve())
    try:
        while not stop.is_set():
            await asyncio.sleep(0.2)
    finally:
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving
        await machine.gateway.close()
        for server in machine.servers.values():
            await server.stop()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
