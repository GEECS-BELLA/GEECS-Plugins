"""Interlock sandbox, the worker half: interlocks, a scan and the status stream.

Offline, on this computer only — the counterpart of
``GeecsCAGateway/examples/interlock_sandbox_machine.py``, which must be
running first (it serves the fake devices as PVs).  This is the worker's
interlock stack minus the GEECS database: the namespace comes from a
hand-written roster of the fake devices, everything else is the real code
over real CA — ``resolve_interlocks``, ``install_interlocks``, the
suspenders on a ``make_run_engine`` RunEngine, a ``ShotControl`` driving
the fake trigger box, and the LabVIEW status stream.

Run it with GeecsBluesky's environment (the ``ca`` extra installed)::

    cd GeecsBluesky && poetry run python examples/interlock_sandbox_worker.py

It writes an example ``interlocks.yaml`` and ``action_library/actions.yaml``
into ``--configs`` (a temporary folder by default; existing files are never
overwritten — edit them and run again), prints what to try in the machine
terminal, then counts the fake magnet's current with the box in SCAN until
Ctrl-C.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

# The sandbox's gateway is on this computer: point CA there before
# geecs_bluesky applies config.ini's address list (an exported variable
# wins) and before libca's context exists.
os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
from bluesky.utils import RunEngineInterrupted

from geecs_bluesky.config_resolver import ConfigsRepoResolver
from geecs_bluesky.devices.shot_control import (
    ShotControl,
    ShotControlWrites,
)
from geecs_bluesky.interlocks import install_interlocks
from geecs_bluesky.interlocks.status_server import InterlockStatusServer
from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace
from geecs_bluesky.run_engine import make_run_engine

EXPERIMENT = "Sandbox"

INTERLOCKS_YAML = """\
schema_version: 1
interlocks:
  - name: chamber vacuum
    description: the sandbox gauge above 1e-5 (try 'p 1e-3', then 'p 1e-6')
    condition:
      kind: threshold
      device: U_FakeGauge
      variable: Pressure
      operator: ">"
      limit: 1.0e-5
      hysteresis: 2.0e-6
    on_trip: magnet_to_zero
    before_resume: magnet_to_one
    settle_s: 2
  - name: gauge alive
    description: the gauge's stamp stops for 3 s (try 'freeze', then 'thaw')
    condition:
      kind: stale
      device: U_FakeGauge
      max_age_s: 3
"""

ACTIONS_YAML = """\
schema_version: 1
plans:
  magnet_to_zero:
    description: what a trip does in the sandbox
    steps:
      - {do: set, device: U_FakeMagnet, variable: Current, value: 0}
  magnet_to_one:
    description: what happens before the scan resumes
    steps:
      - {do: set, device: U_FakeMagnet, variable: Current, value: 1}
"""


TRY_THIS = """
In the machine terminal:
  p 1e-3 / p 1e-6   trip / clear the vacuum interlock: the rows stop, the box
                    goes 'off', on_trip sets the magnet to 0; after settle_s
                    before_resume sets it back to 1 and the rows continue
  freeze / thaw     stop the gauge's stamp: the stale interlock trips
  down / up         take the gauge away: every interlock on it trips
Point the LabVIEW VI (or InterlockScripts/interlock_client.py) at
127.0.0.1:{port} to watch SAFE / WARNING!.  Ctrl-C stops the scan.
"""


def _row(name: str, *, settable: bool = False) -> dict:
    """One variable as the GEECS DB describes it (the namespace's input)."""
    return {
        "name": name,
        "settable": settable,
        "variabletype": None,
        "choices": "numeric",
        "tolerance": None,
        "units": "",
        "min": None,
        "max": None,
    }


# The sandbox's fake devices as the DB would list them.  The gauge's
# acq_timestamp makes it a GeecsDetector, as a real device with a stamp is;
# nothing triggers it here, the interlocks only subscribe to its signals.
ROSTER = DeviceRoster(
    experiment=EXPERIMENT,
    variables={
        "U_FakeGauge": [_row("Pressure"), _row("acq_timestamp")],
        "U_FakeMagnet": [_row("Current", settable=True)],
    },
    types={"U_FakeGauge": "Gauge", "U_FakeMagnet": "Magnet PS"},
    subscribed={"U_FakeGauge": ["Pressure"], "U_FakeMagnet": ["Current"]},
)

TRIGGER_BOX = ShotControlWrites(
    name="sandbox",
    states={
        "OFF": [("U_FakeTrigger", "Trigger.Source", "off")],
        "STANDBY": [("U_FakeTrigger", "Trigger.Source", "edges")],
        "SCAN": [("U_FakeTrigger", "Trigger.Source", "scan")],
    },
)


def write_example_configs(root: Path) -> Path:
    """Write the example documents under *root*/Sandbox unless they exist; return that folder."""
    folder = root / EXPERIMENT
    (folder / ConfigsRepoResolver.ACTION_FOLDER).mkdir(parents=True, exist_ok=True)
    for path, text in (
        (folder / ConfigsRepoResolver.INTERLOCKS_FILE, INTERLOCKS_YAML),
        (folder / ConfigsRepoResolver.ACTION_FOLDER / "actions.yaml", ACTIONS_YAML),
    ):
        if not path.exists():
            path.write_text(text)
    return folder


def print_rows(name: str, doc: dict) -> None:
    """Print each row as it is recorded (a pause shows as the rows stopping)."""
    if name == "event":
        values = ", ".join(
            f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in doc["data"].items()
        )
        print(f"  row {doc['seq_num']:>4}: {values}", flush=True)


def sandbox_scan(magnet, box: ShotControl, shots: int, period: float):
    """Box to SCAN, a slow count of the magnet, box back to STANDBY on every exit."""

    def body():
        yield from bps.mv(box, "SCAN")
        yield from bp.count([magnet], num=shots, delay=period)

    return (yield from bpp.finalize_wrapper(body(), bps.mv(box, "STANDBY")))


def main() -> None:
    """Install the interlocks, start the status stream and run the scan."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--configs",
        type=Path,
        default=None,
        help="folder holding Sandbox/interlocks.yaml (default: a new temporary folder)",
    )
    parser.add_argument("--status-host", default="127.0.0.1")
    parser.add_argument("--status-port", type=int, default=5001)
    parser.add_argument("--shots", type=int, default=600)
    parser.add_argument("--period", type=float, default=1.0, help="seconds per row")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("geecs_bluesky.interlocks").setLevel(logging.INFO)

    root = args.configs or Path(tempfile.mkdtemp(prefix="interlock-sandbox-"))
    folder = write_example_configs(root)
    print(f"configs: {folder}  (edit interlocks.yaml / action_library/actions.yaml)")

    resolver = ConfigsRepoResolver(EXPERIMENT, experiments_root=root)
    namespace = GeecsNamespace(ROSTER)
    RE = make_run_engine(experiment=EXPERIMENT)
    box = ShotControl(TRIGGER_BOX, experiment=EXPERIMENT, name="shot_control")
    asyncio.run_coroutine_threadsafe(box.connect(), RE._loop).result(10)

    registry = install_interlocks(
        RE, resolver.resolve_interlocks(), namespace, resolver, timeout=5.0
    )
    for status in registry.snapshot():
        state = "armed" if status.armed else f"NOT ARMED ({status.diagnostic})"
        print(f"interlock {status.name!r}: {state}")
    if not registry.suspenders:
        raise SystemExit(
            "no interlock armed — is the machine half "
            "(GeecsCAGateway/examples/interlock_sandbox_machine.py) running?"
        )

    server = InterlockStatusServer(
        registry.snapshot, args.status_host, args.status_port
    )
    server.start()
    print(f"status stream: {args.status_host}:{args.status_port}")
    print(TRY_THIS.format(port=args.status_port))

    RE.subscribe(print_rows)
    magnet = namespace["U_FakeMagnet"]
    try:
        RE(sandbox_scan(magnet, box, args.shots, args.period))
    except (KeyboardInterrupt, RunEngineInterrupted):
        if RE.state != "idle":
            RE.abort("stopped from the sandbox")
    finally:
        server.stop()
        registry.remove(RE)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
