"""Phase-1 hardware acceptance: stock plans over the device namespace (#807).

Hardware-marked (skipped in CI; ``-m integration`` does NOT select it — it
arms the machine trigger).  Run by hand on a host with CA reach to the
GEECS gateway and the GEECS DB (the qserver box)::

    GEECS_HW_SCAN_VARIABLE=U_S1H:Current GEECS_HW_SCAN_START=-1 \\
    GEECS_HW_SCAN_END=1 GEECS_HW_SCAN_STEP=0.5 \\
    poetry run pytest tests/test_namespace_hardware.py -m hardware -s

What it proves (``Planning/native_bluesky/01_device_namespace.md``,
"Acceptance"): the namespace builds from the live DB, ``bp.count`` and
``bp.scan`` from ``bluesky.plans`` run over namespace devices with **no
GEECS preamble**, connected lazily by :func:`connect_on_demand`, with the
trigger profile armed/disarmed by the existing :class:`ShotController` (the
phase-2 preprocessor's job, done by hand here).  No scan number is claimed
and Tiled is off: this is a device/plan-layer check, not a data run.

**MOVES HARDWARE** when ``GEECS_HW_SCAN_VARIABLE`` is set: the swept
setpoint is restored to its pre-scan setpoint in a ``finally``.  With the
variable unset only the ``count`` half runs.

Environment
-----------
``GEECS_HW_EXPERIMENT``       experiment (default ``Undulator``)
``GEECS_HW_CAMERA_DEVICE``    GEECS camera device (default ``UC_Amp4_IR_input``)
``GEECS_HW_TRIGGER_PROFILE``  trigger profile (default ``HTU-NoGas``)
``GEECS_HW_SHOTS``            shots for the count (default 3)
``GEECS_HW_SCAN_VARIABLE``    ``Device:Variable`` to sweep (unset → no scan)
``GEECS_HW_SCAN_START/END/STEP``  sweep bounds (mandatory with the variable)
``GEECS_HW_DOCS_OUT``         optional path to dump the documents as JSON
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.ca_mock_helpers import DocCollector

pytestmark = pytest.mark.hardware
pytest.importorskip("aioca")


def _sweep_points(start: float, end: float, step: float) -> list[float]:
    n = int(round(abs(end - start) / abs(step))) + 1
    sign = 1.0 if end >= start else -1.0
    return [round(start + sign * i * abs(step), 6) for i in range(n)]


@pytest.mark.hardware
def test_stock_plans_over_the_namespace_on_hardware() -> None:
    """Namespace from the live DB; stock count + scan over it; trigger by hand."""
    import bluesky.plan_stubs as bps
    import bluesky.plans as bp
    import bluesky.preprocessors as bpp

    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.devices.ca.oneshot import try_caget_once
    from geecs_bluesky.namespace import GeecsNamespace
    from geecs_bluesky.plans.scan_request_plan import _await_in_plan
    from geecs_bluesky.preprocessors import install_connect_on_demand
    from geecs_bluesky.scan_request_runner import trigger_writes_from_profile
    from geecs_bluesky.session import GeecsSession
    from geecs_bluesky.shot_controller import ShotController
    from geecs_core.pv_naming import pv_name, setpoint_pv

    experiment = os.environ.get("GEECS_HW_EXPERIMENT", "Undulator")
    camera_name = os.environ.get("GEECS_HW_CAMERA_DEVICE", "UC_Amp4_IR_input")
    profile_name = os.environ.get("GEECS_HW_TRIGGER_PROFILE", "HTU-NoGas")
    shots = int(os.environ.get("GEECS_HW_SHOTS", "3"))
    sweep_target = os.environ.get("GEECS_HW_SCAN_VARIABLE")
    docs_out = os.environ.get("GEECS_HW_DOCS_OUT")

    resolver = ConfigsRepoResolver(experiment)
    namespace = GeecsNamespace.from_experiment(experiment)
    print(f"\nnamespace: {len(namespace)} devices for {experiment}")
    camera = namespace[camera_name]
    assert hasattr(camera, "trigger"), f"{camera_name} not classified triggerable"
    print(
        f"camera {type(camera).__name__} {camera.name}; children: {len(list(camera.children()))}"
    )

    session = GeecsSession(experiment, tiled=False)
    RE = session.RE
    install_connect_on_demand(RE)

    profile = resolver.resolve_trigger_profile(profile_name)
    controller = ShotController.from_writes(
        trigger_writes_from_profile(profile), experiment=experiment
    )

    movable = initial = None
    points: list[float] = []
    if sweep_target:
        device_name, _, variable = sweep_target.partition(":")
        movable = namespace.resolve(sweep_target)
        points = _sweep_points(
            float(os.environ["GEECS_HW_SCAN_START"]),
            float(os.environ["GEECS_HW_SCAN_END"]),
            float(os.environ["GEECS_HW_SCAN_STEP"]),
        )
        initial = float(
            try_caget_once(
                setpoint_pv(pv_name(experiment, device_name, variable)), timeout=5.0
            )
        )
        print(f"sweep {sweep_target} over {points} (pre-scan setpoint {initial})")

    docs = DocCollector()

    def acceptance():
        # The phase-2 preprocessor's future job, by hand: reachability of the
        # shot-control setters, then SCAN (free-running external trigger).
        yield from _await_in_plan(controller.connect_setters)
        yield from controller.arm()
        yield from bp.count([camera], num=shots, md={"purpose": "807-phase1-count"})
        if movable is not None:
            yield from bp.list_scan(
                [camera], movable, points, md={"purpose": "807-phase1-scan"}
            )

    def cleanup():
        yield from controller.disarm()
        if movable is not None and initial is not None:
            yield from bps.mv(movable, initial)
            print(f"restored {sweep_target} to pre-scan setpoint {initial}")

    RE(bpp.finalize_wrapper(acceptance(), cleanup()), docs)

    if docs_out:
        Path(docs_out).write_text(json.dumps(docs.docs, default=str, indent=1))
        print(f"documents written to {docs_out}")

    starts = docs.docs["start"]
    stops = docs.docs["stop"]
    assert [s["plan_name"] for s in starts] == (
        ["count", "list_scan"] if movable is not None else ["count"]
    )
    assert all(s["exit_status"] == "success" for s in stops), stops

    events = docs.primary_events()
    count_uid = starts[0]["uid"]
    count_events = [e for e in events if _run_of(e, docs) == count_uid]
    assert len(count_events) == shots
    stamps = [e["data"][f"{camera.name}-acq_timestamp"] for e in count_events]
    assert len(set(stamps)) == shots, f"shots did not advance: {stamps}"
    print(f"count: {shots} shots, acq_timestamps {stamps}")
    print(f"count columns: {sorted(count_events[0]['data'])}")

    if movable is not None:
        scan_uid = starts[1]["uid"]
        scan_events = [e for e in events if _run_of(e, docs) == scan_uid]
        assert len(scan_events) == len(points)
        readback_key = movable.position.name if hasattr(movable, "position") else None
        assert readback_key is not None, "sweep target is not a CaMotor"
        readbacks = [e["data"][readback_key] for e in scan_events]
        tol = getattr(movable, "_tolerance", 0.05)
        for want, got in zip(points, readbacks):
            assert abs(got - want) <= tol + 1e-9, f"{readback_key}: {got} vs {want}"
        print(f"scan: {len(points)} points, readbacks {readbacks}")


def _run_of(event: dict, docs: DocCollector) -> str:
    for d in docs.docs["descriptor"]:
        if d["uid"] == event["descriptor"]:
            return d["run_start"]
    raise AssertionError("event without descriptor")
