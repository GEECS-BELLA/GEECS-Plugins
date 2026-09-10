"""Phase-0 hardware acceptance: one camera as a StandardDetector (#807).

Hardware-marked (skipped in CI; ``-m integration`` does NOT select it — it
arms the machine trigger and fires shots).  Run by hand on a host with CA
reach to the GEECS gateway and the GEECS DB (the qserver box)::

    GEECS_HW_SCAN_VARIABLE=U_S1H:Current GEECS_HW_SCAN_START=-1 \\
    GEECS_HW_SCAN_END=1 GEECS_HW_SCAN_STEP=0.5 GEECS_HW_SAVE=1 \\
    poetry run python -u -m pytest tests/test_phase0_hardware.py -m hardware -s

What it proves (``Planning/native_bluesky/03_clean_room_rebuild.md`` §8,
phase 0): a :class:`GeecsDetector` built for a real camera, a
:class:`ShotControl` built from the trigger profile, and stock
``bp.count`` / ``bp.list_scan`` with :func:`geecs_per_shot` /
:func:`geecs_per_step` run a strict GEECS scan — the detector's own
lifecycle turns native saving on and off, every row carries the shot's
stamp, and the files land in the claimed scan folder.  Also records the
per-shot timing the §7 budget depends on.

**MOVES HARDWARE** when ``GEECS_HW_SCAN_VARIABLE`` is set: the swept
setpoint is restored to its pre-scan setpoint in a ``finally``.  With
``GEECS_HW_SAVE=1`` a scan number is **claimed** (the scanner-side action)
and the camera writes its native files into that folder.

Environment
-----------
``GEECS_HW_EXPERIMENT``       experiment (default ``Undulator``)
``GEECS_HW_CAMERA_DEVICE``    GEECS camera device (default ``UC_Amp4_IR_input``)
``GEECS_HW_TRIGGER_PROFILE``  trigger profile (default ``HTU-NoGas``)
``GEECS_HW_SHOTS``            shots for the count (default 3)
``GEECS_HW_SCAN_VARIABLE``    ``Device:Variable`` to sweep (unset → no scan)
``GEECS_HW_SCAN_START/END/STEP``  sweep bounds (mandatory with the variable)
``GEECS_HW_SAVE``             ``1`` → claim a scan number and save natively
``GEECS_HW_DOCS_OUT``         optional path to dump the documents as JSON
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from tests.ca_mock_helpers import DocCollector

pytestmark = pytest.mark.hardware
pytest.importorskip("aioca")


def _sweep_points(start: float, end: float, step: float) -> list[float]:
    n = int(round(abs(end - start) / abs(step))) + 1
    sign = 1.0 if end >= start else -1.0
    return [round(start + sign * i * abs(step), 6) for i in range(n)]


def _wait_for_files(
    directory: Path, expected: int, timeout: float = 10.0
) -> list[Path]:
    """The end-of-run file check: every expected native file exists and stopped growing."""
    deadline = time.monotonic() + timeout
    while True:
        files = sorted(directory.glob("*.png"))
        sizes = [f.stat().st_size for f in files]
        if len(files) >= expected and all(sizes):
            time.sleep(0.5)
            if [f.stat().st_size for f in files] == sizes:
                return files
        if time.monotonic() > deadline:
            raise AssertionError(
                f"{directory}: {len(files)} files after {timeout:.0f} s, expected {expected}"
            )
        time.sleep(0.5)


@pytest.mark.hardware
def test_one_camera_as_a_standard_detector_on_hardware() -> None:
    """GeecsDetector + ShotControl under stock count/list_scan; strict shots; native files."""
    import bluesky.plan_stubs as bps
    import bluesky.plans as bp
    import bluesky.preprocessors as bpp
    from geecs_core.pv_naming import pv_name, setpoint_pv
    from ophyd_async.core import StaticFilenameProvider, StaticPathProvider

    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.devices.ca.oneshot import try_caget_once
    from geecs_bluesky.devices.detector import GeecsDetector
    from geecs_bluesky.devices.shot_control import ShotControl
    from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace, python_type
    from geecs_bluesky.plans.run_wrapper import claim_scan_number
    from geecs_bluesky.plans.strict import geecs_per_shot, geecs_per_step
    from geecs_bluesky.preprocessors import install_connect_on_demand
    from geecs_bluesky.session import GeecsSession

    experiment = os.environ.get("GEECS_HW_EXPERIMENT", "Undulator")
    camera_name = os.environ.get("GEECS_HW_CAMERA_DEVICE", "UC_Amp4_IR_input")
    profile_name = os.environ.get("GEECS_HW_TRIGGER_PROFILE", "HTU-NoGas")
    shots = int(os.environ.get("GEECS_HW_SHOTS", "3"))
    sweep_target = os.environ.get("GEECS_HW_SCAN_VARIABLE")
    save = os.environ.get("GEECS_HW_SAVE", "0") == "1"
    docs_out = os.environ.get("GEECS_HW_DOCS_OUT")

    # --- the camera, from the DB roster (the namespace's rule, for one device)
    roster = DeviceRoster.from_geecs_db(experiment)
    served = roster.served_for(camera_name)
    rows = {str(r["name"]): r for r in roster.variables[camera_name]}
    readables = [
        v
        for v in roster.subscribed.get(camera_name, ())
        if v.lower() in served and v in rows and not rows[v].get("settable")
    ]
    datatypes = {v: python_type(rows[v]) for v in readables}
    print(f"\n{camera_name}: {len(readables)} scalar readables {readables}")

    path_provider = None
    scan_number = folder = None
    if save:
        scan_number, folder = claim_scan_number(experiment)
        assert scan_number is not None and folder, "could not claim a scan number"
        print(f"claimed scan {scan_number} → {folder}")
        path_provider = StaticPathProvider(
            StaticFilenameProvider(camera_name), Path(folder) / camera_name
        )
    camera = GeecsDetector(
        camera_name,
        readables,
        experiment=experiment,
        name=camera_name.lower(),
        datatypes=datatypes,
        path_provider=path_provider,
        shot_timeout=3.0,
    )

    # --- the trigger box, from the profile
    resolver = ConfigsRepoResolver(experiment)
    shot_control = ShotControl.from_profile(
        resolver.resolve_trigger_profile(profile_name),
        experiment=experiment,
        name="shot_control",
    )
    assert shot_control.defines("ARMED") and shot_control.defines("SINGLESHOT")

    session = GeecsSession(experiment, tiled=False)
    RE = session.RE
    install_connect_on_demand(RE)
    for device in (camera, shot_control):
        asyncio.run_coroutine_threadsafe(device.connect(timeout=20.0), RE._loop).result(
            30
        )
    print(f"camera connected; last stamp {camera.last_acq_timestamp}")

    # --- the scan variable, from the namespace (#808)
    movable = initial = None
    points: list[float] = []
    if sweep_target:
        namespace = GeecsNamespace.from_experiment(experiment)
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
    timeline: list[tuple[str, float]] = []
    RE.subscribe(lambda name, doc: timeline.append((name, time.monotonic())), "event")

    def acceptance():
        yield from bps.mv(shot_control, "ARMED")
        yield from bp.count(
            [camera],
            num=shots,
            per_shot=geecs_per_shot(shot_control),
            md={"purpose": "807-phase0-count", "scan_number": scan_number},
        )
        if movable is not None:
            yield from bp.list_scan(
                [camera],
                movable,
                points,
                per_step=geecs_per_step(shot_control),
                md={"purpose": "807-phase0-scan", "scan_number": scan_number},
            )

    def cleanup():
        yield from bps.mv(shot_control, "STANDBY")
        if movable is not None and initial is not None:
            yield from bps.mv(movable, initial)
            print(f"restored {sweep_target} to pre-scan setpoint {initial}")

    t_start = time.monotonic()
    RE(bpp.finalize_wrapper(acceptance(), cleanup()), docs)
    if docs_out:
        Path(docs_out).write_text(json.dumps(docs.docs, default=str, indent=1))
        print(f"documents written to {docs_out}")

    starts, stops = docs.docs["start"], docs.docs["stop"]
    assert [s["plan_name"] for s in starts] == (
        ["count", "list_scan"] if movable is not None else ["count"]
    )
    assert all(s["exit_status"] == "success" for s in stops), stops
    events = docs.primary_events()
    expected_events = shots + len(points)
    assert len(events) == expected_events, (
        f"{len(events)} events, expected {expected_events}"
    )
    stamps = [e["data"][f"{camera.name}-acq_timestamp"] for e in events]
    assert len(set(stamps)) == expected_events, (
        f"stamps did not advance per shot: {stamps}"
    )
    gaps = [round(b - a, 3) for a, b in zip(stamps, stamps[1:])]
    cadence = [round(b - a, 3) for (_, a), (_, b) in zip(timeline, timeline[1:])]
    print(f"stamps: {stamps}\nstamp gaps (s): {gaps}\nevent cadence (s): {cadence}")
    print(f"columns: {sorted(events[0]['data'])}")
    primary = next(d for d in docs.docs["descriptor"] if d["name"] == "primary")
    print(
        f"configuration: { {k: v['data'] for k, v in primary['configuration'].items()} }"
    )
    print(f"wall: {time.monotonic() - t_start:.1f} s for {expected_events} shots")
    assert shot_control.standing_state == "STANDBY"

    if movable is not None:
        scan_events = [e for e in events if _run_of(e, docs) == starts[1]["uid"]]
        readback_key = movable.position.name
        readbacks = [e["data"][readback_key] for e in scan_events]
        tol = getattr(movable, "_tolerance", 0.05)
        for want, got in zip(points, readbacks):
            assert abs(got - want) <= tol + 1e-9, f"{readback_key}: {got} vs {want}"
        print(f"scan: {len(points)} points, readbacks {readbacks}")

    if save:
        directory = Path(folder) / camera_name
        assert [e["data"][f"{camera.name}-nonscalar_save_path"] for e in events] == [
            str(directory)
        ] * len(events)
        files = _wait_for_files(directory, expected_events)
        print(f"native files: {len(files)} in {directory}: {[f.name for f in files]}")
        save_state = asyncio.run_coroutine_threadsafe(
            camera.save.get_value(), RE._loop
        ).result(5)
        assert save_state == "off", f"save left {save_state!r}"


def _run_of(event: dict, docs: DocCollector) -> str:
    for d in docs.docs["descriptor"]:
        if d["uid"] == event["descriptor"]:
            return d["run_start"]
    raise AssertionError("event without descriptor")
