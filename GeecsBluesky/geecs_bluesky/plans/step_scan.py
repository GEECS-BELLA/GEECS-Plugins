"""geecs_step_scan — a Bluesky step-scan plan for GEECS hardware.

Moves a motor through a sequence of positions, collects ``shots_per_step``
shots from one or more detectors at each step, and emits Bluesky event
documents for downstream consumers (live callbacks, Databroker, etc.).

How it fits into Bluesky
------------------------
* :func:`bluesky.plan_stubs.mv` moves the motor by calling ``motor.set(pos)``
  and waiting for the returned :class:`~ophyd_async.core.AsyncStatus`.
* :func:`bluesky.plan_stubs.trigger_and_read` calls ``trigger()`` on every
  :class:`~bluesky.protocols.Triggerable` detector, waits for all triggers
  to complete, then reads all devices in the list (including the motor, so
  motor position is recorded alongside detector data in each event document).

Shot synchronisation
--------------------
Detectors that inherit from
:class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable` complete their
``trigger()`` call by waiting for the hardware ``acq_timestamp`` variable to
advance — the real shot timestamp from the DG645 delay generator.  This is
robust to device restarts (shot numbers drift; timestamps don't).

Example::

    import numpy as np
    from bluesky import RunEngine
    from geecs_bluesky.devices.motor import GeecsMotor
    from geecs_bluesky.devices.camera import GeecsCameraBase
    from geecs_bluesky.plans.step_scan import geecs_step_scan

    RE = RunEngine()

    motor = GeecsMotor("U_ESP_JetXYZ", "Position.Axis 1",
                       "192.168.8.198", 65158,
                       name="jet_x", units="mm")
    cam = GeecsCameraBase("U_ProbeCam", "192.168.8.50", 64000,
                          name="probe_cam")

    await motor.connect()
    await cam.connect()

    RE(geecs_step_scan(
        motor=motor,
        positions=np.linspace(0, 5, 6),
        detectors=[cam],
        shots_per_step=5,
        md={"sample": "He jet", "operator": "jdoe"},
    ))
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

from geecs_bluesky.devices.scan_context import ScanContext
from geecs_bluesky.plans.single_shot import geecs_single_shot


def geecs_step_scan(
    motor: Any | None,
    positions: Iterable[float | None],
    detectors: list[Any],
    shots_per_step: int = 5,
    arm_trigger: Callable | None = None,
    disarm_trigger: Callable | None = None,
    fire_shot: Callable | None = None,
    md: dict[str, Any] | None = None,
):
    """Step-scan plan: move *motor* through *positions*, collect *shots_per_step* shots.

    Parameters
    ----------
    motor:
        Any :class:`~bluesky.protocols.Movable` device — a stage axis
        (:class:`~geecs_bluesky.devices.motor.GeecsMotor`), power supply,
        pressure controller, etc. (anything with ``set() → status``, e.g.
        built on :class:`~geecs_bluesky.devices.settable.GeecsSettable`).
        The name follows the bluesky ``scan(detectors, motor, ...)``
        convention.  ``None`` means no scan variable is moved — statistics
        collection (the former "NOSCAN" mode); pass ``positions=[None]`` for a
        single no-move bin.
    positions:
        Iterable of motor positions to visit.  A ``None`` entry is a bin with
        no motor move (used with ``motor=None``).
    detectors:
        List of :class:`~bluesky.protocols.Readable` / Triggerable devices
        to read at each shot (e.g. camera devices).  The motor is included
        automatically so its position is recorded in every event document.
    shots_per_step:
        Number of shots to collect at each motor position.  Default: ``5``.
    arm_trigger:
        Optional callable returning a plan generator that arms the shot
        controller (e.g. sets DG645 outputs to SCAN state).  Called after
        each motor move, before collecting shots.
    disarm_trigger:
        Optional callable returning a plan generator that disarms the shot
        controller (e.g. sets DG645 outputs to STANDBY state).  Called after
        collecting shots at each step, before the next motor move.
    fire_shot:
        Optional plan-stub callable that fires exactly one trigger (e.g.
        drives the DG645 ``SINGLESHOT`` state).  When provided, the plan
        owns every shot — each row is collected via
        :func:`~geecs_bluesky.plans.single_shot.geecs_single_shot`
        (arm waiters → fire → await → read) instead of waiting on a
        free-running trigger.  This is the full strict-shot-control
        contract; without it, detectors wait for the next free-running
        shot (internal-trigger test mode).
    md:
        Extra metadata merged into the RunEngine ``start`` document.

    Yields
    ------
    Bluesky messages — pass this generator to a :class:`~bluesky.RunEngine`.

    Notes
    -----
    * The motor is appended to the read list for every shot so that each
      event document records the actual motor readback alongside detector
      data.
    * Non-Triggerable devices in *detectors* are read without calling
      ``trigger()`` (standard :func:`bluesky.plan_stubs.trigger_and_read`
      behaviour).
    * ``arm_trigger`` / ``disarm_trigger`` bracket the per-step acquisition
      window so the shot controller is only active while shots are being
      collected, not during motor moves.
    """
    _positions = list(positions)
    scan_context = ScanContext()
    _read_devices = (
        list(detectors) + ([motor] if motor is not None else []) + [scan_context]
    )

    _md: dict[str, Any] = {
        "plan_name": "geecs_step_scan",
        "acquisition_mode": "strict_shot_control",
        "geecs_event_schema": 1,
        "motor": getattr(motor, "name", None) if motor is not None else None,
        "detectors": [getattr(d, "name", str(d)) for d in detectors],
        "positions": _positions,
        "shots_per_step": shots_per_step,
        "num_points": len(_positions),
        **(md or {}),
    }

    @bpp.run_decorator(md=_md)
    def _inner():
        scan_event_index = 0
        for bin_number, pos in enumerate(_positions, start=1):
            if motor is not None and pos is not None:
                yield from bps.mv(motor, pos)
            if arm_trigger is not None:
                yield from arm_trigger()
            for shot_index_in_bin in range(1, shots_per_step + 1):
                scan_event_index += 1
                scan_context.set_context(
                    bin_number=bin_number,
                    shot_index_in_bin=shot_index_in_bin,
                    scan_event_index=scan_event_index,
                )
                if fire_shot is not None:
                    yield from geecs_single_shot(_read_devices, fire_shot)
                else:
                    yield from bps.trigger_and_read(_read_devices)
            if disarm_trigger is not None:
                yield from disarm_trigger()

    yield from _inner()
