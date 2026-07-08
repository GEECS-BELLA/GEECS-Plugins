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
    from geecs_bluesky.devices.generic_detector import GeecsGenericDetector
    from geecs_bluesky.devices.motor import GeecsMotor
    from geecs_bluesky.plans.step_scan import geecs_step_scan

    RE = RunEngine()

    motor = GeecsMotor("U_ESP_JetXYZ", "Position.Axis 1",
                       "192.168.8.198", 65158,
                       name="jet_x", units="mm")
    det = GeecsGenericDetector("U_ProbeCam", ["MeanCounts"],
                               "192.168.8.50", 64000,
                               name="probe_cam")

    await motor.connect()
    await det.connect()

    RE(geecs_step_scan(
        motor=motor,
        positions=np.linspace(0, 5, 6),
        detectors=[det],
        shots_per_step=5,
        md={"sample": "He jet", "operator": "jdoe"},
    ))
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

from geecs_bluesky.devices.scan_context import ScanContext
from geecs_bluesky.plans.single_shot import geecs_single_shot


def normalize_motors(motor: Any | Sequence[Any] | None) -> list[Any]:
    """Return the motor argument as a list (``None`` → ``[]``).

    Step plans accept a single Movable (the classic 1-D scan), a sequence of
    Movables (a multi-axis grid — one per axis, outermost first), or ``None``
    (statistics collection).

    Parameters
    ----------
    motor : Movable, sequence of Movable, or None
        The scan axis (or axes).

    Returns
    -------
    list
        The motors, outermost axis first.
    """
    if motor is None:
        return []
    if isinstance(motor, (list, tuple)):
        return list(motor)
    return [motor]


def motor_md(motors: list[Any]) -> Any:
    """Start-document ``motor`` metadata: name, list of names, or ``None``."""
    if not motors:
        return None
    names = [getattr(m, "name", str(m)) for m in motors]
    return names[0] if len(names) == 1 else names


def move_changed_axes(motors: list[Any], position: Any, previous: tuple | None):
    """Plan stub: move only the axes whose target changed; return the targets.

    A grid point is a tuple aligned with *motors* (a bare float is the 1-D
    case).  With the outer product ordering (first axis outermost, last
    innermost) the innermost axis changes at every grid point while outer
    axes change rarely — moving only the changed axes avoids re-commanding
    stationary hardware.  Changed axes are moved **concurrently** via
    ``bps.mv`` (waits for all).

    Parameters
    ----------
    motors : list
        The scan axes, outermost first.
    position : float or tuple
        The grid point to move to (tuple aligned with *motors*).
    previous : tuple or None
        The previous grid point (``None`` on the first step — every axis is
        moved).

    Yields
    ------
    Msg
        Bluesky messages for the move.

    Returns
    -------
    tuple
        The targets just commanded (pass back as *previous* next step).

    Raises
    ------
    ValueError
        If the grid point's length does not match the number of motors.
    """
    targets = tuple(position) if isinstance(position, (list, tuple)) else (position,)
    if len(targets) != len(motors):
        raise ValueError(
            f"grid point {targets!r} has {len(targets)} value(s) for "
            f"{len(motors)} motor(s) — positions must align with the axes"
        )
    args: list[Any] = []
    for m, target, prev in zip(motors, targets, previous or (None,) * len(motors)):
        if prev is None or target != prev:
            args.extend([m, target])
    if args:
        yield from bps.mv(*args)
    return targets


def geecs_step_scan(
    motor: Any | Sequence[Any] | None,
    positions: Iterable[Any],
    detectors: list[Any],
    shots_per_step: int = 5,
    arm_trigger: Callable | None = None,
    disarm_trigger: Callable | None = None,
    fire_shot: Callable | None = None,
    setup_trigger: Callable | None = None,
    per_step: Callable | None = None,
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
        convention.  A **sequence** of Movables is a multi-axis grid scan
        (one motor per axis, outermost first; each position is then a tuple
        aligned with the motors).  ``None`` means no scan variable is moved —
        statistics collection (the former "NOSCAN" mode); pass
        ``positions=[None]`` for a single no-move bin.
    positions:
        Iterable of motor positions to visit — floats for a single motor,
        tuples (one value per motor, outermost axis first) for a grid.  A
        ``None`` entry is a bin with no motor move (used with
        ``motor=None``).  At each grid point only the axes whose target
        changed are re-moved (the innermost axis varies fastest under the
        outer-product ordering).
    detectors:
        List of :class:`~bluesky.protocols.Readable` / Triggerable devices
        to read at each shot.  The motor is included
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
        free-running trigger.  This is the strict-shot-control contract.
    setup_trigger:
        Optional plan-stub callable run *once* at the start of the run (after
        ``open_run``, before the first step).  Used by plan-owned single-shot
        strict mode to arm the controller into single-shot mode and confirm
        the free-run has stopped (``ARMED`` + quiescence check) — a one-time
        action, distinct from per-step ``arm_trigger``.  Teardown is the
        caller's outer finalize (e.g. disarm to STANDBY).
    per_step:
        Optional plan-stub callable run at **every** step boundary — after
        the move to that step's position completes, before that step's
        shots.  This is where a ScanRequest's ``actions.per_step`` plans
        land (vision doc §4.5: per-step actions are a hook plus a named
        plan, never a new plan type).  In strict mode every shot is
        plan-owned, so the machine is quiescent while per-step actions run.
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
    _motors = normalize_motors(motor)
    scan_context = ScanContext()
    _read_devices = list(detectors) + _motors + [scan_context]

    _md: dict[str, Any] = {
        "plan_name": "geecs_step_scan",
        "acquisition_mode": "strict_shot_control",
        "geecs_event_schema": 1,
        # True when the plan fires each shot (strict single-shot).
        "fires_own_shots": fire_shot is not None,
        "motor": motor_md(_motors),
        "detectors": [getattr(d, "name", str(d)) for d in detectors],
        "positions": _positions,
        "shots_per_step": shots_per_step,
        "num_points": len(_positions),
        **(md or {}),
    }

    @bpp.run_decorator(md=_md)
    def _inner():
        if setup_trigger is not None:
            yield from setup_trigger()
        scan_event_index = 0
        previous: tuple | None = None
        for bin_number, pos in enumerate(_positions, start=1):
            if _motors and pos is not None:
                previous = yield from move_changed_axes(_motors, pos, previous)
            if per_step is not None:
                # After the move, before this step's plan-owned shots — the
                # machine is quiescent here (strict fires each shot itself).
                yield from per_step()
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
