"""geecs_step_scan — a Bluesky step-scan plan for GEECS hardware.

Moves a motor through a sequence of positions, collects ``shots_per_step``
shots from one or more detectors at each step, and emits Bluesky event
documents for downstream consumers (live callbacks, Databroker, etc.).
:class:`~geecs_bluesky.devices.ca.triggerable.CaTriggerable` detectors
complete ``trigger()`` when their hardware ``acq_timestamp`` advances;
shot-ID mechanics are covered in ``GeecsBluesky/CLAUDE.md`` (Device Layer).

Example (devices built and connected through a
:class:`~geecs_bluesky.session.GeecsSession`)::

    import numpy as np
    from geecs_bluesky.session import GeecsSession
    from geecs_bluesky.plans.step_scan import geecs_step_scan

    session = GeecsSession(experiment="Undulator")
    motor = session.motor("U_ESP_JetXYZ", "Position.Axis 1", name="jet_x")
    det = session.detector("U_ProbeCam", ["MeanCounts"], name="probe_cam")

    session.RE(geecs_step_scan(
        motor=motor,
        positions=np.linspace(0, 5, 6),
        detectors=[det],
        shots_per_step=5,
        md={"sample": "He jet"},
    ))
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Literal, Sequence

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky.utils import FailedStatus

from geecs_bluesky.devices.scan_context import ScanContext
from geecs_bluesky.plans.pause_semantics import FAILED_MOVE_LOG_PREFIX
from geecs_bluesky.plans.single_shot import geecs_single_shot

logger = logging.getLogger(__name__)


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


def move_with_failed_move_pause(
    motors: list[Any], position: Any, previous: tuple | None
):
    """Plan stub: :func:`move_changed_axes`, pausing the RE on a failed move.

    The decision-4 move site (queueserver migration, issue #641; design
    rationale and the log-line contract:
    :mod:`~geecs_bluesky.plans.pause_semantics`).  On a move status failure
    the reason is recorded — the
    :data:`~geecs_bluesky.plans.pause_semantics.FAILED_MOVE_LOG_PREFIX`
    ERROR line, which the root-logger scan.log handler puts in the scan's
    record — and the plan issues a **hard** ``bps.pause()``.  Resume replays
    the failed ``set``/``wait`` from the pre-move checkpoint (``pause`` is
    uncacheable, so the pause itself never replays): the retry, at the same
    absolute target.  A retry that fails again is thrown back in at the
    pause yield and pauses again; stop ends the run gracefully through the
    plan's finalize chain.

    Parameters mirror :func:`move_changed_axes`.

    Returns
    -------
    tuple
        The targets commanded (the grid point), whether the move landed on
        the first attempt or via a resume replay.
    """
    try:
        return (yield from move_changed_axes(motors, position, previous))
    except FailedStatus as exc:
        targets = (
            tuple(position) if isinstance(position, (list, tuple)) else (position,)
        )
        # Changed axes move concurrently (bps.mv, one status per axis); a
        # FailedStatus carries only the failing status, not the device
        # (bluesky's _add_status_to_group never tags obj onto the status
        # object), so this cannot say which axis raised.  `detail` lists
        # every commanded target for context; `cause`'s own text (e.g.
        # GeecsMotorTimeoutError's device/variable fields) is what actually
        # identifies the failing axis.
        detail = ", ".join(
            f"{getattr(m, 'name', m)} -> {t!r}" for m, t in zip(motors, targets)
        )
        while True:
            cause = exc.__cause__ or exc
            logger.error(
                "%s: commanded %s, one axis failed - see cause for which: "
                "%s; resume retries the move from the last checkpoint, stop "
                "ends the scan gracefully",
                FAILED_MOVE_LOG_PREFIX,
                detail,
                cause,
            )
            try:
                yield from bps.pause()
            except FailedStatus as retry_exc:  # the replayed retry failed too
                exc = retry_exc
                continue
            # Reaching here means resume's replay re-issued the move and its
            # wait completed - the retry landed.
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
    enable_saving: Callable | None = None,
    failed_move_policy: Literal["raise", "pause"] = "raise",
    md: dict[str, Any] | None = None,
):
    """Step-scan plan: move *motor* through *positions*, collect *shots_per_step* shots.

    Parameters
    ----------
    motor:
        Any :class:`~bluesky.protocols.Movable` device — a stage axis
        (:class:`~geecs_bluesky.devices.ca.motor.CaMotor`), power supply,
        pressure controller, etc. (anything with ``set() → status``, e.g.
        built on :class:`~geecs_bluesky.devices.ca.settable.CaSettable`).
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
        Optional plan-stub callable run *once* at the start of the run
        (after ``open_run``, before the first step).  Strict mode uses it to
        arm single-shot and confirm the free-run has stopped (``ARMED`` +
        quiescence check); teardown is the caller's outer finalize.
    per_step:
        Optional plan-stub callable run at **every** step boundary — after
        the move completes, before that step's shots.  A ScanRequest's
        ``actions.per_step`` plans land here; in strict mode every shot is
        plan-owned, so the machine is quiescent while they run.
    enable_saving:
        Optional plan-stub callable that turns native file saving on
        (typically :func:`~geecs_bluesky.plans.run_wrapper.save_enable_plan`).
        Run once, **after** ``setup_trigger`` — when the trigger can no
        longer free-run, so no orphan frames get saved (Gate-2 save
        windowing: ``GeecsBluesky/CLAUDE.md``).  Save-*off* stays the run
        wrapper's innermost finalize, before the caller's disarm.
    failed_move_policy:
        ``"raise"`` (default): a failed move's ``FailedStatus`` propagates
        normally — exact pre-#641 behavior, so any caller that does not
        opt in is unaffected (in particular the bridge/console path, which
        also sidesteps the coexisting engine-side pause supervisor's
        auto-resume-on-failed-move interaction and the related stop-from-
        paused bypass — both are properties of *entering* the pause path,
        not of this plan).  ``"pause"``: use
        :func:`move_with_failed_move_pause` — a failed move logs the
        documented reason and hard-pauses the RE; resume retries the move
        by replay (decision 4).  Queueserver callers with no supervisor in
        the loop opt into ``"pause"``.
    md:
        Extra metadata merged into the RunEngine ``start`` document.

    Yields
    ------
    Bluesky messages — pass this generator to a :class:`~bluesky.RunEngine`.
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
        if enable_saving is not None:
            # Saving starts only after the trigger is stopped (Gate-2).
            yield from enable_saving()
        move = (
            move_with_failed_move_pause
            if failed_move_policy == "pause"
            else move_changed_axes
        )
        scan_event_index = 0
        previous: tuple | None = None
        for bin_number, pos in enumerate(_positions, start=1):
            # Checkpoint placement is a pause contract (issues #552/#641).
            # Deferred pause: a checkpoint before the move and before every
            # row means request_pause(defer=True) lands with an empty rewind
            # cache — resume replays nothing (no re-move, no re-fire).
            # Hard pause (bps.pause / re pause immediate) resumes by
            # REPLAYING from the last checkpoint, so each checkpoint also
            # bounds what re-executes: the pre-move checkpoint scopes a
            # replay to the move (absolute re-command — idempotent, and the
            # failed-move retry mechanism); the post-move checkpoint keeps
            # the move out of a replayed per_step action prefix; the
            # post-per_step checkpoint (issue #645 cross-vendor addendum,
            # P1) keeps per_step's compiled ActionPlan writes (``bps.abs_set``
            # calls — not guaranteed idempotent, unlike an absolute move) out
            # of a replay landing before arm; the pre-row checkpoint scopes a
            # mid-shot replay to that shot (a re-fire, strict's
            # bounded-refire semantics); the post-rows checkpoint keeps the
            # bin's last COMPLETED row out of a replay landing in the
            # disarm/tail window (re-saving it would duplicate the event
            # row).  Never place a checkpoint between create and save
            # (IllegalMessageSequence).
            #
            # Irreducible residual: a hard pause landing DURING per_step()
            # itself (mid-action, before it yields back to the loop) still
            # replays from the post-move checkpoint through the partial
            # per_step execution — same class as the documented bounded-
            # refire windows, not closable by checkpoint placement alone
            # (would need per-action idempotence in the ActionPlan compiler,
            # out of this module's scope).
            yield from bps.checkpoint()
            if _motors and pos is not None:
                previous = yield from move(_motors, pos, previous)
            yield from bps.checkpoint()
            if per_step is not None:
                # After the move, before this step's plan-owned shots — the
                # machine is quiescent here (strict fires each shot itself).
                yield from per_step()
            yield from bps.checkpoint()
            if arm_trigger is not None:
                yield from arm_trigger()
            for shot_index_in_bin in range(1, shots_per_step + 1):
                yield from bps.checkpoint()
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
            yield from bps.checkpoint()
            if disarm_trigger is not None:
                yield from disarm_trigger()

    yield from _inner()
