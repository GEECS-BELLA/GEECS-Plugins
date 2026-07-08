"""The one scan-orchestration recipe shared by GeecsSession and BlueskyScanner.

Composes the acquisition-mode plan (free-run reference-paced vs strict
plan-owned single-shot), the run wrapper (scan numbering, native saving, run
metadata), the compiled setup/per-step/closeout action plans, and the
guaranteed finalize-disarm into a single ready-to-run plan.  Device
construction and configuration stay with the caller — the session builds CA
devices from factories, the scanner builds either backend from
``exec_config`` — but the *recipe* lives only here, so the two front doors
cannot drift.

Action placement (the §4.4b/§4.5 seams, decided here):

- **setup** runs first thing inside the composed plan — after every device
  is connected (construction-time) and the pre-flight has passed (both
  happen before the RunEngine ever sees this plan), and *before* the
  free-run quiesce/t0-sync stage and the first step — so setup actions
  settle device state before timing synchronization, and a failing setup
  still triggers the finalize chain (saving off → disarm → closeout).
- **per_step** is yielded by the step plans at every step boundary: after
  the move completes, before that step's shots (free-run brackets each step
  with arm/disarm, so per-step actions run *disarmed*; strict fires each
  shot itself, so the machine is quiescent between plan-owned shots).
- **closeout** is the outermost ``finalize_wrapper`` — it runs even on
  mid-scan abort (legacy ActionControl parity), and because it wraps the
  disarm finalize it always executes *after* the trigger has returned to
  STANDBY (data-taking output off, trigger free-running).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import bluesky.preprocessors as bpp

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plans.free_run_step_scan import geecs_free_run_step_scan
from geecs_bluesky.plans.run_wrapper import geecs_run_wrapper
from geecs_bluesky.plans.step_scan import geecs_step_scan, normalize_motors
from geecs_bluesky.shot_controller import ShotController

logger = logging.getLogger(__name__)


def _chain_setup(setup: Callable, inner):
    """Prepend the setup plan to *inner* (fresh generator via the callable)."""
    yield from setup()
    yield from inner


def build_step_scan_plan(
    *,
    strict: bool,
    motor: Any | Sequence[Any] | None,
    positions: Sequence[Any],
    reference: Any | None,
    detectors: Sequence[Any],
    shots_per_step: int,
    controller: ShotController | None,
    experiment: str,
    scan_number: int | None,
    scan_folder: str | None,
    saving_detectors: Sequence[tuple],
    extra_md: dict[str, Any] | None = None,
    setup: Callable | None = None,
    per_step: Callable | None = None,
    closeout: Callable | None = None,
):
    """Build the full scan plan for one step scan / statistics collection.

    Parameters
    ----------
    strict : bool
        ``True`` for plan-owned single-shot (``strict_shot_control``),
        ``False`` for reference-paced free-run (``free_run_time_sync``).
    motor, positions
        The scan axis (or axes — a sequence of Movables is a grid, outermost
        first, with tuple positions) and the positions to visit;
        ``motor=None`` with ``[None]`` is statistics collection (one no-move
        bin).
    reference : Any or None
        Free-run pacemaker (required when not strict).
    detectors : sequence
        All detectors, *including* the reference.
    controller : ShotController or None
        Shot control; optional in free-run (arm/disarm/quiesce become no-ops),
        required in strict.
    experiment, scan_number, scan_folder, saving_detectors, extra_md
        Forwarded to :func:`~geecs_bluesky.plans.run_wrapper.geecs_run_wrapper`.
    setup, per_step, closeout : callable, optional
        Plan-stub callables (each call returns a fresh message generator) —
        typically compiled ActionPlans.  See the module docstring for the
        exact placement and abort semantics of each hook.

    Returns
    -------
    generator
        The composed plan (setup + run wrapper + finalize disarm + finalize
        closeout), ready for ``RE()``.
    """
    detectors = list(detectors)

    if strict:
        if controller is None:
            raise GeecsConfigurationError(
                "strict_shot_control requires a reachable shot-control device. "
                "Use free-run mode for free-running trigger acquisition."
            )
        controller.require_strict_single_shot()
        logger.info("strict mode: plan-owned single-shot (ARMED + fire SINGLESHOT)")
        inner = geecs_step_scan(
            motor=motor,
            positions=list(positions),
            detectors=detectors,
            shots_per_step=shots_per_step,
            setup_trigger=lambda: controller.arm_single_shot(detectors),
            fire_shot=controller.fire_shot,
            per_step=per_step,
        )
    else:
        if reference is None:
            raise GeecsConfigurationError(
                "free-run scans require at least one synchronous device as "
                "the reference (pacemaker)"
            )
        contributors = [d for d in detectors if d is not reference]
        inner = geecs_free_run_step_scan(
            motor=motor,
            positions=list(positions),
            reference=reference,
            detectors=contributors,
            shots_per_step=shots_per_step,
            arm_trigger=controller.arm if controller else None,
            disarm_trigger=controller.disarm if controller else None,
            quiesce_trigger=controller.quiesce if controller else None,
            per_step=per_step,
        )

    if setup is not None:
        # Setup runs before the free-run quiesce/t0-sync and before the
        # first step (module docstring); inside the run wrapper so a failed
        # setup still fires the save-off/disarm/closeout finalizes.
        inner = _chain_setup(setup, inner)

    scalar_devices = detectors + normalize_motors(motor)
    plan = geecs_run_wrapper(
        inner,
        experiment=experiment,
        scan_number=scan_number,
        scan_folder=scan_folder,
        saving_detectors=list(saving_detectors),
        devices=scalar_devices,
        extra_md=extra_md or {},
    )
    # Finalize nesting (innermost → outermost): save-off (inside the run
    # wrapper) → disarm (→ STANDBY) → closeout actions.  Every layer runs
    # even on mid-scan abort; closeout therefore always executes with the
    # trigger already disarmed.
    if controller is not None:
        plan = bpp.finalize_wrapper(plan, controller.disarm())
    if closeout is not None:
        plan = bpp.finalize_wrapper(plan, closeout)
    return plan
