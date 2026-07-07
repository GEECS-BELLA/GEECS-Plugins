"""The one scan-orchestration recipe shared by GeecsSession and BlueskyScanner.

Composes the acquisition-mode plan (free-run reference-paced vs strict
plan-owned single-shot), the run wrapper (scan numbering, native saving, run
metadata), and the guaranteed finalize-disarm into a single ready-to-run plan.
Device construction and configuration stay with the caller — the session
builds CA devices from factories, the scanner builds either backend from
``exec_config`` — but the *recipe* lives only here, so the two front doors
cannot drift.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import bluesky.preprocessors as bpp

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plans.free_run_step_scan import geecs_free_run_step_scan
from geecs_bluesky.plans.run_wrapper import geecs_run_wrapper
from geecs_bluesky.plans.step_scan import geecs_step_scan
from geecs_bluesky.shot_controller import ShotController

logger = logging.getLogger(__name__)


def build_step_scan_plan(
    *,
    strict: bool,
    motor: Any | None,
    positions: Sequence[float | None],
    reference: Any | None,
    detectors: Sequence[Any],
    shots_per_step: int,
    controller: ShotController | None,
    experiment: str,
    scan_number: int | None,
    scan_folder: str | None,
    saving_detectors: Sequence[tuple],
    extra_md: dict[str, Any] | None = None,
):
    """Build the full scan plan for one step scan / statistics collection.

    Parameters
    ----------
    strict : bool
        ``True`` for plan-owned single-shot (``strict_shot_control``),
        ``False`` for reference-paced free-run (``free_run_time_sync``).
    motor, positions
        The scan axis and its positions; ``motor=None`` with ``[None]`` is
        statistics collection (one no-move bin).
    reference : Any or None
        Free-run pacemaker (required when not strict).
    detectors : sequence
        All detectors, *including* the reference.
    controller : ShotController or None
        Shot control; optional in free-run (arm/disarm/quiesce become no-ops),
        required in strict.
    experiment, scan_number, scan_folder, saving_detectors, extra_md
        Forwarded to :func:`~geecs_bluesky.plans.run_wrapper.geecs_run_wrapper`.

    Returns
    -------
    generator
        The composed plan (run wrapper + finalize disarm), ready for ``RE()``.
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
        )

    scalar_devices = detectors + ([motor] if motor is not None else [])
    plan = geecs_run_wrapper(
        inner,
        experiment=experiment,
        scan_number=scan_number,
        scan_folder=scan_folder,
        saving_detectors=list(saving_detectors),
        devices=scalar_devices,
        extra_md=extra_md or {},
    )
    # Outer finalize guarantees the disarm (→ STANDBY) even on mid-scan abort.
    if controller is not None:
        plan = bpp.finalize_wrapper(plan, controller.disarm())
    return plan
