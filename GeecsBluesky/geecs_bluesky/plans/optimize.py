"""geecs_adaptive_scan — optimization as a scan (iteration = bin).

Structurally a :func:`~geecs_bluesky.plans.step_scan.geecs_step_scan` whose
"positions" come from a ``propose`` callable between bins instead of a
precomputed list: one Bluesky run, ``bin_number`` = iteration, the same
schema-v1 rows, shot synchronization, arm/disarm bracketing, and (in free-run
mode) the t0-sync stage and tail flush.  The objective evaluation and
generator ask/tell live in *propose* — a plain function called between bins,
by which time every event of the previous bin has already reached the
caller's subscribers.

*propose* can block for seconds (bounded filesystem waits for native files,
ScanAnalysis image analysis, generator fitting).  Plan code executes on the
RunEngine's event-loop thread, so the plan runs *propose* on a worker thread
and idles with RE-friendly ``bps.sleep`` polls while it works — the loop
stays responsive to pause/abort requests, subscriptions (TiledWriter), and
anything else scheduled on it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

from geecs_bluesky.devices.scan_context import ScanContext
from geecs_bluesky.devices.shot_id import ShotIdSupport
from geecs_bluesky.plans.single_shot import geecs_single_shot
from geecs_bluesky.plans.t0_sync import geecs_t0_sync

#: How often the plan checks the in-flight ``propose`` future (RE-friendly
#: ``bps.sleep`` between checks — never a blocking sleep on the RE loop).
_PROPOSE_POLL_S = 0.05


def geecs_adaptive_scan(
    *,
    movables: dict[str, Any],
    propose: Callable[[int], dict[str, float] | None],
    detectors: list[Any],
    shots_per_iteration: int,
    max_iterations: int,
    reference: Any | None = None,
    fire_shot: Callable | None = None,
    setup_trigger: Callable | None = None,
    arm_trigger: Callable | None = None,
    disarm_trigger: Callable | None = None,
    quiesce_trigger: Callable | None = None,
    t0_sync_window_s: float = 0.2,
    tail_flush: bool = True,
    md: dict[str, Any] | None = None,
):
    """Adaptive scan: acquire one bin per suggested point, up to *max_iterations*.

    Parameters
    ----------
    movables:
        ``{variable_name: Movable}`` — the optimization variables.  Their
        readbacks are read into every event row (like a scan motor).
    propose:
        ``propose(iteration) -> {variable_name: value} | None``.  Called before
        each bin; ``None`` ends the run early (generator converged / budget
        spent).  Evaluating the previous bin and telling the generator happen
        inside this callable, on the caller's side; it may block (see the
        module docstring's threading contract).
    detectors:
        Devices read every shot.  With *reference* set (free-run) the first
        trigger belongs to the reference; otherwise strict semantics apply.
    reference:
        Free-run pacemaker (a configured ``ShotIdSupport`` triggerable).
        ``None`` selects strict semantics.
    fire_shot / setup_trigger:
        Strict plan-owned single-shot hooks (see ``geecs_step_scan``).
    arm_trigger / disarm_trigger / quiesce_trigger:
        Shot-control bracketing hooks (see the step plans).
    md:
        Extra start-document metadata.
    """
    free_run = reference is not None
    scan_context = ScanContext()
    _read_devices = (
        ([reference] if free_run else [])
        + list(detectors)
        + list(movables.values())
        + [scan_context]
    )

    _md: dict[str, Any] = {
        "plan_name": "geecs_adaptive_scan",
        "acquisition_mode": (
            "free_run_time_sync" if free_run else "strict_shot_control"
        ),
        "geecs_event_schema": 1,
        "adaptive": True,
        "fires_own_shots": fire_shot is not None,
        "variables": list(movables),
        "detectors": [
            getattr(d, "name", str(d))
            for d in (([reference] if free_run else []) + list(detectors))
        ],
        "shots_per_step": shots_per_iteration,
        "max_iterations": max_iterations,
        **(md or {}),
    }

    if free_run:
        if (
            not isinstance(reference, ShotIdSupport)
            or reference.shot_id_tracker is None
        ):
            raise ValueError(
                "free-run adaptive scan requires a reference with "
                "configure_shot_id() done"
            )
        for det in detectors:
            if hasattr(det, "set_reference"):
                det.set_reference(reference)
        sync_devices = [
            d
            for d in ([reference] + list(detectors))
            if isinstance(d, ShotIdSupport) and d.shot_id_tracker is not None
        ]
        _quiesce = quiesce_trigger or disarm_trigger
        if _quiesce is not None:
            yield from _quiesce()
        t0s = yield from geecs_t0_sync(sync_devices, window_s=t0_sync_window_s)
        _md["device_t0s"] = t0s

    @bpp.run_decorator(md=_md)
    def _inner():
        if not free_run and setup_trigger is not None:
            yield from setup_trigger()
        # propose() can block for seconds; run it on a worker thread and idle
        # with RE-friendly sleeps so the RE loop stays responsive (module
        # docstring).
        propose_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="geecs-propose"
        )
        try:
            scan_event_index = 0
            for iteration in range(1, max_iterations + 1):
                # Deferred-pause boundary (issue #552): same per-iteration +
                # per-row checkpoints as the step plans, so an operator
                # action during an optimize scan can pause it — without
                # these, request_pause(defer=True) never fires and the
                # console would sit in PAUSING while the scan ran on.
                yield from bps.checkpoint()
                proposal = propose_pool.submit(propose, iteration)
                while not proposal.done():
                    yield from bps.sleep(_PROPOSE_POLL_S)
                inputs = proposal.result()
                if inputs is None:
                    break
                mv_args: list[Any] = []
                for name, value in inputs.items():
                    mv_args.extend([movables[name], value])
                if mv_args:
                    yield from bps.mv(*mv_args)
                if arm_trigger is not None:
                    yield from arm_trigger()
                for shot_index_in_bin in range(1, shots_per_iteration + 1):
                    yield from bps.checkpoint()
                    scan_event_index += 1
                    scan_context.set_context(
                        bin_number=iteration,
                        shot_index_in_bin=shot_index_in_bin,
                        scan_event_index=scan_event_index,
                    )
                    if not free_run and fire_shot is not None:
                        yield from geecs_single_shot(_read_devices, fire_shot)
                    else:
                        yield from bps.trigger_and_read(_read_devices)
                if disarm_trigger is not None:
                    yield from disarm_trigger()
        finally:
            propose_pool.shutdown(wait=False, cancel_futures=True)
        if free_run and tail_flush:
            yield from bps.create(name="flush")
            for dev in _read_devices:
                yield from bps.read(dev)
            yield from bps.save()

    yield from _inner()
