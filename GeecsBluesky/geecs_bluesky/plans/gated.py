"""Gated batch acquisition as the stock ``take_reading`` hook, and the non-essential stream.

Phase 2 of the native-Bluesky rebuild (``Planning/native_bluesky/08_gated_batch.md``
§4.2, §4.3, §4.7; ``03_clean_room_rebuild.md`` §10.9).  Strict single-shot
(:mod:`geecs_bluesky.plans.strict`) fires the box once per row and holds
1 Hz only at short exposures (``05`` M6/M7); the **gated batch** lets the
box free-run in SCAN while the plugin-backed cameras count the frames they
write, and drives it OFF when every essential detector has its quota —
exact by construction, because arming precedes the edges and the frames
are counted by the thing that writes them.

Per step (after ``move_per_step``), for *D* = the plugin-backed essential
detectors, *S* = the per-shot sampler over every other device of the step
(:class:`~geecs_bluesky.devices.sampler.ShotSampler`), and the box *B*::

    mv(B, OFF)                                   # the step opens quiet — also after a
                                                 #   resume, which restored SCAN first
    if the run's first step:                     # arm, then zero the plugin's stale count
        prepare(D); wait_for(D.zero_count)
    if repeating (an immediate pause interrupted the step):
        sleep(period + max drain + margin)       # the in-flight frame lands
        wait_for(D.rewind_to_step_baseline)      # the partial frames leave the stacks
    prepare(D, gated_trigger_info(quota))        # capture on, count baselined
    prepare(S, quota)                            # the sampler: clock + columns
    declare_stream(*D, name="primary")           # first step only
    declare_stream(S, name="shots")              # first step only
    kickoff(*D, S)                               # quota armed, fly mode
    mv(B, SCAN)                                  # edges flow
    complete(*D, S)                              # every D (and S) counted its quota
    mv(B, OFF)                                   # edges stop
    sleep(period + max drain + margin)           # the in-flight frame lands
    wait_for(D.truncate_to_quota)                # rewind to baseline + quota
    collect(*D, name="primary")                  # one datum per D: the step's frames
    collect(S, name="shots")                     # one event per shot: everything else

Two streams per gated run: ``primary`` carries the frames and their
per-frame attributes (a datum stream, no events); ``shots`` carries one
event per shot with the clock stamp, the motors' readbacks, ``bin_number``
and every non-plugin scalar.  With no plugin-backed camera *D* is empty
and the sampler alone gates the step; a run with no essential triggered
device at all is refused ("nothing counts shots; use strict").

**Pause** (Sam, 2026-09-12): the step body is not rewindable and holds no
checkpoint, so a *deferred* pause lands between steps (the stock
``move_per_step`` checkpoint) — a real pause in both modes.  An
*immediate* pause mid-batch drives the box OFF (``ShotControl.pause``);
on resume the RunEngine restores SCAN before the plan runs again, the
plan sees the box's pause counter advanced, abandons the batch (the
pending ``complete`` statuses settle instead of failing into a later
message), rewinds every plugin to the step's baseline and **retakes the
step from its first shot**.

The **non-essential stream** (§4.3) is the run-long job: the detectors
listed ``non_essential=[…]`` are staged, prepared unbounded, kicked off
right after ``open_run`` (the box is quiet then, so no frame lands between
prepare and kickoff — which would make the kickoff refuse), completed and
collected each into its own ``<name>_stream`` before ``close_run``.
Nothing waits on them: a slow or dying non-essential camera never holds a
shot and never aborts a run.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable

import bluesky.plan_stubs as bps
from bluesky.preprocessors import (
    contingency_wrapper,
    finalize_wrapper,
    plan_mutator,
    rewindable_wrapper,
)
from bluesky.utils import (
    FailedStatus,
    Msg,
    ensure_generator,
    separate_devices,
    short_uid,
    single_gen,
)
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.devices.ca._view import ScalarsView
from geecs_bluesky.devices.detector import (
    DEFAULT_SHOT_TIMEOUT,
    UNBOUNDED_TRIGGER_INFO,
    GeecsDetector,
    gated_trigger_info,
)
from geecs_bluesky.devices.sampler import ShotSampler
from geecs_bluesky.exceptions import GeecsConfigurationError, GeecsTriggerTimeoutError
from geecs_bluesky.plans.strict import BinCounter

logger = logging.getLogger(__name__)

#: The trigger period the drain wait budgets, seconds — the laser rep rate
#: (HTU: 1 Hz).  After the box goes OFF at most one edge is in flight; its
#: frame lands within one period plus the camera's drain offset.
TRIGGER_PERIOD_S = 1.0
#: Margin on top of the period and the largest drain offset.
DRAIN_MARGIN_S = 0.25
#: The event stream the sampler's rows go to.
SHOTS_STREAM = "shots"


def shot_clock(devices: Sequence[Any]) -> tuple[Any, str]:
    """The shot clock of a gated step: an essential triggered device's stamp signal.

    The first plugin-backed camera, else the first triggered device without
    one (a scalar device with a stamp, or a camera's ``.scalars`` view) —
    the recommended default; a step with several candidates uses the first
    in the plan's detector order.

    Returns
    -------
    tuple
        ``(acq_timestamp signal, GEECS device name)``.

    Raises
    ------
    GeecsConfigurationError
        No triggered device in the step — nothing counts shots.
    """
    plugin = [d for d in devices if isinstance(d, GeecsDetector) and d.plugin_backed]
    others = [
        d
        for d in devices
        if (isinstance(d, GeecsDetector) and not d.plugin_backed)
        or (isinstance(d, ScalarsView) and isinstance(d._owner, GeecsDetector))
    ]
    for candidate in [*plugin, *others]:
        owner = candidate._owner if isinstance(candidate, ScalarsView) else candidate
        return owner.acq_timestamp, owner._geecs_device_name
    raise GeecsConfigurationError(
        "a gated run needs at least one essential triggered device (a camera "
        "or a triggered scalar device) — nothing here counts shots; use "
        "acquisition='strict'"
    )


def refuse_native_essentials(devices: Sequence[Any]) -> None:
    """Refuse a LabVIEW-native saving device as an essential of a gated run.

    The preflight's rule, worker-side: a gated batch counts frames the
    plugin writes, and a device that saves natively without one (a
    LabVIEW-native camera, a DAQ with its own file writer) can neither
    count nor stream.
    Called by the bound plan at bind time (before the run is claimed or
    anything moves) and by the step itself.

    Raises
    ------
    GeecsConfigurationError
        Naming the cameras.
    """
    native = [
        d
        for d in devices
        if isinstance(d, GeecsDetector) and d.native_save and not d.plugin_backed
    ]
    if native:
        names = ", ".join(d._geecs_device_name for d in native)
        raise GeecsConfigurationError(
            f"gated acquisition: native-saving device(s) without a file plugin: "
            f"{names} — a gated batch counts frames the plugin writes; a "
            "device saving through LabVIEW cannot. Use acquisition='strict' "
            "or record its scalars only (save_images: false)."
        )


def gated_take_reading(
    shot_control: Any,
    *,
    name: str = "primary",
    shots_stream: str = SHOTS_STREAM,
    shot_timeout: float = DEFAULT_SHOT_TIMEOUT,
) -> Callable[[Sequence[Any], int], Any]:
    """Return a ``take_reading(devices, quota)`` that runs one gated batch.

    One closure per plan call: the sampler is built on the first step and
    reused (the ``shots`` stream is declared for exactly that object), and
    the two streams are declared once.

    Parameters
    ----------
    shot_control :
        The :class:`~geecs_bluesky.devices.shot_control.ShotControl`; the
        plan drives it OFF → SCAN → OFF around the batch and reads its
        pause counter to learn the batch was interrupted.
    name :
        The datum stream (the cameras' frames).
    shots_stream :
        The event stream (the sampler's rows).
    shot_timeout :
        Per-frame budget for the cameras and per-tick budget for the sampler.
    """
    state: dict[str, Any] = {"sampler": None, "declared": False, "steps": 0}

    def take_reading(devices: Sequence[Any], quota: int):
        devices = separate_devices(devices)
        views = [d for d in devices if isinstance(d, ScalarsView)]
        devices = [d for d in devices if not any(v.covers(d) for v in views)]
        plugin = [
            d for d in devices if isinstance(d, GeecsDetector) and d.plugin_backed
        ]
        members = [d for d in devices if d not in plugin]
        refuse_native_essentials(members)
        sampler = state["sampler"]
        if sampler is None:
            clock, clock_name = shot_clock(devices)
            sampler = ShotSampler(
                members, clock, clock_name=clock_name, shot_timeout=shot_timeout
            )
            state["sampler"] = sampler
        elif [id(m) for m in sampler.members] != [id(m) for m in members]:
            raise GeecsConfigurationError(
                "a gated run's device set changed between steps — the shots "
                "stream is declared once for one set of devices"
            )
        drain = TRIGGER_PERIOD_S + DRAIN_MARGIN_S
        if plugin:
            offsets = []
            for d in plugin:
                offsets.append(float((yield from bps.rd(d.drain_offset)) or 0.0))
            drain += max(offsets)
        mark = getattr(shot_control, "pause_count", 0)
        first_step = state["steps"] == 0
        attempt = 0
        while True:
            attempt += 1
            yield from bps.mv(shot_control, TriggerState.OFF.value)
            if attempt > 1 or first_step:
                # The in-flight frame lands (STANDBY passed edges before the
                # run opened; a resume restored SCAN before the plan ran).
                yield from bps.sleep(drain)
                if attempt > 1 and plugin:
                    yield from bps.wait_for([d.rewind_to_step_baseline for d in plugin])
            info = gated_trigger_info(quota, exposure_timeout=shot_timeout)
            if plugin and first_step and attempt == 1:
                # The run's first arm: the plugin's NumCaptured_RBV still
                # reads the previous session's count until a frame lands
                # (found on hardware, A2), so arm, zero the count inside the
                # fresh session, and let the prepare below baseline on 0.
                group = short_uid("gated-arm")
                for d in plugin:
                    yield from bps.prepare(d, info, group=group, wait=False)
                yield from bps.wait(group=group)
                yield from bps.wait_for([d.zero_count for d in plugin])
            group = short_uid("gated-prepare")
            for d in plugin:
                yield from bps.prepare(d, info, group=group, wait=False)
            yield from bps.prepare(sampler, quota, group=group, wait=False)
            yield from bps.wait(group=group)
            if not state["declared"]:
                if plugin:
                    yield from bps.declare_stream(*plugin, name=name, collect=True)
                yield from bps.declare_stream(sampler, name=shots_stream, collect=True)
                state["declared"] = True
            yield from bps.kickoff_all(*plugin, sampler, wait=True)
            yield from bps.mv(shot_control, TriggerState.SCAN.value)
            failure: FailedStatus | None = None
            try:
                yield from bps.complete_all(*plugin, sampler, wait=True)
            except FailedStatus as exc:
                failure = exc
            interrupted = getattr(shot_control, "pause_count", 0) != mark
            if interrupted or failure is not None:
                # The batch is over: say so NOW, before yielding another
                # message — a pending complete timing out in the next loop
                # iteration is then pardoned instead of thrown into the plan.
                for d in plugin:
                    d.mark_abandoned()
                sampler.mark_cancelled()
            try:
                yield from bps.mv(shot_control, TriggerState.OFF.value)
                if failure is None and not interrupted:
                    # The in-flight edge lands; a pause landing here counts
                    # too (its resume restored SCAN and more edges came).
                    yield from bps.sleep(drain)
                    interrupted = getattr(shot_control, "pause_count", 0) != mark
                    if interrupted:
                        for d in plugin:
                            d.mark_abandoned()
                        sampler.mark_cancelled()
            except FailedStatus as exc:
                # A status of this batch failed between the wait's return and
                # the mark (one loop iteration): the same abandon path.
                failure = failure or exc
                for d in plugin:
                    d.mark_abandoned()
                sampler.mark_cancelled()
                yield from bps.mv(shot_control, TriggerState.OFF.value)
            if interrupted or failure is not None:
                # Settle the batch's pending statuses before anything else.
                yield from bps.wait_for(
                    [d.abandon_step for d in plugin] + [sampler.cancel_step]
                )
            if interrupted:
                mark = getattr(shot_control, "pause_count", 0)
                logger.warning(
                    "gated step interrupted by a pause after %d/%d shot(s) — "
                    "retaking the step from its first shot",
                    sampler.sampled,
                    quota,
                )
                continue
            if failure is not None:
                cause = failure.__cause__
                if isinstance(cause, GeecsTriggerTimeoutError):
                    raise cause
                raise GeecsTriggerTimeoutError(
                    getattr(cause, "device_name", None) or "gated batch",
                    shot_timeout,
                    f"gated batch failed: {cause!r}",
                ) from failure
            if plugin:
                yield from bps.wait_for([d.truncate_to_quota for d in plugin])
                yield from bps.collect(*plugin, name=name)
            yield from bps.collect(sampler, name=shots_stream)
            state["steps"] += 1
            return None

    return take_reading


def gated_per_shot(shot_control: Any, **kwargs: Any) -> Callable[..., Any]:
    """``bp.count(..., per_shot=gated_per_shot(shot_control))``: one batch of ``num``.

    A gated count is one batch: ``bp.count`` calls the hook ``num`` times,
    so the first call takes the whole batch (``num`` read off the run's
    ``per_shot`` repetition is not available here — the batch size is the
    hook's own ``quota`` keyword, set by the bound plan from ``num``) and
    the later calls are no-ops.  Every row carries ``bin_number = 1``.
    """
    quota = int(kwargs.pop("quota", 1))
    take_reading = gated_take_reading(shot_control, **kwargs)
    bins = BinCounter()
    bins.value = 1
    state = {"taken": False}

    def per_shot(detectors: Sequence[Any], take_reading_: Any = None):
        if state["taken"]:
            return None
        state["taken"] = True
        yield Msg("checkpoint")
        body = take_reading([*detectors, bins], quota)
        # Not rewindable: a resume must not replay the batch's messages —
        # the plan retakes the step itself (module docstring, Pause).
        return (yield from rewindable_wrapper(body, False))

    per_shot.__name__ = per_shot.__qualname__ = "gated_per_shot"
    return per_shot


def gated_per_step(
    shot_control: Any, *, shots_per_step: int = 1, **kwargs: Any
) -> Callable[..., Any]:
    """``bp.scan(..., per_step=gated_per_step(shot_control))``: one batch per position.

    ``bps.one_nd_step`` with the batch in place of the strict shots: after
    the move, *shots_per_step* shots are taken in one gated batch — the
    cameras' frames into ``primary``, one ``shots`` row per shot with the
    motors' readbacks and the step's ``bin_number``.
    """
    if shots_per_step < 1:
        raise ValueError(f"shots_per_step must be >= 1, got {shots_per_step}")
    take_reading = gated_take_reading(shot_control, **kwargs)
    bins = BinCounter()

    def per_step(
        detectors: Sequence[Any], step: Any, pos_cache: Any, take_reading_: Any = None
    ):
        motors = list(step.keys())
        yield from bps.move_per_step(step, pos_cache)
        bins.value += 1
        body = take_reading([*detectors, *motors, bins], shots_per_step)
        return (yield from rewindable_wrapper(body, False))

    per_step.__name__ = per_step.__qualname__ = "gated_per_step"
    return per_step


def non_essential_wrapper(plan: Any, flyers: Sequence[Any]) -> Any:
    """Stream *flyers* for the run's duration, each into its own ``<name>_stream``.

    The stock ``fly_during_wrapper`` inserts ``kickoff`` after ``open_run``
    and ``complete`` + ``collect`` before ``close_run`` but neither stages
    nor prepares; this one does both — stage (a flyer dead at stage time
    fails loudly, as it should), then, right after ``open_run``, an
    unbounded prepare and the stream declaration that routes the collect,
    then the kickoff while the box is still quiet.  Each flyer is collected
    alone (one object: no index, the datum covers everything it wrote), in
    its own stream — a joint stream would cut every camera at the slowest
    one.  From the run's close on, nothing of a non-essential device may
    fail the item: its ``complete`` + ``collect`` and its ``unstage`` are
    each a contingency, logged and skipped (a gateway that went away
    mid-run).  A skipped unstage is not retried by the RunEngine (the object
    leaves its staged set before the status resolves): a stale ``Capture``
    is cleared by the camera's next ``stage()``, and its signal caches by
    its next unstage — nothing functional leaks.

    Parameters
    ----------
    plan :
        The bound stock plan.
    flyers :
        The non-essential detectors (plugin-backed: a device without a
        streamable provider fails ``kickoff`` loudly, "not streamable").
    """
    flyers = list(flyers)
    if not flyers:
        return (yield from plan)

    def stream_name(flyer: Any) -> str:
        return f"{flyer.name}_stream"

    def after_open():
        group = short_uid("non-essential-prepare")
        for flyer in flyers:
            yield from bps.prepare(
                flyer, UNBOUNDED_TRIGGER_INFO, group=group, wait=False
            )
        yield from bps.wait(group=group)
        # The plugin's count PV still reads the previous session's total at
        # the arm (GEECS-Plugins#853): zero it inside the fresh session and
        # prepare again, or the kickoff baselines above what the run will
        # write and the close's count wait never returns (2b A4).
        zero = [f for f in flyers if hasattr(f, "zero_count")]
        if zero:
            yield from bps.wait_for([f.zero_count for f in zero])
            group = short_uid("non-essential-prepare-zeroed")
            for flyer in zero:
                yield from bps.prepare(
                    flyer, UNBOUNDED_TRIGGER_INFO, group=group, wait=False
                )
            yield from bps.wait(group=group)
        for flyer in flyers:
            yield from bps.declare_stream(flyer, name=stream_name(flyer), collect=True)
        yield from bps.kickoff_all(*flyers, wait=True)

    def before_close():
        # Nothing waits on a non-essential device — a plugin whose gateway
        # went away mid-run must not fail the run at its close: each flyer's
        # complete + collect is its own contingency, logged and skipped.
        for flyer in flyers:
            # complete and collect are SEPARATE contingencies: a complete
            # that fails (a stalled or dead plugin) must not cost the
            # datums for the frames it did write.
            for verb, plan_factory in (
                ("complete", lambda f=flyer: bps.complete(f, wait=True)),
                ("collect", lambda f=flyer: bps.collect(f, name=stream_name(f))),
            ):

                def skip(exc, flyer=flyer, verb=verb):
                    logger.warning(
                        "non-essential %s: %s failed at the run's close (%s: %s) — "
                        "its stream carries what it wrote",
                        flyer.name,
                        verb,
                        type(exc).__name__,
                        exc,
                    )
                    yield from bps.null()

                yield from contingency_wrapper(
                    plan_factory(), except_plan=skip, auto_raise=False
                )

    def insert_after_open(msg: Msg):
        if msg.command == "open_run":
            return single_gen(msg), ensure_generator(after_open())
        return None, None

    def insert_before_close(msg: Msg):
        if msg.command == "close_run":

            def new_gen():
                yield from before_close()
                yield msg

            return new_gen(), None
        return None, None

    inner = plan_mutator(plan_mutator(plan, insert_after_open), insert_before_close)

    def unstage_all_tolerant():
        for flyer in flyers:

            def one(flyer=flyer):
                # Wait inside the contingency: a failure of the unstage
                # status must land here, not at the next message outside.
                yield from bps.unstage(flyer, group=short_uid("ne-unstage"), wait=True)

            def skip(exc, flyer=flyer):
                logger.warning(
                    "non-essential %s: unstage failed (%s: %s) — skipped; a stale "
                    "Capture is cleared by its next stage()",
                    flyer.name,
                    type(exc).__name__,
                    exc,
                )
                yield from bps.null()

            yield from contingency_wrapper(one(), except_plan=skip, auto_raise=False)

    def staged():
        yield from bps.stage_all(*flyers)
        return (yield from inner)

    return (yield from finalize_wrapper(staged(), unstage_all_tolerant()))


def run_bracket(plan: Any, shot_control: Any, opening: TriggerState) -> Any:
    """Bracket a run *opening* → … → STANDBY through *shot_control*.

    Strict opens ARMED (the single-shot source); gated opens OFF (quiet —
    the first step also waits one drain period before it arms, for the
    frame an edge under STANDBY may have in flight).  Both close in the
    machine's idle state, whatever the plan did.
    """
    yield from bps.mv(shot_control, opening.value)

    def standby():
        yield from bps.mv(shot_control, TriggerState.STANDBY.value)

    return (yield from finalize_wrapper(plan, standby()))


__all__ = [
    "DRAIN_MARGIN_S",
    "SHOTS_STREAM",
    "TRIGGER_PERIOD_S",
    "gated_per_shot",
    "gated_per_step",
    "gated_take_reading",
    "non_essential_wrapper",
    "refuse_native_essentials",
    "run_bracket",
    "shot_clock",
]
