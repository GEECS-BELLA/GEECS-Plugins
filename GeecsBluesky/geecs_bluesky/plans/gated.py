"""Gated batch acquisition as the stock ``take_reading`` hook, and the non-essential stream.

Strict single-shot (:mod:`geecs_bluesky.plans.strict`) fires the box once
per row; the **gated batch** lets the box free-run in SCAN while the
plugin-backed cameras count the frames they write, and drives it OFF when
every essential detector has its quota.  Per step, for *D* the
plugin-backed essentials, *N* the LabVIEW-native saving essentials
(:func:`native_essentials`), *S* the per-shot sampler over every other
device (:class:`~geecs_bluesky.devices.sampler.ShotSampler`) and the box
*B*::

    mv(B, OFF)
    if the run's first step:
        sleep(period + max drain + margin)       # the in-flight frame lands
        prepare(N, unbounded)                    # saving on, run-long
        prepare(D); wait_for(D.zero_count)       # arm, zero the stale count
    prepare(D, gated_trigger_info(remaining)); prepare(S, remaining)
    declare_stream(*D, "primary"); declare_stream(S, "shots")   # once
    kickoff(*D, S); mv(B, SCAN)
    complete(*D, S) in slices of PROGRESS_PERIOD_S:
        collect(S, "shots"); checkpoint          # rows every D holds a frame of
    mv(B, OFF); sleep(period + max drain + margin)
    wait_for(D.truncate_to_quota)
    collect(*D, "primary"); collect(S, "shots")

``primary`` is a datum stream (the frames and their per-frame
attributes); ``shots`` carries one event per shot.  With no plugin-backed
camera the sampler alone gates the step; a run with no essential
triggered device is refused ("nothing counts shots; use strict").

**Pause means pause now**: a pause mid-batch drives the box OFF
(``ShotControl.pause``); the batch holds the box
(``ShotControl.hold_for_batch``), so the pause ends it at once and the
resume restores nothing.  The plan keeps the shots every device reached
(``truncate_to``; rows past them never went out), records them, and
continues the step with the remaining shots.  At most the in-flight shot
is lost; the step body is not rewindable.

The **non-essential stream**: the devices in ``non_essential=[…]`` are
staged, prepared unbounded and kicked off right after ``open_run``, then
completed and collected each into its own ``<name>_stream`` before
``close_run``.  A plugin-backed camera flies itself; a triggered device
without a plugin is recorded by a
:class:`~geecs_bluesky.devices.sampler.StampStream`.  Nothing waits on
them: a slow or dying non-essential device never holds a shot or aborts
a run, in either mode.
"""

from __future__ import annotations

import asyncio
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

# The stream the sampler's rows go to: defined in ``geecs_data_utils.shot_join``,
# the one home of that document contract — the plan writes it, the s-file
# callback reads it, and the offline re-export reads it back out of Tiled.
from geecs_data_utils.shot_join import SHOTS_STREAM, non_essential_stream
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.devices.ca._view import ScalarsView, owner_of
from geecs_bluesky.devices.detector import (
    DEFAULT_SHOT_TIMEOUT,
    UNBOUNDED_TRIGGER_INFO,
    GeecsDetector,
    gated_trigger_info,
)
from geecs_bluesky.devices.sampler import ShotSampler, StampStream
from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    GeecsTriggerTimeoutError,
    failure_cause_text,
)
from geecs_bluesky.plans.strict import BinCounter, admit_background, name_failed_status

logger = logging.getLogger(__name__)

#: The trigger period the drain wait budgets, seconds — the laser rep rate
#: (HTU: 1 Hz).  After the box goes OFF at most one edge is in flight; its
#: frame lands within one period plus the camera's drain offset.
TRIGGER_PERIOD_S = 1.0
#: Margin on top of the period and the largest drain offset.
DRAIN_MARGIN_S = 0.25
#: How often a running batch collects its settled rows and offers a
#: checkpoint: the scanner's progress cadence and the latency of its Pause.
PROGRESS_PERIOD_S = 1.0


def shot_clock(devices: Sequence[Any]) -> tuple[Any, str]:
    """The shot clock of a gated step: an essential triggered device's stamp signal.

    The first camera **with a file plugin**, else the first triggered device
    without one (a scalar device with a stamp, or a camera's ``.scalars``
    view) — the recommended default; a step with several candidates uses the
    first in the plan's detector order.  ``plugin_backed`` is a static fact
    since gating moved to the DB, so this answers the same at bind time and
    at step time by construction; a scope with every channel disabled has no
    plugin and is ranked with the plain triggered devices, its stamp still
    advancing per shot.

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
        owner = owner_of(candidate)
        return owner.acq_timestamp, owner._geecs_device_name
    raise GeecsConfigurationError(
        "a gated run needs at least one essential triggered device (a camera "
        "or a triggered scalar device) — nothing here counts shots; use "
        "acquisition='strict'"
    )


def native_essentials(devices: Sequence[Any]) -> list[GeecsDetector]:
    """The LabVIEW-native saving essentials of a gated step: no plugin, saving controls.

    A device without a file plugin — a LabVIEW-native camera, a DAQ or a
    wavefront sensor with its own file writer, a gated devicetype whose
    every capture channel is disabled in the DB — is an essential of a
    gated run exactly as strict treats it: its row is
    the sampler's and its files follow by stamp; the plugin count is a
    convenience, not what makes a batch.  The plan prepares these **once**,
    at the run's first step, unbounded: the device's own lifecycle switches
    LabVIEW's saving on then and off at ``unstage``.  A ``.scalars`` view
    is never one — the view leaves its owner's data logics unprepared.
    """
    return [
        d
        for d in devices
        if isinstance(d, GeecsDetector) and d.native_save and not d.plugin_backed
    ]


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
        native = native_essentials(members)
        sampler = state["sampler"]
        if sampler is None:
            clock, clock_name = shot_clock(devices)
            sampler = ShotSampler(
                members,
                clock,
                clock_name=clock_name,
                shot_timeout=shot_timeout,
                gates=plugin,
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
        first_step = state["steps"] == 0
        done = 0  # the step's shots already recorded (a pause keeps them)
        batches = 0
        while True:
            batches += 1
            first_batch = first_step and batches == 1
            yield from bps.mv(shot_control, TriggerState.OFF.value)
            if first_batch:
                # The in-flight frame lands (STANDBY passed edges before the
                # run opened).
                yield from bps.sleep(drain)
            if native and first_batch:
                # The native-saving essentials: ONE prepare per run, here,
                # with the box quiet — the device's lifecycle switches
                # LabVIEW's saving on for the run (off at unstage).  Never
                # per step: a toggle costs the device one LabVIEW loop
                # period, and between steps the box is OFF so a well-behaved
                # device writes nothing.  Nothing waits on them: a dropped
                # frame is a missing file, not a retake (as the LabVIEW
                # scanner had it for years).  They are sampler members: the
                # row carries their scalars, their stamp and the save path.
                group = short_uid("gated-native")
                for d in native:
                    yield from bps.prepare(
                        d, UNBOUNDED_TRIGGER_INFO, group=group, wait=False
                    )
                yield from bps.wait(group=group)
            remaining = quota - done
            info = gated_trigger_info(remaining, exposure_timeout=shot_timeout)
            if plugin and first_batch:
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
            yield from bps.prepare(sampler, remaining, group=group, wait=False)
            yield from bps.wait(group=group)
            if not state["declared"]:
                if plugin:
                    yield from bps.declare_stream(*plugin, name=name, collect=True)
                yield from bps.declare_stream(sampler, name=shots_stream, collect=True)
                state["declared"] = True
            yield from bps.kickoff_all(*plugin, sampler, wait=True)

            def end_batch() -> None:
                # Synchronously: the batch is over — its pending statuses
                # settle instead of failing into whatever message the plan
                # is at next (the RunEngine throws a failed status there).
                for d in plugin:
                    d.mark_abandoned()
                sampler.mark_cancelled()

            mark = shot_control.pause_count
            finished = False
            failure: FailedStatus | None = None
            shot_control.hold_for_batch(end_batch)
            try:
                try:
                    yield from bps.mv(shot_control, TriggerState.SCAN.value)
                    group = short_uid("gated-complete")
                    yield from bps.complete_all(
                        *plugin, sampler, group=group, wait=False
                    )
                    # The batch's wait, in slices: each slice collects the
                    # rows every camera holds the frame of (the scanner's
                    # progress moves shot by shot) and offers a checkpoint,
                    # where a deferred pause — the scanner's Pause — lands
                    # mid-batch.
                    while True:
                        finished = yield from bps.wait(
                            group=group,
                            timeout=PROGRESS_PERIOD_S,
                            error_on_timeout=False,
                        )
                        if shot_control.pause_count != mark:
                            finished = False  # an immediate pause cut the wait
                            break
                        yield from bps.collect(sampler, name=shots_stream)
                        if finished:
                            break
                        yield from bps.checkpoint()
                        if shot_control.pause_count != mark:
                            break  # a deferred pause landed at the checkpoint
                except FailedStatus as exc:
                    failure = exc
                if not finished:
                    end_batch()
                try:
                    yield from bps.mv(shot_control, TriggerState.OFF.value)
                except FailedStatus as exc:
                    # A status of this batch failed between the wait's return
                    # and the mark: the same abandon path.
                    failure = failure or exc
                    finished = False
                    end_batch()
                    yield from bps.mv(shot_control, TriggerState.OFF.value)
            finally:
                # Always released — a stop or abort thrown in while paused
                # included — or a later pause of any run would call this
                # batch's hook and skip its own resume restore.
                shot_control.hold_for_batch(None)
            # The in-flight edge lands.
            yield from bps.sleep(drain)
            if finished:
                if plugin:
                    yield from bps.wait_for([d.truncate_to_quota for d in plugin])
                    yield from bps.collect(*plugin, name=name)
                yield from bps.collect(sampler, name=shots_stream)
                state["steps"] += 1
                return None
            # Paused (or failed) mid-batch: keep the shots every device
            # reached — the frames past them leave the stacks, the rows past
            # them are dropped (none of them went out: a row is collected only
            # once every camera holds its frame) — and record them.
            if plugin:
                yield from bps.wait_for([d.abandon_step for d in plugin])
            kept: dict[str, int] = {}

            async def settle() -> None:
                await sampler.stop()
                shots = sampler.sampled
                if plugin:
                    frames = await asyncio.gather(
                        *(d.frames_this_batch() for d in plugin)
                    )
                    shots = min(shots, *frames)
                await asyncio.gather(*(d.truncate_to(shots) for d in plugin))
                sampler.keep(shots)
                kept["shots"] = shots

            if failure is not None:
                # Record what the batch kept, best effort — the batch's own
                # error is what the operator must see, never a settle
                # timeout on the camera that caused it.
                try:
                    yield from bps.wait_for([settle])
                    if plugin:
                        yield from bps.collect(*plugin, name=name)
                    yield from bps.collect(sampler, name=shots_stream)
                except Exception:  # noqa: BLE001 - the batch's failure wins
                    logger.warning(
                        "gated batch failed; recording its kept shots failed too",
                        exc_info=True,
                    )
                cause = failure.__cause__
                if isinstance(cause, GeecsTriggerTimeoutError):
                    raise cause
                raise GeecsTriggerTimeoutError(
                    getattr(cause, "device_name", None) or "gated batch",
                    shot_timeout,
                    f"gated batch failed: {failure_cause_text(failure)}",
                ) from failure
            yield from bps.wait_for([settle])
            if plugin:
                yield from bps.collect(*plugin, name=name)
            yield from bps.collect(sampler, name=shots_stream)
            done += kept["shots"]
            if done >= quota:
                state["steps"] += 1
                return None
            logger.warning(
                "gated batch paused after %d/%d shot(s) — the step continues "
                "from shot %d",
                done,
                quota,
                done + 1,
            )

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
        # the plan continues the step itself (module docstring, Pause).
        return (yield from name_failed_status(rewindable_wrapper(body, False)))

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

    def one_step(detectors: Sequence[Any], step: Any, pos_cache: Any):
        motors = list(step.keys())
        yield from admit_background(detectors, motors)
        yield from bps.move_per_step(step, pos_cache)
        bins.value += 1
        body = take_reading([*detectors, *motors, bins], shots_per_step)
        return (yield from rewindable_wrapper(body, False))

    def per_step(
        detectors: Sequence[Any], step: Any, pos_cache: Any, take_reading_: Any = None
    ):
        return (yield from name_failed_status(one_step(detectors, step, pos_cache)))

    per_step.__name__ = per_step.__qualname__ = "gated_per_step"
    return per_step


def refuse_free_running_non_essentials(devices: Sequence[Any]) -> None:
    """Refuse a non-essential device with no shot stamp (a free-running one).

    A non-essential device is recorded by stamp — a plugin camera's frames,
    or the :class:`~geecs_bluesky.devices.sampler.StampStream` of a
    triggered device without a plugin — and joined to the rows by it.  A
    scalar-only device publishes no ``acq_timestamp``, so nothing could
    place its readings on a shot: not admitted yet (the free-running case
    is deferred).  Called by the bound plans before anything moves.

    Raises
    ------
    GeecsConfigurationError
        Naming the devices.
    """
    unstamped = [d for d in devices if not isinstance(owner_of(d), GeecsDetector)]
    if unstamped:
        names = ", ".join(getattr(d, "name", str(d)) for d in unstamped)
        raise GeecsConfigurationError(
            f"non-essential device(s) with no shot stamp: {names} — a "
            "non-essential device is recorded by its acq_timestamp and joined "
            "to the shots by it; a free-running device (no stamp) is not "
            "admitted as non-essential yet. Make it essential."
        )


def non_essential_wrapper(plan: Any, flyers: Sequence[Any]) -> Any:
    """Stream the non-essential devices for the run, each into its own ``<name>_stream``.

    Two kinds, one contract — recorded by stamp, joined to the rows by it
    afterwards, never waited on:

    - a **plugin-backed** camera flies itself: its frames and their
      per-frame scalars go to its stack, the stream is datum-only;
    - a triggered device **without** a plugin (a LabVIEW-native saver like
      the HASO, a scalar device with a stamp, or any detector's ``.scalars``
      view) is recorded by a
      :class:`~geecs_bluesky.devices.sampler.StampStream`: one event per
      stamp it publishes, carrying the stamp, its scalars and — a native
      saver listed itself — its ``-nonscalar_save_path``.  A native saver is
      prepared unbounded, so its own lifecycle switches LabVIEW's saving on
      here and off at its ``unstage`` (rule 2; 2a's fly path); a
      ``.scalars`` view is never prepared and writes no files.

    The stock ``fly_during_wrapper`` inserts ``kickoff`` after ``open_run``
    and ``complete`` + ``collect`` before ``close_run`` but neither stages
    nor prepares; this one does both — stage (a device dead at stage time
    fails loudly, as it should), then, right after ``open_run``, the
    unbounded prepares and the stream declarations that route the collects,
    then the kickoffs while the box is still quiet.  Each stream is
    collected alone (one object: no index, the datum covers everything it
    wrote), in its own stream — a joint stream would cut every device at
    the slowest one.  From the run's close on, nothing of a non-essential
    device may fail the item: its ``complete`` + ``collect`` and its
    ``unstage`` are each a contingency, logged and skipped (a gateway that
    went away mid-run), and a stamp stream that recorded nothing is a
    WARNING, never a failure.  A skipped unstage is not retried by the
    RunEngine (the object leaves its staged set before the status
    resolves): a stale ``Capture`` or ``save`` is cleared by the device's
    next ``stage()`` — nothing functional leaks.

    Parameters
    ----------
    plan :
        The bound stock plan.
    flyers :
        The non-essential devices: triggered detectors (with or without a
        file plugin) and ``.scalars`` views of them.  A device with no
        stamp is refused by :func:`refuse_free_running_non_essentials`
        before the run is bound.
    """
    flyers = list(flyers)
    if not flyers:
        return (yield from plan)
    refuse_free_running_non_essentials(flyers)
    # What the run stages: a view's owner (a view has no lifecycle of its own).
    roots: list[Any] = []
    for flyer in flyers:
        root = owner_of(flyer)
        if all(root is not r for r in roots):
            roots.append(root)
    plugin = [f for f in flyers if isinstance(f, GeecsDetector) and f.plugin_backed]
    stamped = [StampStream(f) for f in flyers if f not in plugin]
    # A native saver listed itself (not its view): its LabVIEW files are the
    # record, switched on by its own unbounded prepare.
    native = [
        s.device
        for s in stamped
        if isinstance(s.device, GeecsDetector) and s.device.native_save
    ]
    streams: list[Any] = [*plugin, *stamped]

    def after_open():
        group = short_uid("non-essential-prepare")
        for flyer in [*plugin, *native]:
            yield from bps.prepare(
                flyer, UNBOUNDED_TRIGGER_INFO, group=group, wait=False
            )
        yield from bps.wait(group=group)
        # The plugin's count PV still reads the previous session's total at
        # the arm (GEECS-Plugins#853): zero it inside the fresh session and
        # prepare again, or the kickoff baselines above what the run will
        # write and the close's count wait never returns (2b A4).
        if plugin:
            yield from bps.wait_for([f.zero_count for f in plugin])
            group = short_uid("non-essential-prepare-zeroed")
            for flyer in plugin:
                yield from bps.prepare(
                    flyer, UNBOUNDED_TRIGGER_INFO, group=group, wait=False
                )
            yield from bps.wait(group=group)
        for stream in streams:
            yield from bps.declare_stream(
                stream, name=non_essential_stream(stream.name), collect=True
            )
        yield from bps.kickoff_all(*streams, wait=True)

    def before_close():
        # Nothing waits on a non-essential device — a plugin whose gateway
        # went away mid-run must not fail the run at its close: each
        # stream's complete + collect is its own contingency, logged and
        # skipped.
        for stream in streams:
            # complete and collect are SEPARATE contingencies: a complete
            # that fails (a stalled or dead plugin) must not cost the
            # datums for the frames it did write.
            for verb, plan_factory in (
                ("complete", lambda f=stream: bps.complete(f, wait=True)),
                (
                    "collect",
                    lambda f=stream: bps.collect(f, name=non_essential_stream(f.name)),
                ),
            ):

                def skip(exc, flyer=stream, verb=verb):
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
        for stream in stamped:
            if not stream.published:
                logger.warning(
                    "non-essential %s: published no shot stamp this run — "
                    "%s is empty and its columns read NaN in the s-file",
                    stream.device_name,
                    non_essential_stream(stream.name),
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
        for root in roots:

            def one(root=root):
                # Wait inside the contingency: a failure of the unstage
                # status must land here, not at the next message outside.
                yield from bps.unstage(root, group=short_uid("ne-unstage"), wait=True)

            def skip(exc, root=root):
                logger.warning(
                    "non-essential %s: unstage failed (%s: %s) — skipped; a stale "
                    "Capture or save is cleared by its next stage()",
                    root.name,
                    type(exc).__name__,
                    exc,
                )
                yield from bps.null()

            yield from contingency_wrapper(one(), except_plan=skip, auto_raise=False)

    def staged():
        yield from bps.stage_all(*roots)
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
    "native_essentials",
    "non_essential_wrapper",
    "refuse_free_running_non_essentials",
    "run_bracket",
    "shot_clock",
]
