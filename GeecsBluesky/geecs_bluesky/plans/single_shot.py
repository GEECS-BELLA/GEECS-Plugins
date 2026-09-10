"""geecs_single_shot — one plan-owned shot: arm waiters, fire, await, read.

The strict-shot-control acquisition primitive.  ``bps.trigger_and_read``
cannot express it because the fire must be injected *between* trigger
initiation and the wait::

    bps.trigger(each detector, wait=False)   # baseline + arm the waiters
    fire()                                   # e.g. DG645 SINGLESHOT state
    bps.wait(group)                          # every detector saw the shot
    create / read / save                     # one complete event row

Ordering is load-bearing: :class:`~geecs_bluesky.devices.ca.triggerable.CaTriggerable`
baselines its ``acq_timestamp`` synchronously inside ``trigger()``, so a shot
fired any time after the trigger messages are processed cannot be missed.
A detector that does not respond to the fired shot raises
:exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError` — in strict mode
this is an attributable failure, recovered by a **bounded refire** (see
:func:`geecs_single_shot`): a missed pulse never yields a frame, so waiting
longer cannot help, but firing again can.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

import bluesky.plan_stubs as bps
from bluesky.protocols import Triggerable
from bluesky.utils import FailedStatus, short_uid

from geecs_bluesky.plans.liveness import rd_confirmed_down
from geecs_bluesky.exceptions import (
    GeecsDeviceDownError,
    GeecsQuiescenceTimeoutError,
    GeecsTriggerTimeoutError,
)

logger = logging.getLogger(__name__)


def _no_frame_timeout(exc: BaseException) -> GeecsTriggerTimeoutError | None:
    """The no-frame timeout behind a wait failure — or ``None`` if it is not one.

    The RunEngine wraps the failing status's own error in
    :exc:`~bluesky.utils.FailedStatus` (``raise FailedStatus(ret) from exc``),
    so a device-attributed :exc:`GeecsTriggerTimeoutError` rides on
    ``__cause__``.

    An attempt creates two kinds of status and only one of them is a frame
    drop: the detectors' ``trigger()`` waits (which fail *only* as a named
    :exc:`GeecsTriggerTimeoutError`) and ``fire()``'s own gateway ``:SP``
    put (a rejected write surfaces as ``aioca.CANothing``, whose text
    carries the PV and the CA message).  Returning ``None`` for the latter
    is what keeps a failed *fire* from being reported — and retried — as a
    camera frame drop.
    """
    cause = exc.__cause__
    return cause if isinstance(cause, GeecsTriggerTimeoutError) else None


def _confirm_device_down(devices: Sequence[Any], device_name: str):
    """Plan: ``True`` iff the named device's gateway ``CONNECTED`` PV says down.

    Distinguishes the two causes of a no-frame timeout: a dropped frame
    (device live — re-firing recovers) vs a device that went down mid-scan
    (its TCP stream to the gateway died — re-firing cannot help).  The
    frameless device is matched by GEECS device name (``_geecs_device_name``,
    falling back to the ophyd name) against *devices*, and its
    ``connected_status`` signal is read via the shared
    :func:`~geecs_bluesky.plans.liveness.rd_confirmed_down` stub.

    **Fail-open**: no matching device, no ``connected_status`` attribute, or
    a failed read (old gateway without status PVs) all return ``False``
    ("not confirmed down"), preserving the refire behavior.  Only the exact
    ``Disconnected`` choice string confirms the device is down — a mock
    backend's ``""`` default reads live.
    """
    device = next(
        (
            obj
            for obj in devices
            if device_name
            in (getattr(obj, "_geecs_device_name", None), getattr(obj, "name", None))
        ),
        None,
    )
    if device is None:
        return False
    # The shared fail-open read (plans/liveness.py) — the one in-plan
    # CONNECTED idiom, so the convention cannot drift between copies.
    confirmed = yield from rd_confirmed_down(device)
    return confirmed


def geecs_single_shot(
    devices: Sequence[Any],
    fire: Callable,
    name: str = "primary",
    max_refires: int = 2,
):
    """Fire one plan-owned shot and bundle all *devices* into one event.

    If a triggered device produces no frame for a fire (the group wait fails
    with :exc:`~bluesky.utils.FailedStatus`), the shot is re-fired up to
    *max_refires* times before the failure propagates.  Strict semantics
    survive refire: a failed attempt records nothing, and the next attempt's
    ``trigger()`` re-baselines and drains any orphan frame
    (:class:`~geecs_bluesky.devices.ca.triggerable.CaTriggerable`), so every
    recorded row is one physical shot.  Refire — not a longer timeout — is
    the recovery because a missed pulse never yields a frame (live-verified;
    numbers in ``GeecsBluesky/CHANGELOG.md`` 0.20.0).

    Refire is gated twice, because only one kind of failure is a frame drop.
    First on **attribution**: the ``try`` spans the whole attempt (trigger +
    ``fire()`` + wait), so a failed ``fire()`` — a rejected gateway ``:SP``
    put — also arrives as ``FailedStatus``.  Only a
    :exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError` cause is a
    missing frame; anything else is logged with its real cause and
    propagates on the first attempt rather than burning the budget against
    a fault refire cannot fix (see :func:`_no_frame_timeout`).  That ERROR
    line is deliberately the record: the ``FailedStatus`` propagates
    unwrapped (the run's stop-document reason is the status repr), so the
    scan log is where the failing PV and its CA message are written down.
    Then on **gateway liveness**: a frameless device whose ``CONNECTED`` PV
    reads Disconnected went down mid-scan, so
    :exc:`~geecs_bluesky.exceptions.GeecsDeviceDownError` is raised instead
    of burning refires; a live or unreadable status (fail-open) keeps the
    bounded-refire behavior.

    Parameters
    ----------
    devices:
        Devices to record.  Triggerable ones are armed before the fire and
        awaited after it; the rest are read into the same event.
    fire:
        Plan-stub callable that emits exactly one trigger (e.g.
        ``lambda: scanner._set_trigger_state("SINGLESHOT")``).
    name:
        Event stream name.
    max_refires:
        Extra fire attempts after the first one fails (default 2, so at most
        three physical fires per recorded shot).  ``0`` restores the old
        hard-fail-on-first-miss behavior.

    Yields
    ------
    Bluesky messages.
    """
    triggerables = [obj for obj in devices if isinstance(obj, Triggerable)]
    attempts = max_refires + 1
    for attempt in range(1, attempts + 1):
        # A fresh, unique group per attempt: statuses from a failed attempt
        # must never be waited on again (a fresh trigger supersedes them).
        grp = short_uid("single_shot")
        try:
            for obj in triggerables:
                yield from bps.trigger(obj, group=grp, wait=False)
            fire_t0 = time.monotonic()
            yield from fire()
            fire_done = time.monotonic()
            if triggerables:
                yield from bps.wait(group=grp)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "shot phases: fire %.1f ms, frame wait %.1f ms (%d triggerables)",
                    (fire_done - fire_t0) * 1e3,
                    (time.monotonic() - fire_done) * 1e3,
                    len(triggerables),
                )
        except FailedStatus as exc:
            # No cancellation of abandoned statuses is needed: the RunEngine
            # stashes a late FailedStatus and throws it into the plan at the
            # next yield, but co-missing devices share the same ~3 s deadline
            # so the stash is consumed right here at the wait; a straggler
            # that lands inside the next attempt is caught by this same try
            # (it wraps the whole attempt: trigger + fire + wait) and merely
            # consumes one refire instead of aborting the scan.  That the try
            # spans the whole attempt is also why the cause must be
            # classified before anything is retried: a fire that never fired
            # lands here too.
            timeout = _no_frame_timeout(exc)
            if timeout is None:
                # Not a missing frame, so a refire cannot help and would only
                # burn the budget against the real fault (live 2026-09-10,
                # Scan033: a rejected SINGLESHOT put failed all three attempts
                # in under a second and the scan died blaming the cameras).
                # Name the actual cause — this line is the only record of
                # which PV failed and why: the FailedStatus that propagates
                # carries the status object's repr, and ``aioca.CANothing``
                # renders the CA message only through ``str`` (its repr is
                # the bare ECA errorcode).  Hence ``%s`` on the cause, with
                # its type spelled out separately so an exception with an
                # empty message still names itself.
                cause = exc.__cause__ or exc
                logger.error(
                    "single-shot attempt %d of %d failed, but not from a "
                    "missing frame — re-firing cannot help, so the failure "
                    "propagates: %s: %s",
                    attempt,
                    attempts,
                    type(cause).__name__,
                    cause,
                )
                raise
            device_name = timeout.device_name
            down = yield from _confirm_device_down(devices, device_name)
            if down:
                raise GeecsDeviceDownError(
                    f"{device_name}: the gateway reports this device "
                    "DISCONNECTED — it went down mid-scan (not a frame "
                    "drop; re-firing cannot help). Check the GEECS device.",
                    device_name=device_name,
                ) from exc
            if attempt == attempts:
                raise
            logger.warning(
                "single-shot attempt %d of %d: no frame from %s — re-firing "
                "(known camera frame-drop intermittency, ~1%% observed)",
                attempt,
                attempts,
                device_name,
            )
        else:
            break
    yield from bps.create(name)
    for obj in devices:
        yield from bps.read(obj)
    yield from bps.save()


def geecs_confirm_quiescent(
    devices: Sequence[Any],
    quiet_s: float = 1.5,
    timeout_s: float = 10.0,
    poll_s: float = 0.2,
):
    """Wait until no sync device's ``acq_timestamp`` advances for *quiet_s*.

    The inverse of :meth:`~geecs_bluesky.devices.ca.triggerable.CaTriggerable.trigger`
    (which waits for an advance): this confirms the free-running trigger has
    *stopped* after the controller was put in single-shot (``ARMED``) mode, so
    plan-owned single-shot firing can begin without mistaking a residual
    free-running shot for the plan's fired shot.

    Watches every device exposing ``last_acq_timestamp`` (the sync devices);
    others are ignored.  ``quiet_s`` should exceed one trigger period so a
    genuine pause is distinguishable from the gap between shots.

    Parameters
    ----------
    devices:
        Devices to watch (non-sync ones without ``last_acq_timestamp`` are
        skipped).
    quiet_s:
        Required span of no advance to declare the trigger stopped.
    timeout_s:
        Give up (raise) if the timestamps keep advancing this long.
    poll_s:
        Poll interval.

    Raises
    ------
    GeecsQuiescenceTimeoutError
        If timestamps keep advancing past *timeout_s*.
    """
    watched = [d for d in devices if hasattr(d, "last_acq_timestamp")]
    if not watched:
        return

    def _snapshot() -> dict[int, Any]:
        return {id(d): d.last_acq_timestamp for d in watched}

    quiet = 0.0
    waited = 0.0
    last = _snapshot()
    while quiet < quiet_s:
        yield from bps.sleep(poll_s)
        waited += poll_s
        now = _snapshot()
        if now == last:
            quiet += poll_s
        else:
            quiet = 0.0
            last = now
            if waited >= timeout_s:
                raise GeecsQuiescenceTimeoutError(timeout_s)
    logger.info("trigger quiescent (%.1fs no advance); ready for single-shot", quiet_s)
