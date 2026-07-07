"""geecs_single_shot — one plan-owned shot: arm waiters, fire, await, read.

The strict-shot-control acquisition primitive.  ``bps.trigger_and_read``
cannot express it because the fire must be injected *between* trigger
initiation and the wait::

    bps.trigger(each detector, wait=False)   # baseline + arm the waiters
    fire()                                   # e.g. DG645 SINGLESHOT state
    bps.wait(group)                          # every detector saw the shot
    create / read / save                     # one complete event row

Ordering is load-bearing: :class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable`
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
from typing import Any, Callable, Sequence

import bluesky.plan_stubs as bps
from bluesky.protocols import Triggerable
from bluesky.utils import FailedStatus, short_uid

from geecs_bluesky.devices.ca._pv import GATEWAY_DISCONNECTED
from geecs_bluesky.exceptions import (
    GeecsDeviceDownError,
    GeecsQuiescenceTimeoutError,
    GeecsTriggerTimeoutError,
)

logger = logging.getLogger(__name__)


def _no_frame_device(exc: BaseException) -> str:
    """Name the device that produced no frame, from a wait failure.

    The RunEngine wraps the status's own error in
    :exc:`~bluesky.utils.FailedStatus` (``raise FailedStatus(ret) from exc``),
    so the device-attributed :exc:`GeecsTriggerTimeoutError` rides on
    ``__cause__``.
    """
    cause = exc.__cause__
    if isinstance(cause, GeecsTriggerTimeoutError):
        return cause.device_name
    return "unknown device"


def _confirm_device_down(devices: Sequence[Any], device_name: str):
    """Plan: ``True`` iff the named device's gateway ``CONNECTED`` PV says down.

    Distinguishes the two causes of a no-frame timeout: a dropped frame
    (device live — re-firing recovers) vs a device that went down mid-scan
    (its TCP stream to the gateway died — re-firing cannot help).  The
    frameless device is matched by GEECS device name (``_geecs_device_name``,
    falling back to the ophyd name) against *devices*, and its
    ``connected_status`` signal is read in plan context via ``bps.rd``.

    **Fail-open**: no matching device, no ``connected_status`` attribute, or
    a failed read (old gateway without status PVs) all return ``False``
    ("not confirmed down"), preserving the refire behavior.  Only the exact
    :data:`~geecs_bluesky.devices.ca._pv.GATEWAY_DISCONNECTED` choice string
    confirms the device is down — a mock backend's ``""`` default reads live.
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
    signal = getattr(device, "connected_status", None)
    if signal is None:
        return False
    try:
        value = yield from bps.rd(signal)
    except Exception:
        logger.debug(
            "CONNECTED read failed for %s; assuming live (fail-open)",
            device_name,
            exc_info=True,
        )
        return False
    return value == GATEWAY_DISCONNECTED


def geecs_single_shot(
    devices: Sequence[Any],
    fire: Callable,
    name: str = "primary",
    max_refires: int = 2,
):
    """Fire one plan-owned shot and bundle all *devices* into one event.

    If a triggered device produces no frame for a fire (the group wait fails
    with :exc:`~bluesky.utils.FailedStatus`), the shot is re-fired up to
    *max_refires* times before the failure propagates.

    **Why refire preserves strict semantics.**  A failed attempt records
    nothing: ``create``/``read``/``save`` run only after the group wait
    succeeds, so no event row exists for the failed fire.  A device that
    *did* capture the failed fire holds an orphan frame, but the next
    attempt's ``trigger()`` re-baselines against it and drains the shot queue
    synchronously (:class:`~geecs_bluesky.devices.ca.triggerable.CaTriggerable`'s
    warm-path drain), so its fresh status only completes on a frame from the
    new fire.  Every recorded row therefore remains one physical shot across
    all devices.

    **Why refire (not a longer timeout).**  Live evidence, 2026-07-06
    Undulator Scan011 (first GUI optimization campaign): 84 SINGLESHOT fires
    produced 83 camera frames; fire→frame offset over the 83 good shots was
    median 0.21 s (max 1.06 s) against the 3.0 s trigger timeout — ample
    margin.  The one unmatched fire produced no frame *ever* (the Basler's
    known ~1% frame-drop intermittency) and aborted the whole campaign.  At a
    ~1.2% drop rate a 100-shot campaign aborts with ~70% probability without
    refire; a missed pulse never yields a frame, so only re-firing recovers.

    **Refire is gated on gateway liveness.**  Before re-firing, the frameless
    device's ``connected_status`` (the gateway ``CONNECTED`` PV) is read: if
    the device reports Disconnected, the timeout was not a frame drop — the
    device went down mid-scan — and :exc:`~geecs_bluesky.exceptions.GeecsDeviceDownError`
    is raised immediately instead of burning refires against dead hardware.
    A live (or unreadable — fail-open) status keeps the bounded-refire
    behavior.

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
            yield from fire()
            if triggerables:
                yield from bps.wait(group=grp)
        except FailedStatus as exc:
            # The RunEngine (bluesky 1.15.1, run_engine.py) reports a failed
            # status via _status_object_completed:
            #
            #     raise FailedStatus(ret) from exc
            #   except Exception as e:
            #       self._exception = e
            #       fut.set_exception(e)
            #       ...
            #       fut.exception()   # squash "never retrieved" at teardown
            #
            # and the run loop throws that stashed exception into the plan at
            # the next message boundary:
            #
            #     if self._exception is not None:
            #         stashed_exception = self._exception
            #         self._exception = None
            #     ...
            #     msg = self._plan_stack[-1].throw(stashed_exception or resp)
            #
            # Consequences for statuses abandoned by a failed attempt:
            # - A pending status that later *succeeds* (its device did frame)
            #   just resolves its uncollected future — completely inert.
            # - A pending status that later *errors* (a second device also
            #   missed) is NOT mere log noise: its FailedStatus lands in
            #   RE._exception and is thrown at the plan's next yield.  In
            #   practice all of an attempt's statuses share the same ~3 s
            #   deadline (triggered within milliseconds of each other), so
            #   co-missing devices error effectively together and the single
            #   RE._exception slot is consumed right here at the wait (later
            #   stashes overwrite earlier ones; the prefetched fut.exception()
            #   keeps teardown quiet).  If a straggler does land inside the
            #   next attempt, this try wraps the *whole* attempt (trigger +
            #   fire + wait), so the stale FailedStatus is caught and merely
            #   consumes one refire instead of aborting the scan — no
            #   cancellation machinery needed.
            #
            # Refire is gated on gateway liveness: a no-frame timeout from a
            # device whose CONNECTED PV reports Disconnected is not a frame
            # drop — the device went down mid-scan, and re-firing burns
            # attempts (~3 s each) against hardware that cannot answer.
            device_name = _no_frame_device(exc)
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

    The inverse of :meth:`~geecs_bluesky.devices.triggerable.GeecsTriggerable.trigger`
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
