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
this is the desired, attributable hard failure.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import bluesky.plan_stubs as bps
from bluesky.protocols import Triggerable
from bluesky.utils import short_uid

from geecs_bluesky.exceptions import GeecsQuiescenceTimeoutError

logger = logging.getLogger(__name__)


def geecs_single_shot(
    devices: Sequence[Any],
    fire: Callable,
    name: str = "primary",
):
    """Fire one plan-owned shot and bundle all *devices* into one event.

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

    Yields
    ------
    Bluesky messages.
    """
    grp = short_uid("single_shot")
    triggerables = [obj for obj in devices if isinstance(obj, Triggerable)]
    for obj in triggerables:
        yield from bps.trigger(obj, group=grp, wait=False)
    yield from fire()
    if triggerables:
        yield from bps.wait(group=grp)
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
