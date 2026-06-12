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

from typing import Any, Callable, Sequence

import bluesky.plan_stubs as bps
from bluesky.protocols import Triggerable
from bluesky.utils import short_uid


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
