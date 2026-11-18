"""The shot gate in the plans: connect before the claim, hold before every fire.

:class:`~geecs_bluesky.shot_gate.ShotGate` is the client; here are
the plan stubs every firing plan shares:

- :func:`open_gate` — before the box is driven and before the claim: start
  the reader and wait (bounded) for a connection and its first frame.  An unreachable gate
  refuses the run with nothing claimed, the liveness gate's posture.
- :func:`hold_for_permission` — before a fire's baseline: wait for one
  fresh verdict; a hold longer than the gate's ``hold_pause_s`` pauses
  the run with the gate's reason, and the operator's resume waits again.
  A pause never fires: permission is always checked again after one.
- :func:`fire_gated` — the last check and the SINGLESHOT put, the fire
  marked at dispatch; a lapsed permission fires nothing, and
  :func:`pause_on_lapses` pauses a shot whose permission keeps lapsing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import bluesky.plan_stubs as bps

from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.shot_gate import CONNECT_TIMEOUT_S, ShotGate
from geecs_bluesky.exceptions import GeecsShotGateError

logger = logging.getLogger(__name__)


def open_gate(gate: ShotGate | None, timeout: float = CONNECT_TIMEOUT_S):
    """Plan: the gate's reader running and connected, else refuse the run."""
    if gate is None:
        return
    yield from bps.wait_for([gate.start])
    out: dict[str, bool] = {}

    async def connect() -> None:
        out["ok"] = await gate.wait_connected(timeout)

    yield from bps.wait_for([connect])
    if not out.get("ok"):
        raise GeecsShotGateError(
            f"shot gate {gate.address} did not answer within {timeout:g} s "
            f"({gate.reason}) — no shot can be permitted; refusing the run "
            "before anything moved"
        )


def hold_for_permission(gate: ShotGate | None, pause: Any = None):
    """Plan: wait for the gate's permission; return the seconds spent waiting.

    A hold pauses the run only when ``hold_pause_s`` has passed without
    permission; a verdict that lapses as the wait returns waits again.
    The seconds returned are the waiting alone, never a pause's.

    Parameters
    ----------
    gate :
        The worker's gate; ``None`` returns 0 at once.
    pause :
        Plan stub taking the reason, issued when a hold outlasts
        ``hold_pause_s`` (default: log at ERROR and ``bps.pause()``).
    """
    if gate is None or gate.permits():
        return 0.0
    logger.info("shot held by the gate: %s", gate.reason)
    held = 0.0
    while True:
        # One hold of up to hold_pause_s, waited in pieces: a permission that
        # lapses before the plan sees it is waited for again in the same hold.
        deadline = time.monotonic() + gate.hold_pause_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            t0 = time.monotonic()

            async def wait(remaining: float = remaining) -> None:
                await gate.wait_safe(remaining)

            yield from bps.wait_for([wait])
            held += time.monotonic() - t0
            if gate.permits():
                logger.info("shot gate: safe after %.1f s", held)
                return held
        yield from _pause_run(
            f"shot gate {gate.address}: no permission to fire for "
            f"{gate.hold_pause_s:g} s — {gate.reason or 'not safe'}; "
            "resume to wait for it again",
            pause,
        )


class LapseClock:
    """When one shot's permission first lapsed before its fire, for :func:`pause_on_lapses`.

    A permission that lapses between the hold and the fire sends the caller
    back to the hold, which starts its own deadline, so a verdict that keeps
    flickering would never fire and never pause.  The clock starts at the
    shot's first lapse and starts again after any pause of the run
    (*pauses*: the box's ``pause_count``), so neither a hold before the
    first lapse nor the operator's pause counts.  One clock per shot.
    """

    def __init__(self, pauses: Callable[[], int]) -> None:
        self._pauses = pauses
        self._seen = -1
        self.since: float | None = None

    def lapsed(self) -> float:
        """Record a lapse; when this shot's lapses began (since the last pause)."""
        seen = self._pauses()
        if self.since is None or seen != self._seen:
            self.since, self._seen = time.monotonic(), seen
        return self.since

    def reset(self) -> None:
        """Start again: after a pause, or a shot that fired."""
        self.since = None


def pause_on_lapses(gate: ShotGate | None, clock: LapseClock, pause: Any = None):
    """Plan: record a lapse before the fire; pause once they have lasted ``hold_pause_s``."""
    if gate is None or time.monotonic() - clock.lapsed() < gate.hold_pause_s:
        return
    yield from _pause_run(
        f"shot gate {gate.address}: no fire for {gate.hold_pause_s:g} s since "
        "the shot's permission first lapsed before its fire — "
        f"{gate.reason or 'not safe'}; resume to try again",
        pause,
    )
    clock.reset()


def _pause_run(reason: str, pause: Any):
    if pause is None:
        logger.error("pausing the run: %s", reason)
        yield from bps.pause()
    else:
        yield from pause(reason)


def fire_gated(shot_control: Any, gate: ShotGate | None):
    """Plan: the SINGLESHOT put, if the gate still permits it; whether it fired.

    The last permission check sits here, with no await between it and the
    put, so a verdict that lapsed while the caller announced or armed stops
    the fire: nothing is put and ``False`` is returned, and the caller holds
    again.  The fire is marked before the put, not after it: the put returns
    only once the box acknowledges, and a verdict taken during that wait
    must count.  A failed put marks it again — it may have fired.
    """
    if gate is not None:
        if not gate.permits():
            logger.info(
                "shot gate: permission lapsed before the fire (%s) — not fired",
                gate.reason,
            )
            return False
        gate.fired()
    try:
        yield from bps.mv(shot_control, TriggerState.SINGLESHOT.value)
    except BaseException:
        if gate is not None:
            gate.fired()
        raise
    return True


__all__ = [
    "LapseClock",
    "fire_gated",
    "hold_for_permission",
    "open_gate",
    "pause_on_lapses",
]
