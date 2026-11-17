"""The shot gate in the plans: connect before the claim, hold before every fire.

:class:`~geecs_bluesky.shot_gate.ShotGate` is the client; here are
the two plan stubs every firing plan shares:

- :func:`open_gate` — before the box is driven and before the claim: start
  the reader and wait (bounded) for a connection.  An unreachable gate
  refuses the run with nothing claimed, the liveness gate's posture.
- :func:`hold_for_permission` — before a fire's baseline: wait for one
  fresh ``safe``; a hold longer than the gate's ``hold_pause_s`` pauses
  the run with the gate's reason, and the operator's resume waits again.
  A pause never fires: permission is always checked again after one.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import bluesky.plan_stubs as bps

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
    """Plan: wait for the gate's permission; return the seconds held.

    Parameters
    ----------
    gate :
        The worker's gate; ``None`` returns 0 at once.
    pause :
        Plan stub taking the reason, issued when a hold outlasts
        ``hold_pause_s`` (default: log at ERROR and ``bps.pause()``).
    """
    if gate is None:
        return 0.0
    t0 = time.monotonic()
    if gate.permits():
        return 0.0
    logger.info("shot held by the gate: %s", gate.reason)
    while True:
        out: dict[str, bool] = {}

        async def wait(out: dict[str, bool] = out) -> None:
            out["ok"] = await gate.wait_safe(gate.hold_pause_s)

        yield from bps.wait_for([wait])
        if out.get("ok") and gate.permits():
            held = time.monotonic() - t0
            logger.info("shot gate: safe after %.1f s", held)
            return held
        reason = (
            f"shot gate {gate.address}: no permission to fire for "
            f"{gate.hold_pause_s:g} s — {gate.reason or 'not safe'}; "
            "resume to wait for it again"
        )
        if pause is None:
            logger.error("pausing the run: %s", reason)
            yield from bps.pause()
        else:
            yield from pause(reason)


__all__ = ["hold_for_permission", "open_gate"]
