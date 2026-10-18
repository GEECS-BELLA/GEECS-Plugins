"""The one budget rule for the verbs that block, in every domain.

Sized to the MCP client, not to the scan (the OSPREY team's answer,
2026-10-18): Claude Code moves an interactive call still running after
2 minutes to a background task, drops an HTTP server's call after 5
minutes without traffic, and starts a first-byte timer on HTTP servers.
So a wait is short and repeatable — ``DEFAULT_WAIT_S`` by default,
``MAX_WAIT_S`` at most, ``finished: false`` when it ends, call again —
and every blocking call reports MCP progress while it runs
(:func:`heartbeat`, :func:`keepalive`), which is what resets those
timers.  A client that sends no progress token gets no notifications;
nothing else changes.
"""

from __future__ import annotations

import builtins
from typing import Any, Awaitable, Callable, Optional, TypeVar

import anyio

#: A wait verb's default budget, and the cap on any one call — under the
#: client's 2-minute auto-background, so a call answers in the turn.
DEFAULT_WAIT_S = 90.0
MAX_WAIT_S = 110.0
#: How often a blocking service call reports that it is still running.
HEARTBEAT_S = 10.0

T = TypeVar("T")


def bounded_wait(timeout_s: Any) -> float:
    """``timeout_s`` as a float clamped to ``[0, MAX_WAIT_S]``; garbage is 0."""
    try:
        value = float(timeout_s)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(value, MAX_WAIT_S))


async def heartbeat(elapsed_s: float, total_s: Optional[float], message: str) -> None:
    """Report progress on the current MCP request, if there is one; never raises."""
    from fastmcp.server.dependencies import get_context

    try:
        ctx = get_context()
    except RuntimeError:  # not inside a tool call (the impls' own tests)
        return
    try:
        await ctx.report_progress(progress=elapsed_s, total=total_s, message=message)
    except (anyio.ClosedResourceError, anyio.BrokenResourceError):
        return  # the client went away; the verb's own answer still stands


async def keepalive(
    call: Callable[[], Awaitable[T]], *, message: str, every_s: float = HEARTBEAT_S
) -> T:
    """Await ``call()``, sending a :func:`heartbeat` every ``every_s`` until it returns.

    The call's own exception is re-raised as itself — anyio's task group
    would wrap it in an exception group, which the guard does not map.
    """

    async def beats() -> None:
        elapsed = 0.0
        while True:
            await anyio.sleep(every_s)
            elapsed += every_s
            await heartbeat(elapsed, None, message)

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(beats)
            try:
                return await call()
            finally:
                tg.cancel_scope.cancel()
    except builtins.BaseExceptionGroup as group:  # 3.11+, the package's floor
        if len(group.exceptions) == 1:
            raise group.exceptions[0] from None
        raise
