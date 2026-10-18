"""The one budget rule for the verbs that block, in every domain.

Sized to the MCP client, not to the scan (the OSPREY team's answer,
2026-10-18).  Claude Code moves an interactive call still running after
**2 minutes** to a background task — counted from the call's start,
whatever it sends — drops an HTTP server's call after 5 minutes without
traffic, and runs a first-byte timer on HTTP servers.  So:

- a whole call is held under ``CALL_BUDGET_S`` of wall clock
  (:class:`CallBudget` bounds every HTTP read inside it), and a wait's
  own polling under ``MAX_WAIT_S``, leaving room for its trailing reads;
  a wait that ends is ``finished: false`` and the skill calls again;
- every wait poll and every scanner verb reports MCP progress
  (:func:`heartbeat`, :func:`keepalive`), which resets the **idle** timer
  — not the 2-minute one, which only the budget above respects.  Whether
  progress also satisfies the first-byte timer depends on the client
  sending a progress token over its transport; a client that sends none
  gets no notifications.
"""

from __future__ import annotations

import builtins
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

import anyio

from geecs_mcp import clients

#: Wall clock one blocking call may take, all reads included — under the
#: client's 2-minute auto-background, with margin for connects.
CALL_BUDGET_S = 110.0
#: A wait verb's polling budget (default and cap): CALL_BUDGET_S less
#: room for the reads that follow the last poll.
DEFAULT_WAIT_S = 90.0
MAX_WAIT_S = 90.0
#: How often a blocking service call reports that it is still running.
HEARTBEAT_S = 10.0
#: The shortest read a budget hands out, so a late call fails as a
#: timeout rather than never being sent.
MIN_READ_S = 1.0

T = TypeVar("T")


#: What a bounded result says when the call's own clock ran out.
BUDGET_SPENT = (
    "this call's wall-clock budget ran out before the service answered "
    "(a slow or stalled peer); nothing was cancelled on the service — call again"
)


class CallBudget:
    """One blocking call's wall clock.

    :meth:`scope` is the guarantee — an absolute cancel scope around the
    whole operation, every await inside it included; :meth:`read_timeout`
    hands each HTTP request what is left, so a request inside the scope
    times out on its own no later than the scope does.
    """

    def __init__(self, seconds: Optional[float] = None) -> None:
        budget = CALL_BUDGET_S if seconds is None else seconds
        self._end = time.monotonic() + budget

    def left(self) -> float:
        """Seconds remaining (never negative)."""
        return max(0.0, self._end - time.monotonic())

    def read_timeout(self) -> float:
        """The read budget for the next request: the client default, cut to what is left."""
        return max(MIN_READ_S, min(clients.READ_TIMEOUT_S, self.left()))

    def scope(self) -> anyio.CancelScope:
        """A cancel scope that ends the operation when the budget is spent.

        Leaving it by expiry is silent (``cancelled_caught``): the caller
        answers with its bounded result after the ``with`` block.
        """
        return anyio.move_on_after(self.left())


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
