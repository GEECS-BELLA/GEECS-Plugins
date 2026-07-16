"""Direct ActionPlan step executor — runs steps while the RunEngine is paused.

``session.run_action`` executes compiled steps *as a plan* on the RunEngine
— impossible while the RE is paused holding a scan (G-actions v2, issue
#552).  This module is the other dispatch over the same flattened steps:
:func:`execute_action_steps_directly` walks the output of
:func:`~geecs_bluesky.plans.action_compiler.flatten_action_steps` and
performs each step by scheduling a coroutine onto the RE's **persistent
asyncio loop** (alive while the RE is paused) via
``asyncio.run_coroutine_threadsafe`` — the same dispatch pattern the
session uses for device connect/disconnect.  Ophyd-async signal ``set()``
calls are made *inside* the loop (an ``AsyncStatus`` must be created where
the loop runs), never on the calling thread.

Step semantics are the legacy ActionManager semantics pinned in
:mod:`~geecs_bluesky.plans.action_compiler`, one for one — including the
RunEngine's actual treatment of fire-and-forget puts:

- ``set`` — write and block on completion.  ``wait_for_execution: false``
  schedules the put and moves on, but **a put that fails while later
  steps are still executing aborts the sequence** — exact parity with the
  RunEngine, whose ``_status_object_completed`` raises ``FailedStatus``
  into a still-running plan for *any* failed set status, grouped or not;
  only failures landing after the sequence has finished are pardoned
  (logged, never raised), again matching the RE's end-of-plan
  ``_pardon_failures``.
- ``wait`` — sleep, in small slices so an abort lands promptly.
- ``check`` — read back and compare via
  :func:`~geecs_bluesky.plans.action_compiler.values_match` (quirks
  preserved); a mismatch raises
  :class:`~geecs_bluesky.exceptions.ActionCheckFailedError`.

Blocking dispatches are waited in slices too, so the abort probe can
interrupt even a wedged CA put promptly (the in-loop coroutine is
cancelled — ``Task.cancel`` propagates into the awaited status), and the
loop's liveness is re-checked each slice so a torn-down RE is reported as
such instead of burning the step budget.

No compile logic lives here — nested ``run`` references were already
inlined by the flatten walk, and signals come from the same injected
factory (pre-connected by the caller, exactly as ``run_action`` does).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import time
from typing import Any, Callable, Sequence

from geecs_schemas.action_plan import CheckStep, SetStep, WaitStep

from geecs_bluesky.exceptions import ActionCheckFailedError, ActionStepTimeoutError
from geecs_bluesky.plans.action_compiler import SettableFactory, values_match

logger = logging.getLogger(__name__)

__all__ = [
    "ActionExecutionAborted",
    "ActionStepTimeoutError",
    "execute_action_steps_directly",
]

#: Per-step budget for a blocking ``set`` / ``check`` dispatch.  Slightly
#: above the action signals' own 30 s put budget so the signal's richer
#: timeout error surfaces first when the put itself is stuck.
_STEP_TIMEOUT_S = 45.0

#: Slice for every wait (sleep steps AND blocking dispatches) so the abort
#: probe, pending-put failures, and loop liveness are consulted promptly.
_WAIT_SLICE_S = 0.2


class ActionExecutionAborted(RuntimeError):
    """The abort probe tripped while action steps were executing.

    Deliberately *not* a :class:`~geecs_bluesky.exceptions.GeecsError`:
    this is operator-initiated control flow (the analogue of bluesky's
    ``RequestAbort``), not a failure to surface.
    """


def execute_action_steps_directly(
    steps: Sequence[tuple[SetStep | WaitStep | CheckStep, str | None]],
    settables: SettableFactory,
    loop: asyncio.AbstractEventLoop,
    *,
    should_abort: Callable[[], bool] | None = None,
    step_timeout_s: float = _STEP_TIMEOUT_S,
) -> None:
    """Execute flattened action steps without the RunEngine.

    Parameters
    ----------
    steps :
        The output of
        :func:`~geecs_bluesky.plans.action_compiler.flatten_action_steps`
        — concrete ``set``/``wait``/``check`` steps in execution order,
        nested plans already inlined.
    settables :
        The signal factory; every signal a step touches must already be
        connected (the caller runs the same pre-connect walk as
        ``run_action`` — a lazy connect inside the RE loop would deadlock).
    loop :
        The RunEngine's persistent asyncio loop (running — the RE keeps it
        alive while paused).
    should_abort :
        Optional probe consulted before every step, between wait slices,
        and between blocking-dispatch slices; ``True`` raises
        :class:`ActionExecutionAborted` promptly (an in-flight in-loop
        coroutine is cancelled).
    step_timeout_s :
        Budget for each blocking ``set``/``check`` dispatch.

    Raises
    ------
    ActionExecutionAborted
        The abort probe tripped.
    ActionCheckFailedError
        A ``check`` step read back a value different from ``expected``.
    ActionStepTimeoutError
        A blocking dispatch was still pending at the end of its budget.
    RuntimeError
        The RunEngine's loop is not running (refused loudly, never a
        silent hang).
    Exception
        A failed ``wait_for_execution: false`` put whose failure landed
        while the sequence was still executing re-raises the put's own
        error (RunEngine ``FailedStatus`` parity).
    """
    #: Fire-and-forget puts still in flight: (future, label).  Consulted at
    #: every slice point; a failure here aborts the sequence (RE parity).
    pending: list[tuple[concurrent.futures.Future, str]] = []

    def _abort_check() -> None:
        if should_abort is not None and should_abort():
            raise ActionExecutionAborted("action execution aborted by operator")

    def _loop_check(label: str) -> None:
        if not loop.is_running():
            raise RuntimeError(
                f"cannot execute action step {label}: the RunEngine's event "
                "loop is not running"
            )

    def _reap_pending() -> None:
        """Drop finished fire-and-forget puts; a failed one aborts (RE parity)."""
        for future, label in list(pending):
            if not future.done():
                continue
            pending.remove((future, label))
            exc = None if future.cancelled() else future.exception()
            if exc is not None:
                logger.error("action %s failed while the sequence ran: %s", label, exc)
                raise exc

    def _dispatch(coro: Any, label: str) -> concurrent.futures.Future:
        _loop_check(label)
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def _result(future: concurrent.futures.Future, label: str) -> Any:
        """Sliced blocking wait: abort/pending/loop checks every slice."""
        deadline = time.monotonic() + step_timeout_s
        while True:
            try:
                return future.result(timeout=_WAIT_SLICE_S)
            except TimeoutError:
                if future.done():
                    # The step ITSELF raised a TimeoutError-family error —
                    # that is the signal's own richer failure, not budget
                    # expiry; never re-label it.
                    raise
            try:
                _abort_check()
                _reap_pending()
                _loop_check(label)
            except BaseException:
                future.cancel()  # propagates into the in-loop coroutine
                raise
            if time.monotonic() > deadline:
                future.cancel()
                raise ActionStepTimeoutError(label, step_timeout_s)

    for step, _origin in steps:
        _abort_check()
        _reap_pending()
        if isinstance(step, SetStep):
            label = f"set {step.device}:{step.variable}"
            signal = settables.get_settable(step.device, step.variable)
            future = _dispatch(_set_in_loop(signal, step.value), label)
            if step.wait_for_execution:
                _result(future, label)
            else:
                pending.append((future, label))
        elif isinstance(step, WaitStep):
            deadline = time.monotonic() + step.seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(_WAIT_SLICE_S, remaining))
                _abort_check()
                _reap_pending()
        elif isinstance(step, CheckStep):
            label = f"check {step.device}:{step.variable}"
            signal = settables.get_readable(step.device, step.variable)
            actual = _result(_dispatch(_read_in_loop(signal), label), label)
            if not values_match(step.expected, actual):
                raise ActionCheckFailedError(
                    device=step.device,
                    variable=step.variable,
                    expected=step.expected,
                    actual=actual,
                )
        else:  # pragma: no cover - flatten only emits the three concrete kinds
            raise TypeError(f"Unrecognized action step type: {type(step)!r}")

    # Sequence complete: puts still in flight are pardoned from here on —
    # a late failure is logged, never raised (RunEngine end-of-plan parity,
    # where _pardon_failures is set and outstanding statuses are not waited).
    for future, label in pending:
        future.add_done_callback(
            lambda f, lbl=label: (
                logger.warning("action %s failed late: %s", lbl, f.exception())
                if not f.cancelled() and f.exception() is not None
                else None
            )
        )


async def _await_maybe(value: Any) -> Any:
    """Await *value* if it is awaitable (mock factories may be sync)."""
    if inspect.isawaitable(value):
        return await value
    return value


async def _set_in_loop(signal: Any, value: Any) -> None:
    """Call ``signal.set`` on the loop and await its status there."""
    await _await_maybe(signal.set(value))


async def _read_in_loop(signal: Any) -> Any:
    """Read one value from *signal* on the loop (``get_value`` preferred)."""
    getter = getattr(signal, "get_value", None)
    if callable(getter):
        return await _await_maybe(getter())
    reading = await _await_maybe(signal.read())
    (single,) = reading.values()
    return single["value"]
