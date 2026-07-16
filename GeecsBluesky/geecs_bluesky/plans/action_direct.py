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
:mod:`~geecs_bluesky.plans.action_compiler`, one for one:

- ``set`` — write and block on completion; ``wait_for_execution: false``
  schedules the put and moves on (a late failure is logged, never raised).
- ``wait`` — sleep, in small slices so an abort lands promptly.
- ``check`` — read back and compare via
  :func:`~geecs_bluesky.plans.action_compiler.values_match` (quirks
  preserved); a mismatch raises
  :class:`~geecs_bluesky.exceptions.ActionCheckFailedError`.

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

from geecs_bluesky.exceptions import ActionCheckFailedError
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

#: Wait-step slice so an abort interrupts a long ``wait`` promptly.
_WAIT_SLICE_S = 0.2


class ActionExecutionAborted(RuntimeError):
    """The abort probe tripped while action steps were executing."""


class ActionStepTimeoutError(RuntimeError):
    """A blocking step dispatch did not complete within its budget."""


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


def _dispatch(
    coro: Any, loop: asyncio.AbstractEventLoop, label: str
) -> concurrent.futures.Future:
    """Schedule *coro* onto *loop*; the loop must be running (paused RE)."""
    if not loop.is_running():
        coro.close()
        raise RuntimeError(
            f"cannot execute action step {label}: the RunEngine's event "
            "loop is not running"
        )
    return asyncio.run_coroutine_threadsafe(coro, loop)


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
        Optional probe consulted before every step and between wait
        slices; ``True`` raises :class:`ActionExecutionAborted` promptly.
    step_timeout_s :
        Budget for each blocking ``set``/``check`` dispatch.

    Raises
    ------
    ActionExecutionAborted
        The abort probe tripped.
    ActionCheckFailedError
        A ``check`` step read back a value different from ``expected``.
    ActionStepTimeoutError
        A blocking dispatch exceeded *step_timeout_s*.
    """

    def _abort_check() -> None:
        if should_abort is not None and should_abort():
            raise ActionExecutionAborted("action execution aborted by operator")

    def _result(future: concurrent.futures.Future, label: str) -> Any:
        try:
            return future.result(timeout=step_timeout_s)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise ActionStepTimeoutError(
                f"action step {label} did not complete within {step_timeout_s:.0f} s"
            ) from None

    for step, _origin in steps:
        _abort_check()
        if isinstance(step, SetStep):
            label = f"set {step.device}:{step.variable}"
            signal = settables.get_settable(step.device, step.variable)
            future = _dispatch(_set_in_loop(signal, step.value), loop, label)
            if step.wait_for_execution:
                _result(future, label)
            else:
                # Legacy fire-and-continue: a late failure is logged, never
                # raised into the step sequence (matches bps.abs_set(wait=False),
                # whose status the plan never awaited either).
                future.add_done_callback(
                    lambda f, lbl=label: (
                        logger.warning("action %s failed late: %s", lbl, f.exception())
                        if not f.cancelled() and f.exception() is not None
                        else None
                    )
                )
        elif isinstance(step, WaitStep):
            deadline = time.monotonic() + step.seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(_WAIT_SLICE_S, remaining))
                _abort_check()
        elif isinstance(step, CheckStep):
            label = f"check {step.device}:{step.variable}"
            signal = settables.get_readable(step.device, step.variable)
            actual = _result(_dispatch(_read_in_loop(signal), loop, label), label)
            if not values_match(step.expected, actual):
                raise ActionCheckFailedError(
                    device=step.device,
                    variable=step.variable,
                    expected=step.expected,
                    actual=actual,
                )
        else:  # pragma: no cover - flatten only emits the three concrete kinds
            raise TypeError(f"Unrecognized action step type: {type(step)!r}")
