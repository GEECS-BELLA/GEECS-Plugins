"""Compile :class:`~geecs_schemas.action_plan.ActionPlan` into Bluesky plan stubs.

The successor of the legacy ``ActionManager`` executor
(``geecs_scanner.engine.action_manager``): an action plan — set a device
variable, wait, check a readback, run another named plan — becomes a plain
Bluesky message generator, so action sequences inherit the RunEngine's
abort, pause, and document machinery instead of running through a bespoke
executor.

Legacy semantics preserved verbatim
-----------------------------------
- ``set`` — the legacy executor called ``device.set(variable, value,
  sync=step.wait_for_execution)``: a blocking set-and-wait when
  ``wait_for_execution`` is true (the default), fire-and-forget otherwise.
  Here that becomes ``bps.abs_set(signal, value, wait=wait_for_execution)``.
- ``wait`` — legacy ``time.sleep(seconds)`` becomes ``bps.sleep(seconds)``
  (RunEngine-interruptible, no blocking).
- ``check`` — the legacy ``GetStep`` compared ``device.get(variable) ==
  expected`` with plain equality, where the device layer had already coerced
  the wire string to ``float`` when it parses as one
  (``GeecsDevice.interpret_value``).  :func:`values_match` reproduces exactly
  that: coerce the *actual* reading (string → float when parseable), then
  compare with ``==`` — no tolerance.  A mismatch raises
  :class:`~geecs_bluesky.exceptions.ActionCheckFailedError`, matching the
  legacy headless behaviour (auto-abort; the GUI prompt is a front-end
  concern, not an executor one).
- ``run`` — nested plans resolve by name from the registry and recurse,
  like the legacy ``ExecuteStep``.  A missing name raises
  :class:`~geecs_bluesky.exceptions.ActionPlanNotFoundError` (legacy:
  ``ActionError("Action '<name>' is not defined.")``).  Cycles raise
  :class:`~geecs_bluesky.exceptions.ActionPlanCycleError` — an improvement
  over legacy, which had no cycle guard and recursed to the interpreter
  limit.

Purity contract
---------------
This module never touches Channel Access, sessions, or PV strings.  It
receives GEECS device/variable names from the schema and asks the injected
:class:`SettableFactory` for signals; PV derivation is entirely the
factory's business (production factories hand out CA-backed signals whose
``:SP`` puts ride GEECS blocking sets).
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

import bluesky.plan_stubs as bps
from bluesky.protocols import Movable, Readable
from bluesky.utils import MsgGenerator

from geecs_schemas.action_plan import (
    ActionPlan,
    CheckStep,
    RunPlanStep,
    SetStep,
    WaitStep,
)

from geecs_bluesky.exceptions import (
    ActionCheckFailedError,
    ActionPlanCycleError,
    ActionPlanNotFoundError,
)

__all__ = [
    "SettableFactory",
    "compile_action_plan",
    "values_match",
]


@runtime_checkable
class SettableFactory(Protocol):
    """Hands out signals for GEECS ``(device, variable)`` pairs.

    The compiler is deliberately ignorant of how a GEECS name becomes a
    signal: the production implementation returns CA-backed signals against
    the gateway PVs (setpoint puts on ``:SP`` riding GEECS blocking sets),
    while tests return in-memory mock signals.  Factories may raise their own
    errors for unknown devices/variables; the compiler does not second-guess
    them.
    """

    def get_settable(self, device: str, variable: str) -> Movable:
        """Return a settable signal for a ``set`` step.

        Parameters
        ----------
        device : str
            GEECS device name (e.g. ``"U_148_PLC"``).
        variable : str
            Writable variable on that device (e.g. ``"DO.Ch9"``).

        Returns
        -------
        Movable
            An ophyd-async style signal accepted by ``bps.abs_set``.
        """
        ...

    def get_readable(self, device: str, variable: str) -> Readable:
        """Return a readable signal for a ``check`` step.

        Parameters
        ----------
        device : str
            GEECS device name (e.g. ``"U_GaiaSVEReader"``).
        variable : str
            Variable to read on that device (e.g. ``"InternalShutterA"``).

        Returns
        -------
        Readable
            An ophyd-async style signal accepted by ``bps.rd``.
        """
        ...


def values_match(expected: object, actual: object) -> bool:
    """Compare a ``check`` step reading against its expected value, legacy-style.

    The legacy pipeline was: the device layer coerced the wire string to
    ``float`` when it parses as one (``GeecsDevice.interpret_value``:
    ``try: return float(val_string) except: return val_string``), and the
    ActionManager then compared with plain ``==`` — no tolerance, and no
    coercion of the *expected* side.  This reproduces exactly that, including
    its quirks: a numeric expected value matches a device reporting ``"25"``
    (the actual is floated first), but a *string* expected value like ``"25"``
    does **not** match a numeric-looking readback (legacy floated it to
    ``25.0``, and ``25.0 == "25"`` is false).

    Parameters
    ----------
    expected : object
        The ``expected`` value from the plan (``str | float | int``).
    actual : object
        The value read back from the signal.

    Returns
    -------
    bool
        True when the values match under the legacy comparison rules.
    """
    if isinstance(actual, str):
        try:
            actual = float(actual)
        except ValueError:
            pass
    return bool(actual == expected)


def compile_action_plan(
    plan: ActionPlan,
    *,
    registry: Mapping[str, ActionPlan],
    settables: SettableFactory,
) -> MsgGenerator:
    """Compile an :class:`ActionPlan` into a Bluesky plan executing its steps.

    Steps run strictly in order, exactly like the legacy ActionManager:
    ``set`` writes a value (blocking on completion unless the step says
    ``wait_for_execution: false``), ``wait`` sleeps, ``check`` reads a value
    and stops the plan on a mismatch, and ``run`` executes another named
    plan from *registry*.

    Parameters
    ----------
    plan : ActionPlan
        The validated plan to execute.
    registry : Mapping[str, ActionPlan]
        Named plans that ``run`` steps may reference (typically
        ``ActionPlanLibrary.plans``).
    settables : SettableFactory
        Source of signals for ``set`` and ``check`` steps.  The compiler
        passes GEECS device/variable names through untouched; any PV or
        transport concern lives behind this factory.

    Yields
    ------
    Msg
        Bluesky messages for the RunEngine.

    Raises
    ------
    ActionCheckFailedError
        When a ``check`` step reads back a value different from ``expected``.
    ActionPlanNotFoundError
        When a ``run`` step names a plan missing from *registry*.
    ActionPlanCycleError
        When nested ``run`` steps form a loop (legacy had no guard for this;
        it would recurse until Python's recursion limit).
    """
    yield from _compile(plan, registry, settables, stack=())


def _compile(
    plan: ActionPlan,
    registry: Mapping[str, ActionPlan],
    settables: SettableFactory,
    stack: tuple[str, ...],
) -> MsgGenerator:
    """Execute *plan*'s steps; *stack* is the chain of nested plan names.

    Parameters
    ----------
    plan : ActionPlan
        The plan whose steps to execute.
    registry : Mapping[str, ActionPlan]
        Named plans available to ``run`` steps.
    settables : SettableFactory
        Signal source for ``set`` / ``check`` steps.
    stack : tuple of str
        Names of the plans currently executing above this one (cycle guard).

    Yields
    ------
    Msg
        Bluesky messages for the RunEngine.
    """
    for step in plan.steps:
        if isinstance(step, SetStep):
            signal = settables.get_settable(step.device, step.variable)
            # Legacy: device.set(variable, value, sync=wait_for_execution) —
            # sync=True blocks until the device confirms, sync=False is
            # fire-and-forget.  abs_set(wait=...) is the message-level twin.
            yield from bps.abs_set(signal, step.value, wait=step.wait_for_execution)
        elif isinstance(step, WaitStep):
            yield from bps.sleep(step.seconds)
        elif isinstance(step, CheckStep):
            signal = settables.get_readable(step.device, step.variable)
            actual = yield from bps.rd(signal)
            if not values_match(step.expected, actual):
                raise ActionCheckFailedError(
                    device=step.device,
                    variable=step.variable,
                    expected=step.expected,
                    actual=actual,
                )
        elif isinstance(step, RunPlanStep):
            if step.plan in stack:
                raise ActionPlanCycleError([*stack, step.plan])
            nested = registry.get(step.plan)
            if nested is None:
                raise ActionPlanNotFoundError(step.plan, list(registry))
            yield from _compile(nested, registry, settables, (*stack, step.plan))
        else:  # pragma: no cover - schema discriminator prevents this
            raise TypeError(f"Unrecognized action step type: {type(step)!r}")
