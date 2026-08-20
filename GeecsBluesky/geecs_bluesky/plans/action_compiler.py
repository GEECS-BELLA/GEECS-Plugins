"""Compile :class:`~geecs_schemas.action_plan.ActionPlan` into Bluesky plan stubs.

An action plan — set a device variable, wait, check a readback, run another
named plan — becomes a plain Bluesky message generator, so action sequences
inherit the RunEngine's abort, pause, and document machinery.

Legacy ``ActionManager`` semantics are pinned here (the one canonical
statement):

- ``set`` → ``bps.abs_set(signal, value, wait=wait_for_execution)``
  (blocking set-and-wait by default, fire-and-forget otherwise).
- ``wait`` → ``bps.sleep(seconds)`` (RunEngine-interruptible).
- ``check`` → :func:`values_match`: float-coerce the *actual* reading when
  it parses, then plain ``==`` — no tolerance, no coercion of the expected
  side.  A mismatch raises
  :class:`~geecs_bluesky.exceptions.ActionCheckFailedError` (legacy
  headless auto-abort; the GUI prompt is a front-end concern).
- ``run`` → resolve by name from the registry and recurse.  A missing name
  raises :class:`~geecs_bluesky.exceptions.ActionPlanNotFoundError`; cycles
  raise :class:`~geecs_bluesky.exceptions.ActionPlanCycleError` (legacy had
  no cycle guard).

Purity contract: this module never touches Channel Access, sessions, or PV
strings — signals come from the injected :class:`SettableFactory`.
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
    "flatten_action_steps",
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
        """Return a settable signal (accepted by ``bps.abs_set``) for a ``set`` step."""
        ...

    def get_readable(self, device: str, variable: str) -> Readable:
        """Return a readable signal (accepted by ``bps.rd``) for a ``check`` step."""
        ...


def values_match(expected: object, actual: object) -> bool:
    """Compare a ``check`` step reading against its expected value, legacy-style.

    Legacy comparison rules (see the module docstring), quirks included: a
    numeric expected matches a device reporting ``"25"`` (the actual is
    floated first), but a *string* expected ``"25"`` does **not** match a
    numeric-looking readback.

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


def flatten_action_steps(
    plan: ActionPlan,
    *,
    registry: Mapping[str, ActionPlan],
) -> list[tuple[SetStep | WaitStep | CheckStep, str | None]]:
    """Flatten *plan* into its concrete steps, resolving nested ``run`` steps.

    The dry-run / validation counterpart of :func:`compile_action_plan`: it
    walks the exact same step order the compiler executes — nested plans
    inlined where their ``run`` step sits — but touches no signals and needs
    no factory, so it is safe to call with zero hardware.  Every nested
    ``run`` reference is resolved eagerly, making this the one fail-fast
    walk for unknown nested names and cycles.

    Parameters
    ----------
    plan : ActionPlan
        The validated plan to flatten.
    registry : Mapping[str, ActionPlan]
        Named plans that ``run`` steps may reference.

    Returns
    -------
    list of (step, from_plan)
        Concrete steps (``set`` / ``wait`` / ``check``) in execution order.
        ``from_plan`` is the name of the nested plan a step was inlined
        from (the innermost enclosing ``run`` target), or ``None`` for
        *plan*'s own steps.

    Raises
    ------
    ActionPlanNotFoundError
        When a ``run`` step names a plan missing from *registry*.
    ActionPlanCycleError
        When nested ``run`` steps form a loop.
    """
    flattened: list[tuple[SetStep | WaitStep | CheckStep, str | None]] = []

    def _walk(current: ActionPlan, origin: str | None, stack: tuple[str, ...]) -> None:
        for step in current.steps:
            if isinstance(step, RunPlanStep):
                if step.plan in stack:
                    raise ActionPlanCycleError([*stack, step.plan])
                nested = registry.get(step.plan)
                if nested is None:
                    raise ActionPlanNotFoundError(step.plan, list(registry))
                _walk(nested, step.plan, (*stack, step.plan))
            else:
                flattened.append((step, origin))

    _walk(plan, None, ())
    return flattened


def compile_action_plan(
    plan: ActionPlan,
    *,
    registry: Mapping[str, ActionPlan],
    settables: SettableFactory,
) -> MsgGenerator:
    """Compile an :class:`ActionPlan` into a Bluesky plan executing its steps.

    Steps run strictly in order: ``set`` writes a value (blocking on
    completion unless the step says ``wait_for_execution: false``), ``wait``
    sleeps, ``check`` reads a value and stops the plan on a mismatch, and
    ``run`` executes another named plan from *registry*.

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
        When nested ``run`` steps form a loop.
    """
    yield from _compile(plan, registry, settables, stack=())


def _compile(
    plan: ActionPlan,
    registry: Mapping[str, ActionPlan],
    settables: SettableFactory,
    stack: tuple[str, ...],
) -> MsgGenerator:
    """Execute *plan*'s steps; *stack* is the chain of nested plan names (cycle guard)."""
    for step in plan.steps:
        if isinstance(step, SetStep):
            signal = settables.get_settable(step.device, step.variable)
            # Legacy sync semantics — see the module docstring.
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
