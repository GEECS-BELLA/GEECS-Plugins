"""Flatten an action plan into its concrete steps — import-light on purpose.

The one walk both sides of the queue share: the worker's compiler
(:mod:`geecs_bluesky.plans.action_compiler`) executes exactly this order,
and a client's preview (the web scanner's Actions panel) promises it.  It
depends on the schema models and this package's exceptions alone — no
bluesky, no hardware — so a client that must never import
``geecs_bluesky.plans`` can still show the operator the steps the worker
will run, from the same function.
"""

from __future__ import annotations

from collections.abc import Mapping

from geecs_schemas.action_plan import (
    ActionPlan,
    CheckStep,
    RunPlanStep,
    SetStep,
    WaitStep,
)

from geecs_bluesky.exceptions import ActionPlanCycleError, ActionPlanNotFoundError

__all__ = ["flatten_action_steps"]


def flatten_action_steps(
    plan: ActionPlan,
    *,
    registry: Mapping[str, ActionPlan],
) -> list[tuple[SetStep | WaitStep | CheckStep, str | None]]:
    """Flatten *plan* into its concrete steps, resolving nested ``run`` steps.

    The dry-run / validation counterpart of
    :func:`~geecs_bluesky.plans.action_compiler.compile_action_plan`: it
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
