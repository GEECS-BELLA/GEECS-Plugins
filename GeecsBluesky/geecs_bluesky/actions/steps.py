"""Flatten an action plan into its concrete steps — import-light on purpose.

The walk itself is GEECS-Schemas' (:func:`geecs_schemas.action_plan.flatten_action_steps`),
the one step library this worker and GeecsCanopy's module share; this
wrapper raises this package's exceptions, so the worker's compiler and a
client's preview (the web scanner's Actions panel) see the errors they
always did.  It depends on the schema models and this package's
exceptions alone — no bluesky, no hardware.
"""

from __future__ import annotations

from collections.abc import Mapping

from geecs_schemas import action_plan as schema
from geecs_schemas.action_plan import ActionPlan, ConcreteStep

from geecs_bluesky.exceptions import ActionPlanCycleError, ActionPlanNotFoundError

__all__ = ["flatten_action_steps"]


def flatten_action_steps(
    plan: ActionPlan,
    *,
    registry: Mapping[str, ActionPlan],
) -> list[tuple[ConcreteStep, str | None]]:
    """Flatten *plan* into its concrete steps, resolving nested ``run`` steps.

    See :func:`geecs_schemas.action_plan.flatten_action_steps`: the same
    order :func:`~geecs_bluesky.actions.compiler.compile_action_plan`
    executes, touching no signals.

    Raises
    ------
    ActionPlanNotFoundError
        When a ``run`` step names a plan missing from *registry*.
    ActionPlanCycleError
        When nested ``run`` steps form a loop.
    """
    try:
        return schema.flatten_action_steps(plan, registry=registry)
    except schema.ActionPlanNotFoundError as exc:
        raise ActionPlanNotFoundError(exc.plan_name, exc.known_plans) from None
    except schema.ActionPlanCycleError as exc:
        raise ActionPlanCycleError(exc.chain) from None
