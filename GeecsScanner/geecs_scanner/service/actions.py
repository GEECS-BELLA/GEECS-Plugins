"""The action-plan preview: the worker's step order, described for the page.

``03 §4``: a manual move is a stock ``mv`` queue item and *a preview is
client-side resolver work*.  The walk itself is
:func:`geecs_bluesky.action_steps.flatten_action_steps` — import-light, the
same function the worker's compiler executes, so the preview and the run
cannot drift; this module only turns each flattened step into the words
the page shows.  Unknown nested names and cycles are refused before
anything is shown as runnable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.models import ActionStepOut


def _fmt(value: Any) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def describe_step(step: Any, origin: str | None) -> ActionStepOut:
    """One flattened step as the page lists it."""
    do = str(step.do)
    out = ActionStepOut(do=do, from_plan=origin, text="")
    if do == "set":
        out.device, out.variable, out.value = step.device, step.variable, step.value
        out.wait = bool(step.wait_for_execution)
        out.text = f"set {step.device}:{step.variable} = {_fmt(step.value)}" + (
            "" if step.wait_for_execution else " (no wait)"
        )
    elif do == "wait":
        out.seconds = float(step.seconds)
        out.text = f"wait {step.seconds:g} s"
    elif do == "check":
        out.device, out.variable, out.expected = (
            step.device,
            step.variable,
            step.expected,
        )
        out.text = f"check {step.device}:{step.variable} == {_fmt(step.expected)}"
    else:  # pragma: no cover - the schema discriminator prevents this
        out.text = do
    return out


def flatten(name: str, plan: Any, registry: Mapping[str, Any]) -> list[ActionStepOut]:
    """Every concrete step of *plan*, nested ``run`` steps inlined, in execution order.

    The walk is the worker's own (:func:`geecs_bluesky.action_steps.flatten_action_steps`),
    so the preview promises the order the compiler executes.

    Raises
    ------
    ScannerError
        ``invalid_request`` for a ``run`` step naming a plan the library
        does not hold, or a chain of ``run`` steps that loops.
    """
    from geecs_bluesky.action_steps import flatten_action_steps
    from geecs_bluesky.exceptions import ActionPlanCycleError, ActionPlanNotFoundError

    try:
        steps = flatten_action_steps(plan, registry=registry)
    except (ActionPlanNotFoundError, ActionPlanCycleError) as exc:
        raise ScannerError("invalid_request", f"action {name!r}: {exc}") from exc
    return [describe_step(step, origin) for step, origin in steps]
