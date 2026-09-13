"""The action-plan preview: flatten a plan's steps without touching hardware.

``03 §4``: a manual move is a stock ``mv`` queue item and *a preview is
client-side resolver work*.  The worker's compiler
(``geecs_bluesky.plans.action_compiler``) has the same walk, but that
module is worker-side and the scanner never imports ``geecs_bluesky.plans``
(``tests/test_boundaries.py``), so the walk is written here over the
schema models alone: nested ``run`` steps inlined where they sit, unknown
names and cycles refused before anything is shown as runnable.
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

    Raises
    ------
    ScannerError
        ``invalid_request`` for a ``run`` step naming a plan the library
        does not hold, or a chain of ``run`` steps that loops.
    """
    out: list[ActionStepOut] = []

    def walk(current: Any, origin: str | None, stack: tuple[str, ...]) -> None:
        for step in current.steps:
            if str(step.do) == "run":
                if step.plan in stack:
                    raise ScannerError(
                        "invalid_request",
                        f"action {name!r}: nested run steps loop "
                        f"({' → '.join((*stack, step.plan))})",
                    )
                nested = registry.get(step.plan)
                if nested is None:
                    raise ScannerError(
                        "invalid_request",
                        f"action {name!r}: run step names {step.plan!r}, which is "
                        f"not in the library ({', '.join(sorted(registry)) or 'empty'})",
                    )
                walk(nested, step.plan, (*stack, step.plan))
            else:
                out.append(describe_step(step, origin))

    walk(plan, None, (name,))
    return out
