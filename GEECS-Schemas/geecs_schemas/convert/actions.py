"""Convert the legacy action library YAML to :class:`ActionPlanLibrary`.

Legacy dialect (``action_library/actions.yaml``)::

    actions:
      zero_pressure_voltage:
        steps:
        - action: set
          device: U_HP_Daq
          variable: AnalogOutput.Channel 1
          value: 0
          wait_for_execution: true
        - action: wait
          wait: 3
        - action: get
          device: U_148_PLC
          variable: DI.Ch17
          expected_value: 'off'
        - action: execute
          action_name: close_gaia_internal_shutters

Step mapping (semantics preserved verbatim): ``set`` → ``set``; ``wait`` →
``wait`` (``wait`` → ``seconds``); ``get`` → ``check`` (``expected_value`` →
``expected``); ``execute`` → ``run`` (``action_name`` → ``plan``).  The
legacy ``run`` step (external script/class execution) has no v1
representation and raises loudly.

``assigned_actions.yaml`` is GUI front-end state (which plans are pinned as
buttons), not a schema document; :func:`convert_assigned_actions` extracts
the plain name list and can validate it against a converted library.
"""

from __future__ import annotations

from typing import Optional

from geecs_schemas.action_plan import ActionPlan, ActionPlanLibrary
from geecs_schemas.convert._common import (
    LegacyDocument,
    SchemaConversionError,
    load_legacy,
    require_known_keys,
)


def _convert_step(step: dict, context: str) -> dict:
    """Convert one legacy action step dict to the new step shape.

    Parameters
    ----------
    step : dict
        Legacy step (``action: set|wait|get|execute|run`` + parameters).
    context : str
        Human-readable location for error messages.

    Returns
    -------
    dict
        The equivalent new-schema step dict.

    Raises
    ------
    SchemaConversionError
        For unknown step types, the unsupported legacy script ``run`` step,
        or unmappable step keys.
    """
    kind = step.get("action")
    if kind == "set":
        require_known_keys(
            step,
            ["action", "device", "variable", "value", "wait_for_execution"],
            context,
        )
        converted = {
            "do": "set",
            "device": step["device"],
            "variable": step["variable"],
            "value": step["value"],
        }
        if "wait_for_execution" in step and step["wait_for_execution"] is not None:
            converted["wait_for_execution"] = step["wait_for_execution"]
        return converted
    if kind == "wait":
        require_known_keys(step, ["action", "wait"], context)
        return {"do": "wait", "seconds": step["wait"]}
    if kind == "get":
        require_known_keys(
            step, ["action", "device", "variable", "expected_value"], context
        )
        return {
            "do": "check",
            "device": step["device"],
            "variable": step["variable"],
            "expected": step["expected_value"],
        }
    if kind == "execute":
        require_known_keys(step, ["action", "action_name"], context)
        return {"do": "run", "plan": step["action_name"]}
    if kind == "run":
        raise SchemaConversionError(
            f"{context}: legacy 'run' steps (external script execution, "
            f"file_name={step.get('file_name')!r}) are not carried into "
            "schema v1 — port the script to a Bluesky plan instead."
        )
    raise SchemaConversionError(
        f"{context}: unknown legacy step type {kind!r} — expected one of "
        "set / wait / get / execute."
    )


def convert_action_library(source: LegacyDocument) -> ActionPlanLibrary:
    """Convert a legacy ``actions.yaml`` document to an :class:`ActionPlanLibrary`.

    Parameters
    ----------
    source : dict or Path or str
        The legacy document or a path to it.

    Returns
    -------
    ActionPlanLibrary
        The validated library (nested ``run`` references checked).

    Raises
    ------
    SchemaConversionError
        Naming any key, step type, or reference that could not be mapped.
    """
    document = load_legacy(source)
    require_known_keys(document, ["actions"], "action library")
    plans: dict[str, ActionPlan] = {}
    for name, body in (document.get("actions") or {}).items():
        context = f"action {name!r}"
        require_known_keys(body, ["steps"], context)
        steps = [
            _convert_step(step, f"{context} step {index}")
            for index, step in enumerate(body.get("steps") or [])
        ]
        if not steps:
            raise SchemaConversionError(f"{context}: has no steps.")
        plans[name] = ActionPlan(steps=steps)
    return ActionPlanLibrary(plans=plans)


def convert_assigned_actions(
    source: LegacyDocument, library: Optional[ActionPlanLibrary] = None
) -> list[str]:
    """Extract the pinned-plan name list from ``assigned_actions.yaml``.

    Parameters
    ----------
    source : dict or Path or str
        The legacy document or a path to it.
    library : ActionPlanLibrary, optional
        When given, every name is checked to exist in the library.

    Returns
    -------
    list of str
        The plan names, in order.

    Raises
    ------
    SchemaConversionError
        On unmappable keys, or names missing from *library*.
    """
    document = load_legacy(source)
    require_known_keys(document, ["assigned_actions"], "assigned actions")
    names: list[str] = []
    for entry in document.get("assigned_actions") or []:
        require_known_keys(entry, ["name"], "assigned action entry")
        names.append(entry["name"])
    if library is not None:
        missing = [name for name in names if name not in library.plans]
        if missing:
            raise SchemaConversionError(
                f"Assigned actions reference plans missing from the library: {missing}."
            )
    return names
