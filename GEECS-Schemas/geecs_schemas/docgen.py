"""Render schema models as plain-language Markdown reference sections.

The field descriptions and class docstrings on the models are written for
operators and are the single source of truth for documentation: this module
turns them into Markdown so the published reference can never drift from the
code.  GUI tooltips are expected to consume the same descriptions.

Dependency-free by design (standard library + Pydantic introspection only) —
no mkdocs plugins, no YAML library.  ``render_reference()`` emits one big
Markdown string; the docs build decides where to put it.
"""

from __future__ import annotations

import inspect
import types
import typing
from enum import Enum
from typing import Any, Iterator, Mapping, Optional

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from geecs_schemas import SCHEMA_REGISTRY

# Hand-written example YAML per registry kind, shown under each top-level
# model's reference section. Kept here (not in docstrings) so examples can
# be longer than a docstring comfortably allows.
EXAMPLES: dict[str, str] = {
    "scan_request": """\
schema_version: 1
mode: step
axes:
  - variable: jet_z
    positions: {start: 4.0, end: 6.0, step: 0.5}
  # add more axes to scan a grid — the first axis is the outermost
  # (slowest) loop, the last the innermost (fastest), e.g.:
  # - variable: gas_pressure
  #   positions: {values: [1.5, 2.0, 2.5]}
shots_per_step: 10
acquisition: free_run
save_set: undulator_baseline
trigger_profile: htu_shot_control
actions:
  setup: [pre_scan_ebeam]
  per_step: []
  closeout: []
description: "jet z scan with probe"
""",
    "save_set": """\
schema_version: 1
name: undulator_baseline
# the REQUIRED devices — everything else is still logged in the background
entries:
  - device: UC_Amp4_IR_input
    images: true                     # images are always required-tier
    scalars: [MaxCounts, centroidx]  # extras beyond the DB's standard telemetry
  - device: U_HP_Daq
    db_scalars: false                # record ONLY the listed scalars, not the DB set
    scalars: [AnalogOutput.Channel 1]
    at_scan_start: {Analysis: "on"}  # replace the DB's scan-start value
    at_scan_end: {Analysis: null}    # suppress the DB's scan-end write
  - device: U_BCaveHallProbe
    scalars: [Field, Rawfield]
    role: snapshot
  - device: UC_UndulatorRad2
    images: true
    scalars: [MeanCounts]
    # this device's ritual travels with it: these named plans run once
    # before/after any scan whose save set includes this entry
    setup: [visa1_spectrometer_setup]
    closeout: [visa1_spectrometer_closeout]
""",
    "scan_variables": """\
schema_version: 1
variables:
  jet_z:
    target: "U_ESP_JetXYZ:Position.Axis 3"
    kind: motor
  gas_pressure:
    target: "U_HP_Daq:AnalogOutput.Channel 1"
  e_beam_angle_x:
    kind: pseudo
    mode: relative
    targets:
      - target: "U_S3H:Current"
        forward: "composite_var * 1"
      - target: "U_S4H:Current"
        forward: "composite_var * -2"
""",
    "trigger_profile": """\
schema_version: 1
name: htu_shot_control
# each state lists its writes IN ORDER (top to bottom); a transition may
# touch several devices
states:
  OFF:
    - {device: U_DG645_ShotControl, variable: Amplitude.Ch AB, value: "0.5"}
    - {device: U_DG645_ShotControl, variable: Trigger.Source,
       value: Single shot external rising edges}
  STANDBY:
    - {device: U_DG645_ShotControl, variable: Amplitude.Ch AB, value: "0.5"}
    - {device: U_DG645_ShotControl, variable: Trigger.Source,
       value: External rising edges}
  SCAN:
    - {device: U_DG645_ShotControl, variable: Amplitude.Ch AB, value: "4.0"}
    - {device: U_DG645_ShotControl, variable: Trigger.Source,
       value: External rising edges}
    - {device: U_GasJetPLC, variable: DO.Jet, value: "on"}
  ARMED:
    - {device: U_DG645_ShotControl, variable: Amplitude.Ch AB, value: "4.0"}
    - {device: U_DG645_ShotControl, variable: Trigger.Source,
       value: Single shot external rising edges}
  SINGLESHOT:
    - {device: U_DG645_ShotControl, variable: Trigger.ExecuteSingleShot,
       value: "on"}
variants:
  laser_off:
    states:
      OFF:
        - {device: U_DG645_ShotControl, variable: Trigger.Source,
           value: Single shot}
      SCAN:
        - {device: U_DG645_ShotControl, variable: Trigger.Source,
           value: Internal}
      ARMED:
        - {device: U_DG645_ShotControl, variable: Trigger.Source,
           value: Single shot}
""",
    "action_plan": """\
schema_version: 1
description: "Zero the pressure voltage and confirm the PLC output"
steps:
  - do: set
    device: U_HP_Daq
    variable: AnalogOutput.Channel 1
    value: 0
  - do: wait
    seconds: 3
  - do: check
    device: U_148_PLC
    variable: DI.Ch17
    expected: "off"
  - do: run
    plan: close_gaia_internal_shutters
""",
    "action_plan_library": """\
schema_version: 1
plans:
  zero_pressure_voltage:
    steps:
      - do: set
        device: U_HP_Daq
        variable: AnalogOutput.Channel 1
        value: 0
  experiment_closeout:
    steps:
      - do: run
        plan: zero_pressure_voltage
""",
    "experiment_defaults": """\
schema_version: 1
# applied only where a scan request is silent: defaults run first,
# then the scan's own
trigger_profile: htu_shot_control
actions:
  setup: [pre_scan_checklist]
  closeout: [experiment_closeout]
background_telemetry: true   # soft-log every live device not in the save set
description: "HTU standing defaults"
""",
}


def _type_name(annotation: Any) -> str:
    """Render a type annotation as a compact human-readable string.

    Parameters
    ----------
    annotation : Any
        A type annotation from a Pydantic field.

    Returns
    -------
    str
        A readable name such as ``list[str]`` or ``PositionRange | PositionList``.
    """
    if annotation is None or annotation is type(None):
        return "None"
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        parts = [_type_name(a) for a in args if a is not type(None)]
        rendered = " | ".join(parts)
        if type(None) in args:
            rendered += " (optional)"
        return rendered
    if origin is typing.Literal:
        return " | ".join(repr(a) for a in args)
    if origin is not None:
        inner = ", ".join(_type_name(a) for a in args)
        name = getattr(origin, "__name__", str(origin))
        return f"{name}[{inner}]" if inner else name
    return getattr(annotation, "__name__", str(annotation))


def _default_repr(field: FieldInfo) -> str:
    """Render a field's default for the reference table.

    Parameters
    ----------
    field : pydantic.fields.FieldInfo
        The field to describe.

    Returns
    -------
    str
        ``"—"`` for required fields, otherwise a short default description.
    """
    if field.is_required():
        return "—"
    if field.default_factory is not None:
        produced = field.default_factory()
        return "empty" if produced in ({}, [], "") else repr(produced)
    if field.default is PydanticUndefined:
        return "—"
    if isinstance(field.default, Enum):
        return repr(field.default.value)
    return repr(field.default)


def _operator_paragraph(model: type[BaseModel]) -> str:
    """Extract the first (operator-language) paragraph of a model docstring.

    Parameters
    ----------
    model : type of pydantic.BaseModel
        The model whose docstring to read.

    Returns
    -------
    str
        The first paragraph, whitespace-normalized; empty if undocumented.
    """
    doc = inspect.getdoc(model) or ""
    first = doc.split("\n\n", 1)[0]
    return " ".join(first.split())


def iter_nested_models(
    model: type[BaseModel], _seen: Optional[set[type[BaseModel]]] = None
) -> Iterator[type[BaseModel]]:
    """Yield *model* and every Pydantic model reachable from its fields.

    Parameters
    ----------
    model : type of pydantic.BaseModel
        The root model to walk.
    _seen : set, optional
        Internal recursion guard; leave unset.

    Yields
    ------
    type of pydantic.BaseModel
        Each distinct model class, depth-first, root first.
    """
    seen = _seen if _seen is not None else set()
    if model in seen:
        return
    seen.add(model)
    yield model

    def _walk(annotation: Any) -> Iterator[type[BaseModel]]:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield from iter_nested_models(annotation, seen)
            return
        for arg in typing.get_args(annotation):
            yield from _walk(arg)

    for field in model.model_fields.values():
        yield from _walk(field.annotation)


def render_model_markdown(model: type[BaseModel], example: str | None = None) -> str:
    """Render one model as a Markdown reference section.

    Parameters
    ----------
    model : type of pydantic.BaseModel
        The model to document.
    example : str, optional
        Example YAML to include under the field table.

    Returns
    -------
    str
        Markdown: a heading, the operator paragraph, a field table, and the
        example (when given).
    """
    lines = [f"### {model.__name__}", ""]
    intro = _operator_paragraph(model)
    if intro:
        lines += [intro, ""]
    lines += [
        "| Field | Type | Required | Default | What it does |",
        "|---|---|---|---|---|",
    ]
    for name, field in model.model_fields.items():
        description = " ".join((field.description or "").split())
        required = "yes" if field.is_required() else "no"
        lines.append(
            f"| `{name}` | `{_type_name(field.annotation)}` | {required} "
            f"| {_default_repr(field)} | {description} |"
        )
    lines.append("")
    if example:
        lines += ["Example:", "", "```yaml", example.rstrip("\n"), "```", ""]
    return "\n".join(lines)


def render_reference(
    registry: Optional[Mapping[str, type[BaseModel]]] = None,
) -> str:
    """Render the full Markdown reference for every registered schema.

    Parameters
    ----------
    registry : mapping, optional
        kind → model mapping; defaults to
        :data:`geecs_schemas.SCHEMA_REGISTRY`.

    Returns
    -------
    str
        One Markdown document: a section per config kind, each covering the
        top-level model and every nested model it uses.
    """
    registry = SCHEMA_REGISTRY if registry is None else registry
    documented: set[type[BaseModel]] = set()
    out: list[str] = ["# GEECS config schema reference", ""]
    for kind, model in registry.items():
        out += [f"## `{kind}`", ""]
        out.append(render_model_markdown(model, example=EXAMPLES.get(kind)))
        for nested in iter_nested_models(model):
            if nested is model or nested in documented:
                continue
            documented.add(nested)
            out.append(render_model_markdown(nested))
        documented.add(model)
    return "\n".join(out)
