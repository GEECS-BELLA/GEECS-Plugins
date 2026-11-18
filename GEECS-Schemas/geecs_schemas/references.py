"""Which documents name which: the references one configs-repo document makes to another, by name.

A document names another by its file stem: a preset names an optimizer
config (its plan's ``optimizer_config``), an optimizer config the
diagnostics it measures, a preset or the experiment
defaults name a trigger profile, a setup's ``apply`` step names another
setup, a screen's button names a setup, the home layout names screens, an
analysis group names diagnostics (recipes or ``.himg`` jobs, by stem across
the whole analyzers tree). A named document that is deleted while another
still names it breaks that reader, so this is the one table a delete's
check and the configs repo's gate both read. Within an experiment a name is
looked up in the same experiment's folder; an analysis group's in the
shared analyzers tree.

A name is the one its reader looks up (``READ_AS``): an optimizer config
named ``align.yaml`` in a preset's plan is ``align``, because the worker's
resolver strips the suffix (``storage.yaml_stem``, the function it calls).
Every other reader looks the literal stem up, so its names are kept as
written: the worker's trigger profiles are keyed by stem (a preset or the
defaults naming ``laser_off.yaml`` is refused when it runs, though the
scanner's preflight would resolve it), setups by their ``name`` (= stem),
the Control Room's screens by ``<name>.yaml``, the analyzers tree's
diagnostics by stem.

Not references: the shot offsets' ``trigger_profile`` (a record of what a
measurement fired through, not something read by name) and a scan
request's save sets (not files in the configs repo).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from pydantic import BaseModel

from geecs_schemas.analysis import AnalysisGroup
from geecs_schemas.experiment_defaults import ExperimentDefaults
from geecs_schemas.home_layout import HomeLayout
from geecs_schemas.optimizer_config import DiagnosticMeasurement, OptimizerConfig
from geecs_schemas.preset import Preset
from geecs_schemas.screens import Screen
from geecs_schemas.setup import Setup, SetupApplyStep
from geecs_schemas.storage import yaml_stem

#: The kinds a kind's documents may name (a referrer kind -> the kinds it names).
REFERS_TO: dict[str, tuple[str, ...]] = {
    "preset": ("trigger_profile", "optimizer_config"),
    "optimizer_config": ("analysis_recipe", "himg_job"),
    "experiment_defaults": ("trigger_profile",),
    "setup": ("setup",),
    "screen": ("setup",),
    "home_layout": ("screen",),
    "analysis_group": ("analysis_recipe", "himg_job"),
}

#: How a kind's reader resolves the name it is given, where that is not the literal stem.
READ_AS: dict[str, Callable[[str], str]] = {"optimizer_config": yaml_stem}


def _actions(value: Any, parent: str = "") -> Iterator[str]:  # Any: a dumped document
    """Every ``bind.action`` in a dumped screen, however deep its widgets sit."""
    if isinstance(value, dict):
        if parent == "bind" and isinstance(value.get("action"), str):
            yield value["action"]
        for key, inner in value.items():
            yield from _actions(inner, key)
    elif isinstance(value, list):
        for inner in value:
            yield from _actions(inner, parent)


def references(document: BaseModel) -> list[tuple[str, str]]:
    """The documents ``document`` names: ``(kind, name)`` pairs, in order, each once, each name as its reader resolves it.

    For an analysis group the kind is ``analysis_recipe``; a ``.himg`` job
    of that stem answers it too (the two share the analyzers tree).
    """
    out: list[tuple[str, str]] = []
    if isinstance(document, (Preset, ExperimentDefaults)) and document.trigger_profile:
        out.append(("trigger_profile", document.trigger_profile))
    if isinstance(
        document, Preset
    ):  # an optimize preset names its optimizer config in the plan's kwargs
        name = document.plan.kwargs.get("optimizer_config") if document.plan else None
        if isinstance(name, str) and name:
            out.append(("optimizer_config", name))
    elif isinstance(document, OptimizerConfig):
        out += [("analysis_recipe", m.diagnostic) for m in document.measurements.values() if isinstance(m, DiagnosticMeasurement)]  # fmt: skip
    elif isinstance(document, Setup):
        out += [("setup", s.setup) for s in document.steps if isinstance(s, SetupApplyStep)]  # fmt: skip
    elif isinstance(document, Screen):
        out += [("setup", a) for a in _actions(document.model_dump(mode="json"))]
    elif isinstance(document, HomeLayout):
        out += [("screen", s) for area in document.areas for s in area.screens]
    elif isinstance(document, AnalysisGroup):
        out += [("analysis_recipe", r.ref) for r in document.analyzers]
    return list(
        dict.fromkeys((kind, READ_AS.get(kind, str)(name)) for kind, name in out)
    )


__all__ = ["READ_AS", "REFERS_TO", "references"]
