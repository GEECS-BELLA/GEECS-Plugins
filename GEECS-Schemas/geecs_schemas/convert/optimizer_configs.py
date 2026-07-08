"""Convert legacy optimizer YAML into a ScanRequest ``optimization`` block.

Legacy dialect (one file per problem under ``optimizer_configs/``, validated
today by ``geecs_scanner.optimization.config_models.BaseOptimizerConfig``)::

    vocs:
      variables: {U_Hexapod:ypos: [17, 19]}
      objectives: {f: MINIMIZE}
      observables: [x_CoM]
      constraints: {}
    evaluator: {module: ..., class: ..., kwargs: {...}}
    generator: {name: bayes_default}
    xopt_config_overrides:
      multipoint_bax_alignment_l2: {...}   # keyed by generator name
    device_requirements:
      Devices: {UC_HiResMagCam: {synchronous: true, ...}}

Mapping:

- ``vocs`` fields map 1:1 onto :class:`OptimizationSpec` (variables,
  objectives, observables, constraints).
- ``xopt_config_overrides[generator.name]`` becomes ``generator.options``;
  overrides keyed by any *other* name cannot be mapped and raise.
- ``device_requirements`` is really a save-device list in disguise: it is
  converted to a :class:`SaveSet` (via the save-element rules) so nothing is
  lost, and returned alongside the spec.  In the target architecture it is
  auto-derived from the evaluator's analyzers, so a converted config may
  simply discard it.
- ``name`` / ``description`` are returned for the enclosing
  :class:`ScanRequest` to use.
- ``save_devices`` / ``save_devices_file`` (inline or external save-element
  attachments) also convert through the save-element rules; a file path
  cannot be resolved by a pure converter and raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from geecs_schemas.convert._common import (
    LegacyDocument,
    SchemaConversionError,
    load_legacy,
    require_known_keys,
    source_name,
)
from geecs_schemas.convert.save_elements import convert_save_element
from geecs_schemas.save_set import SaveSet
from geecs_schemas.scan_request import OptimizationSpec

_KNOWN_KEYS = [
    "vocs",
    "evaluator",
    "generator",
    "xopt_config_overrides",
    "device_requirements",
    "save_devices",
    "save_devices_file",
    "seed_dump_files",
    "move_to_best_on_finish",
    "name",
    "description",
]


@dataclass
class OptimizerConversion:
    """Everything one legacy optimizer config converts into.

    Attributes
    ----------
    optimization : OptimizationSpec
        The optimization block for a ``mode: optimize`` ScanRequest.
    save_set : SaveSet or None
        Converted ``device_requirements`` / ``save_devices`` (when present) —
        derivable from the analyzers in the target architecture, preserved
        here so nothing is dropped silently.
    name : str
        The config's name (from the file or the legacy ``name`` field).
    description : str
        The legacy ``description`` field, for the enclosing ScanRequest.
    notes : list of str
        Non-fatal information recorded during conversion.
    """

    optimization: OptimizationSpec
    save_set: Optional[SaveSet] = None
    name: str = ""
    description: str = ""
    notes: list[str] = field(default_factory=list)


def convert_optimizer_config(
    source: LegacyDocument, name: str | None = None
) -> OptimizerConversion:
    """Convert one legacy optimizer config document.

    Parameters
    ----------
    source : dict or Path or str
        The legacy document or a path to it (the filename names the config).
    name : str, optional
        Explicit config name (overrides the filename / legacy ``name``).

    Returns
    -------
    OptimizerConversion
        The optimization spec plus preserved side products.

    Raises
    ------
    SchemaConversionError
        Naming any key or value that could not be mapped.
    """
    document = load_legacy(source)
    config_name = name or document.get("name") or source_name(source, fallback="")
    context = f"optimizer config {config_name or '<dict>'!r}"
    require_known_keys(document, _KNOWN_KEYS, context)
    notes: list[str] = []

    vocs = document.get("vocs") or {}
    require_known_keys(
        vocs, ["variables", "objectives", "observables", "constraints"], context
    )
    generator = document.get("generator") or {}
    require_known_keys(generator, ["name"], f"{context} generator")
    generator_name = generator.get("name")
    if not generator_name:
        raise SchemaConversionError(f"{context}: generator has no name.")

    options: dict = {}
    overrides = dict(document.get("xopt_config_overrides") or {})
    if overrides:
        options = overrides.pop(generator_name, {})
        if overrides:
            raise SchemaConversionError(
                f"{context}: xopt_config_overrides keyed by "
                f"{sorted(overrides)} do not match the generator "
                f"{generator_name!r} and cannot be mapped."
            )

    evaluator = document.get("evaluator") or {}
    require_known_keys(evaluator, ["module", "class", "kwargs"], f"{context} evaluator")

    spec = OptimizationSpec.model_validate(
        {
            "variables": vocs.get("variables") or {},
            "objectives": vocs.get("objectives") or {},
            "observables": vocs.get("observables") or [],
            "constraints": vocs.get("constraints") or {},
            "evaluator": {
                "module": evaluator.get("module"),
                "class": evaluator.get("class"),
                "kwargs": evaluator.get("kwargs") or {},
            },
            "generator": {"name": generator_name, "options": options},
            "seed_dump_files": [str(p) for p in document.get("seed_dump_files") or []],
            "move_to_best_on_finish": document.get("move_to_best_on_finish") or False,
        }
    )

    save_set: Optional[SaveSet] = None
    if document.get("save_devices_file"):
        raise SchemaConversionError(
            f"{context}: 'save_devices_file' points outside the document "
            f"({document['save_devices_file']!r}) — convert that file with "
            "convert_save_element and attach it yourself."
        )
    device_requirements = document.get("device_requirements")
    save_devices = document.get("save_devices")
    if device_requirements and save_devices:
        raise SchemaConversionError(
            f"{context}: has both 'device_requirements' and 'save_devices' — "
            "merge them by hand before converting."
        )
    legacy_saves = device_requirements or save_devices
    if legacy_saves and legacy_saves.get("Devices"):
        conversion = convert_save_element(legacy_saves, name=f"{config_name}_saves")
        save_set = conversion.save_set
        notes.extend(conversion.notes)
        if conversion.actions:
            raise SchemaConversionError(
                f"{context}: embedded setup/closeout actions in "
                "device_requirements/save_devices are not expected — convert "
                "the element separately."
            )
        notes.append(
            f"{context}: 'device_requirements' preserved as save set "
            f"{save_set.name!r}; the target architecture derives it from the "
            "evaluator's analyzers instead."
        )

    return OptimizerConversion(
        optimization=spec,
        save_set=save_set,
        name=config_name,
        description=document.get("description") or "",
        notes=notes,
    )
