"""Convert legacy save-element YAML to :class:`SaveSet` (+ extracted actions).

Legacy dialect (one file per element under ``save_devices/``)::

    Devices:
      UC_ALineEbeam1:
        synchronous: true
        save_nonscalar_data: true
        variable_list: [acq_timestamp, MaxCounts]
        add_all_variables: false
        scan_setup: {Analysis: ['on', 'off']}
        post_analysis_class: MagSpecStitcherAnalysis   # deprecated
    setup_action:
      steps: [...]
    closeout_action:
      steps: [...]

Mapping (intent extracted, mechanics dropped per the schema's derivation
rules — see ``geecs_schemas.save_set``):

- ``save_nonscalar_data`` → ``images``.
- ``variable_list`` → ``scalars``; the bookkeeping ``acq_timestamp`` entry is
  dropped (implicit) with a note.
- ``add_all_variables`` → ``all_scalars``.
- ``synchronous: false`` → ``role: snapshot``; ``synchronous: true`` needs no
  role (derived from the acquisition mode).
- ``setup_action`` / ``closeout_action`` become standalone
  :class:`~geecs_schemas.action_plan.ActionPlan` objects named
  ``<element>_setup`` / ``<element>_closeout``, **and** every entry of the
  element references those plan names in its ``setup`` / ``closeout``
  fields — so the ritual keeps travelling with the devices when entries are
  composed into bigger save sets (references de-duplicate; each plan runs
  once per scan).
- Per-device ``scan_setup`` ``[pre, post]`` pairs become set-steps appended
  to those setup/closeout plans (creating them if needed) — same writes at
  the same moments as the legacy DeviceManager.
- ``post_analysis_class`` was already deprecated in the legacy model; it is
  dropped with an explicit note rather than an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geecs_schemas.action_plan import ActionPlan
from geecs_schemas.convert._common import (
    LegacyDocument,
    SchemaConversionError,
    load_legacy,
    require_known_keys,
    source_name,
)
from geecs_schemas.convert.actions import _convert_step
from geecs_schemas.save_set import SaveSet

_KNOWN_TOP_KEYS = ["Devices", "setup_action", "closeout_action"]
_KNOWN_DEVICE_KEYS = [
    "synchronous",
    "save_nonscalar_data",
    "variable_list",
    "add_all_variables",
    "scan_setup",
    "post_analysis_class",
]


@dataclass
class SaveElementConversion:
    """Everything one legacy save element converts into.

    Attributes
    ----------
    save_set : SaveSet or None
        The what-to-record part.  ``None`` for the corpus's *action-only*
        elements (``Devices: {}`` with only setup/closeout actions).
    actions : dict of str to ActionPlan
        Extracted setup/closeout plans (``<name>_setup`` / ``<name>_closeout``),
        empty when the element had none.
    notes : list of str
        Non-fatal information dropped or normalized during conversion
        (implicit ``acq_timestamp``, deprecated ``post_analysis_class``).
    """

    save_set: SaveSet | None
    actions: dict[str, ActionPlan] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def convert_save_element(
    source: LegacyDocument, name: str | None = None
) -> SaveElementConversion:
    """Convert one legacy save-element document.

    Parameters
    ----------
    source : dict or Path or str
        The legacy document or a path to it (the filename becomes the save
        set's name).
    name : str, optional
        Explicit name for the save set (required when *source* is a dict
        and overrides the filename otherwise).

    Returns
    -------
    SaveElementConversion
        The converted save set, extracted action plans, and notes.

    Raises
    ------
    SchemaConversionError
        Naming any key that could not be mapped.
    """
    document = load_legacy(source)
    set_name = name or source_name(source, fallback="")
    if not set_name:
        raise SchemaConversionError(
            "convert_save_element needs a name: pass name= when converting a dict."
        )
    require_known_keys(document, _KNOWN_TOP_KEYS, f"save element {set_name!r}")

    notes: list[str] = []
    entries: list[dict] = []
    setup_steps: list[dict] = []
    closeout_steps: list[dict] = []

    for action_key, target in (
        ("setup_action", setup_steps),
        ("closeout_action", closeout_steps),
    ):
        body = document.get(action_key)
        if body:
            require_known_keys(
                body, ["steps"], f"save element {set_name!r} {action_key}"
            )
            target.extend(
                _convert_step(step, f"save element {set_name!r} {action_key}")
                for step in body.get("steps") or []
            )

    devices = document.get("Devices") or {}
    if not devices and not (setup_steps or closeout_steps):
        raise SchemaConversionError(
            f"save element {set_name!r}: has no Devices and no actions — "
            "nothing to convert."
        )
    for device, config in devices.items():
        config = config or {}
        context = f"save element {set_name!r} device {device!r}"
        require_known_keys(config, _KNOWN_DEVICE_KEYS, context)

        scalars = list(config.get("variable_list") or [])
        if "acq_timestamp" in scalars:
            scalars = [scalar for scalar in scalars if scalar != "acq_timestamp"]
            notes.append(
                f"{context}: dropped 'acq_timestamp' from the scalar list — "
                "it is recorded implicitly for every entry."
            )
        entry: dict = {"device": device, "scalars": scalars}
        if config.get("save_nonscalar_data"):
            entry["images"] = True
        if config.get("add_all_variables"):
            entry["all_scalars"] = True
        if config.get("synchronous") is False:
            entry["role"] = "snapshot"
        entries.append(entry)

        if config.get("post_analysis_class"):
            notes.append(
                f"{context}: dropped deprecated 'post_analysis_class' "
                f"({config['post_analysis_class']!r}) — post-scan analysis is "
                "configured in ScanAnalysis, not in save sets."
            )

        scan_setup = config.get("scan_setup") or {}
        for variable, pair in scan_setup.items():
            if not isinstance(pair, list) or len(pair) != 2:
                raise SchemaConversionError(
                    f"{context}: scan_setup entry {variable!r} must be a "
                    f"[pre, post] pair, got {pair!r}."
                )
            pre, post = pair
            setup_steps.append(
                {"do": "set", "device": device, "variable": variable, "value": pre}
            )
            closeout_steps.append(
                {"do": "set", "device": device, "variable": variable, "value": post}
            )
            notes.append(
                f"{context}: scan_setup {variable!r} became set-steps in the "
                f"'{set_name}_setup' / '{set_name}_closeout' action plans."
            )

    actions: dict[str, ActionPlan] = {}
    if setup_steps:
        actions[f"{set_name}_setup"] = ActionPlan(
            steps=setup_steps,
            description=f"Extracted from legacy save element {set_name!r}.",
        )
    if closeout_steps:
        actions[f"{set_name}_closeout"] = ActionPlan(
            steps=closeout_steps,
            description=f"Extracted from legacy save element {set_name!r}.",
        )

    # The element's ritual travels with its devices: every entry references
    # the extracted plans by name (references are de-duplicated and each
    # plan runs once per scan, so N entries do not mean N runs).
    if entries and actions:
        for entry in entries:
            if setup_steps:
                entry["setup"] = [f"{set_name}_setup"]
            if closeout_steps:
                entry["closeout"] = [f"{set_name}_closeout"]

    save_set: SaveSet | None = None
    if entries:
        save_set = SaveSet(name=set_name, entries=entries)
    else:
        notes.append(
            f"save element {set_name!r}: action-only element (empty Devices) "
            "— produced no save set, only action plans."
        )

    return SaveElementConversion(save_set=save_set, actions=actions, notes=notes)
