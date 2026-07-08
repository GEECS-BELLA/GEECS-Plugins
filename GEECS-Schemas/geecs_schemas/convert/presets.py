"""Convert legacy scan presets to :class:`ScanRequest`.

Legacy dialect (one file per preset under ``scan_presets/``)::

    Devices:              # names of save elements to record
    - LP-FocusDiagnostics
    Info: Focus Scan
    Scan Mode: 1D Scan    # '1D Scan' | 'No Scan' | 'Background'
    Shot per Step: 10     # 1D Scan only
    Start: -18.0          # 1D Scan only
    Stop: -26.0
    Step Size: 0.5
    Variable: Mode Imager Stage
    Num Shots: 1000       # No Scan / Background only

Mapping:

- ``1D Scan`` → ``mode: step`` with a single-entry ``axes`` list whose
  positions are a start/end/step range (start→stop direction preserved;
  legacy allowed a "descending" range with a positive step size, which the
  new range keeps by ignoring the step's sign).  Legacy presets are always
  1-D; multi-axis grids are a new-schema capability.
- ``No Scan`` → ``mode: noscan`` with ``shots_per_step = Num Shots`` (one
  motionless bin).
- ``Background`` → ``mode: noscan`` + ``background: true`` — background was
  never a distinct acquisition behaviour, only a metadata flag.
- ``Info`` → ``description``.
- ``Devices`` names several save *elements*; a :class:`ScanRequest` names
  one save *set*.  When the converted elements are supplied, they are
  composed into a single save set named after the preset; otherwise the
  referenced element names are returned for later composition.

Not expressed by legacy presets (left at schema defaults): ``acquisition``,
``trigger_profile``, ``actions``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from geecs_schemas.convert._common import (
    LegacyDocument,
    SchemaConversionError,
    load_legacy,
    require_known_keys,
    source_name,
)
from geecs_schemas.save_set import SaveSet, SaveSetEntry
from geecs_schemas.scan_request import ScanRequest

_KNOWN_KEYS = [
    "Devices",
    "Info",
    "Scan Mode",
    "Num Shots",
    "Shot per Step",
    "Start",
    "Stop",
    "Step Size",
    "Variable",
]

_MODE_MAP = {"1D Scan": "step", "No Scan": "noscan", "Background": "noscan"}


@dataclass
class PresetConversion:
    """Everything one legacy scan preset converts into.

    Attributes
    ----------
    scan_request : ScanRequest
        The converted request; its ``save_set`` names the composed set.
    element_names : list of str
        The legacy save-element names the preset referenced.
    composed_save_set : SaveSet or None
        The union of those elements as one save set — only when the caller
        supplied the converted elements.
    notes : list of str
        Non-fatal information recorded during conversion.
    """

    scan_request: ScanRequest
    element_names: list[str] = field(default_factory=list)
    composed_save_set: Optional[SaveSet] = None
    notes: list[str] = field(default_factory=list)


def compose_save_sets(name: str, parts: list[SaveSet]) -> SaveSet:
    """Union several save sets into one (a preset's device list, merged).

    When two parts record the same device, the entries merge the way the
    legacy DeviceManager did ("if the device already exists, extends its
    variable subscription"): scalar lists union in order, ``images`` /
    ``all_scalars`` OR together.  Contradictory role *overrides* have no
    legacy merge rule and raise.

    Parameters
    ----------
    name : str
        Name for the composed set.
    parts : list of SaveSet
        The sets to merge.

    Returns
    -------
    SaveSet
        One set containing every device of every part.

    Raises
    ------
    SchemaConversionError
        If two parts give the same device different explicit ``role``
        overrides.
    """
    merged: dict[str, SaveSetEntry] = {}
    for part in parts:
        for entry in part.entries:
            existing = merged.get(entry.device)
            if existing is None:
                merged[entry.device] = entry.model_copy(deep=True)
                continue
            if (
                existing.role is not None
                and entry.role is not None
                and existing.role != entry.role
            ):
                raise SchemaConversionError(
                    f"Composing save set {name!r}: device {entry.device!r} "
                    f"gets conflicting role overrides "
                    f"({existing.role.value!r} vs {entry.role.value!r})."
                )
            for field_name in ("at_scan_start", "at_scan_end"):
                first = getattr(existing, field_name)
                second = getattr(entry, field_name)
                conflicts = sorted(
                    variable
                    for variable in set(first) & set(second)
                    if first[variable] != second[variable]
                )
                if conflicts:
                    raise SchemaConversionError(
                        f"Composing save set {name!r}: device "
                        f"{entry.device!r} gets conflicting {field_name} "
                        f"overrides for {conflicts}."
                    )
            merged[entry.device] = SaveSetEntry(
                device=entry.device,
                scalars=existing.scalars
                + [s for s in entry.scalars if s not in existing.scalars],
                all_scalars=existing.all_scalars or entry.all_scalars,
                images=existing.images or entry.images,
                role=existing.role or entry.role,
                setup=existing.setup
                + [p for p in entry.setup if p not in existing.setup],
                closeout=existing.closeout
                + [p for p in entry.closeout if p not in existing.closeout],
                db_scalars=existing.db_scalars or entry.db_scalars,
                at_scan_start={**existing.at_scan_start, **entry.at_scan_start},
                at_scan_end={**existing.at_scan_end, **entry.at_scan_end},
            )
    return SaveSet(name=name, entries=list(merged.values()))


def convert_scan_preset(
    source: LegacyDocument,
    name: str | None = None,
    save_sets: Optional[Mapping[str, Optional[SaveSet]]] = None,
) -> PresetConversion:
    """Convert one legacy scan preset to a :class:`ScanRequest`.

    Parameters
    ----------
    source : dict or Path or str
        The legacy preset document or a path to it (the filename names the
        preset).
    name : str, optional
        Explicit preset name (required when *source* is a dict and overrides
        the filename otherwise).
    save_sets : mapping, optional
        Already-converted save sets keyed by legacy element name.  When
        given, the preset's ``Devices`` list is composed into one save set
        named after the preset.  A ``None`` value marks an *action-only*
        legacy element (it contributes no devices, only its extracted
        action plans) and is skipped with a note.

    Returns
    -------
    PresetConversion
        The converted request plus composition results.

    Raises
    ------
    SchemaConversionError
        Naming any key, mode, or missing element that could not be mapped.
    """
    document = load_legacy(source)
    preset_name = name or source_name(source, fallback="")
    if not preset_name:
        raise SchemaConversionError(
            "convert_scan_preset needs a name: pass name= when converting a dict."
        )
    context = f"scan preset {preset_name!r}"
    require_known_keys(document, _KNOWN_KEYS, context)

    legacy_mode = document.get("Scan Mode")
    if legacy_mode not in _MODE_MAP:
        raise SchemaConversionError(
            f"{context}: unknown Scan Mode {legacy_mode!r} — expected one of "
            f"{sorted(_MODE_MAP)}. (Optimization presets did not exist in "
            "the legacy preset dialect.)"
        )

    request: dict = {
        "mode": _MODE_MAP[legacy_mode],
        "description": document.get("Info") or "",
        "background": legacy_mode == "Background",
        "save_set": preset_name,
    }
    if legacy_mode == "1D Scan":
        for key in ("Variable", "Start", "Stop", "Step Size", "Shot per Step"):
            if key not in document:
                raise SchemaConversionError(
                    f"{context}: 1D Scan preset is missing {key!r}."
                )
        # Legacy presets are always 1-D: one entry in the new axes list.
        request["axes"] = [
            {
                "variable": document["Variable"],
                "positions": {
                    "start": document["Start"],
                    "end": document["Stop"],
                    "step": document["Step Size"],
                },
            }
        ]
        request["shots_per_step"] = document["Shot per Step"]
    else:
        if "Num Shots" not in document:
            raise SchemaConversionError(
                f"{context}: {legacy_mode} preset is missing 'Num Shots'."
            )
        request["shots_per_step"] = document["Num Shots"]

    element_names = list(document.get("Devices") or [])
    notes: list[str] = []
    composed: Optional[SaveSet] = None
    if save_sets is not None:
        missing = [n for n in element_names if n not in save_sets]
        if missing:
            raise SchemaConversionError(
                f"{context}: references save elements with no converted "
                f"save set: {missing}."
            )
        parts: list[SaveSet] = []
        for element in element_names:
            part = save_sets[element]
            if part is None:
                notes.append(
                    f"{context}: element {element!r} is action-only — it "
                    "contributes no devices to the composed save set."
                )
            else:
                parts.append(part)
        if parts:
            composed = compose_save_sets(preset_name, parts)
        else:
            notes.append(
                f"{context}: every referenced element is action-only — no "
                "save set was composed."
            )
    else:
        notes.append(
            f"{context}: referenced save elements {element_names} were not "
            f"composed — pass save_sets= to build the {preset_name!r} save set."
        )

    return PresetConversion(
        scan_request=ScanRequest.model_validate(request),
        element_names=element_names,
        composed_save_set=composed,
        notes=notes,
    )
