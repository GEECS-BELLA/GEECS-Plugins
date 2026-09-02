"""Convert legacy shot-control YAML to :class:`TriggerProfile`.

Legacy dialect (one file per condition under ``shot_control_configurations/``,
validated today by ``geecs_bluesky.models.shot_control.ShotControlConfig``)::

    device: U_DG645_ShotControl
    variables:
      Trigger.Source:
        'OFF': Single shot external rising edges
        SCAN: External rising edges
        STANDBY: External rising edges
        ARMED: Single shot external rising edges
      Trigger.ExecuteSingleShot:
        SINGLESHOT: 'on'
        SCAN: ''            # empty string = no-op for this state

Mapping (semantics reused, not contradicted):

- The single-device per-variable table pivots to per-state **ordered write
  lists**: the legacy file's one ``device`` is emitted into every write, and
  within each state the writes keep the file's variable order (which is the
  order the legacy controller sent them).
- Empty-string values (the legacy "no-op" convention) are simply omitted —
  exactly what ``ShotControlConfig.values_for_state`` did when building the
  writes for a state.
- An empty/deviceless document (Bella's ``{}``, Undulator's ``No Device``)
  means "no shot control configured" and converts to ``None``, mirroring
  ``ShotControlConfig.from_information``.

Parallel operating-condition files (the laser-on/off pair) stay separate
profiles — format v2 has no variant overlay; each converts on its own.
"""

from __future__ import annotations

from typing import Optional

from geecs_schemas.convert._common import (
    LegacyDocument,
    SchemaConversionError,
    as_wire_value,
    load_legacy,
    require_known_keys,
    source_name,
)
from geecs_schemas.trigger_profile import (
    TriggerProfile,
    TriggerState,
)


def convert_shot_control(
    source: LegacyDocument, name: str | None = None
) -> Optional[TriggerProfile]:
    """Convert one legacy shot-control document to a :class:`TriggerProfile`.

    Parameters
    ----------
    source : dict or Path or str
        The legacy document or a path to it (the filename becomes the
        profile name).
    name : str, optional
        Explicit profile name (required when *source* is a dict and
        overrides the filename otherwise).

    Returns
    -------
    TriggerProfile or None
        The converted profile, or ``None`` when the document is empty or
        names no device ("no shot control configured").

    Raises
    ------
    SchemaConversionError
        Naming any key or state that could not be mapped.
    """
    document = load_legacy(source)
    if not document or not document.get("device"):
        return None
    profile_name = name or source_name(source, fallback="")
    if not profile_name:
        raise SchemaConversionError(
            "convert_shot_control needs a name: pass name= when converting a dict."
        )
    require_known_keys(
        document, ["device", "variables"], f"shot control {profile_name!r}"
    )

    device = document["device"]
    states: dict[str, list[dict]] = {}
    # Iterate variables in file order so each state's write list keeps the
    # order the legacy controller sent them.
    for variable, state_values in (document.get("variables") or {}).items():
        for state, value in (state_values or {}).items():
            try:
                state_key = TriggerState(state).value
            except ValueError as exc:
                raise SchemaConversionError(
                    f"shot control {profile_name!r} variable {variable!r}: "
                    f"unknown state {state!r} — expected one of "
                    f"{[s.value for s in TriggerState]}."
                ) from exc
            if value is None or value == "":
                continue  # legacy no-op convention: omit the write
            states.setdefault(state_key, []).append(
                {
                    "device": device,
                    "variable": variable,
                    "value": as_wire_value(value),
                }
            )

    return TriggerProfile(name=profile_name, states=states)
