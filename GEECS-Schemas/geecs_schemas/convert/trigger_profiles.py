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

The laser-on/off duality — parallel near-identical files — becomes an
explicit **variant** via :func:`merge_trigger_variant`: the second file is
diffed against the base and only the differing writes are stored.
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
    TriggerVariant,
    TriggerWrite,
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


def merge_trigger_variant(
    base: TriggerProfile, other: TriggerProfile, variant_name: str
) -> TriggerProfile:
    """Fold a parallel legacy profile into *base* as a named variant.

    Parameters
    ----------
    base : TriggerProfile
        The profile whose states are the defaults (e.g. converted
        ``HTU-Normal``).
    other : TriggerProfile
        The parallel condition to express as a variant (e.g. converted
        ``HTU-LaserOFF``).
    variant_name : str
        Name for the variant (e.g. ``"laser_off"``).

    Returns
    -------
    TriggerProfile
        A copy of *base* with ``variants[variant_name]`` holding exactly the
        writes where *other* differs from (or adds to) *base*.

    Raises
    ------
    SchemaConversionError
        If *other* omits a device variable that *base* writes (a variant
        overlay can override and add writes, but cannot remove one).
    """
    overlay: dict[str, list[TriggerWrite]] = {}
    for state in TriggerState:
        base_values = {
            (write.device, write.variable): write.value
            for write in base.writes_for(state)
        }
        other_writes = other.writes_for(state)
        other_targets = {(write.device, write.variable) for write in other_writes}
        removed = sorted(set(base_values) - other_targets)
        if removed:
            raise SchemaConversionError(
                f"Cannot fold {other.name!r} into {base.name!r} as variant "
                f"{variant_name!r}: state {state.value} drops write(s) "
                f"{removed}, which a variant overlay cannot express."
            )
        diff = [
            write
            for write in other_writes
            if base_values.get((write.device, write.variable)) != write.value
        ]
        if diff:
            overlay[state.value] = diff

    variants = dict(base.variants)
    variants[variant_name] = TriggerVariant(
        states=overlay,
        description=f"Folded from legacy profile {other.name!r}.",
    )
    return base.model_copy(update={"variants": variants})
