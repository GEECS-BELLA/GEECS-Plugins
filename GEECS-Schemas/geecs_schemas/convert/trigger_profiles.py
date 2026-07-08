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

- The per-variable table pivots to per-state ``states: {STATE: {var: value}}``.
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
from geecs_schemas.trigger_profile import TriggerProfile, TriggerState, TriggerVariant


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

    states: dict[str, dict[str, str]] = {}
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
            states.setdefault(state_key, {})[variable] = as_wire_value(value)

    return TriggerProfile(name=profile_name, device=document["device"], states=states)


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
        ``HTU-LaserOFF``).  Must drive the same device.
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
        If the profiles drive different devices, or *other* omits a write
        that *base* has (a variant can override and add writes, but cannot
        remove one).
    """
    if base.device != other.device:
        raise SchemaConversionError(
            f"Cannot fold {other.name!r} into {base.name!r} as a variant: "
            f"they drive different devices ({other.device!r} vs "
            f"{base.device!r})."
        )

    overlay: dict[str, dict[str, str]] = {}
    for state in TriggerState:
        base_writes = base.writes_for(state)
        other_writes = other.writes_for(state)
        removed = sorted(set(base_writes) - set(other_writes))
        if removed:
            raise SchemaConversionError(
                f"Cannot fold {other.name!r} into {base.name!r} as variant "
                f"{variant_name!r}: state {state.value} drops write(s) "
                f"{removed}, which a variant overlay cannot express."
            )
        diff = {
            variable: value
            for variable, value in other_writes.items()
            if base_writes.get(variable) != value
        }
        if diff:
            overlay[state.value] = diff

    variants = dict(base.variants)
    variants[variant_name] = TriggerVariant(
        states=overlay,
        description=f"Folded from legacy profile {other.name!r}.",
    )
    return base.model_copy(update={"variants": variants})
