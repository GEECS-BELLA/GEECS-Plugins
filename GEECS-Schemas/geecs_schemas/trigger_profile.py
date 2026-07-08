"""TriggerProfile — how the machine trigger is driven through its named states.

A trigger profile tells the scanner which device controls the shot trigger
(and the gas jet gating that rides on it) and exactly what to write to put it
in each state: OFF, STANDBY, SCAN, SINGLESHOT, ARMED.  You would edit one
when the timing hardware changes, or when a state needs an extra write (a
different delay, a different amplitude).  A scan picks a profile by name in
its :class:`~geecs_schemas.scan_request.ScanRequest`.

Variants replace parallel files
-------------------------------
Operating conditions that used to be near-duplicate YAML files (the classic
laser-on / laser-off pair) become **variants** inside one profile: a variant
overlays a few state writes on top of the base profile (e.g. ``laser_off``
switches ``Trigger.Source`` to internal).  One file, explicit differences.

Developer notes
---------------
Successor of the shot-control YAML validated today by
``geecs_bluesky.models.shot_control.ShotControlConfig``.  Semantics are kept,
not contradicted:

- The layout pivots from per-variable ``{variable: {state: value}}`` to
  per-state ``{state: {variable: value}}`` — the state is the unit an
  operator (and the future ``TRIG:STATE`` PV) thinks in.
- Values are **verbatim wire strings** ("4.0", "on", enum labels); nothing is
  coerced.
- The legacy empty-string "no-op for this state" convention is retired:
  a no-op is expressed by *omitting* the variable from the state (matching
  ``ShotControlConfig.values_for_state``, which skipped empty strings).
  Empty-string values are rejected.
- A state with no writes is "not defined" for this profile, exactly like
  ``ShotControlConfig.defines_state``.

The model is deliberately **device-agnostic**: shot-control devices are
usually DG645 delay generators, but nothing here assumes that — any device
with settable variables can be a trigger device.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel


class TriggerState(str, Enum):
    """The named states a trigger device can be driven to.

    These names survived from the legacy system because they match how
    operators think about the machine.

    Attributes
    ----------
    OFF : str
        Trigger fully stopped (used e.g. before timing synchronization).
    STANDBY : str
        Between-steps idle: trigger free-runs but data-taking output (e.g.
        the gas jet) is off.
    SCAN : str
        Taking data: trigger running with data-taking output on.
    SINGLESHOT : str
        Fire exactly one shot (strict acquisition).
    ARMED : str
        Ready for single shots: data-taking output on, trigger switched to
        single-shot source, waiting for SINGLESHOT commands.
    """

    OFF = "OFF"
    STANDBY = "STANDBY"
    SCAN = "SCAN"
    SINGLESHOT = "SINGLESHOT"
    ARMED = "ARMED"


# {variable name: verbatim wire value}. A variable simply absent from a state
# means "leave it alone" — the retired legacy convention used "" for this.
StateWrites = dict[str, str]


def _normalize_state_keys(states: dict) -> dict:
    """Undo YAML 1.1's parsing of a bare ``OFF:`` key into boolean ``False``.

    Parameters
    ----------
    states : dict
        Raw mapping of state → writes, as parsed from YAML.

    Returns
    -------
    dict
        The mapping with a ``False`` key replaced by ``"OFF"`` so operators
        don't have to remember to quote it.
    """
    if isinstance(states, dict) and False in states:
        states = {
            ("OFF" if key is False else key): value for key, value in states.items()
        }
    return states


def _reject_empty_values(states: dict) -> dict:
    """Reject empty-string writes (legacy no-op convention is retired).

    Parameters
    ----------
    states : dict
        Mapping of state → {variable: value}.

    Returns
    -------
    dict
        The validated mapping, unchanged.

    Raises
    ------
    ValueError
        If any variable's value is an empty string.
    """
    for state, writes in states.items():
        for variable, value in (writes or {}).items():
            if value == "":
                state_name = getattr(state, "value", state)
                raise ValueError(
                    f"State {state_name!r} gives variable {variable!r} an "
                    "empty value. To leave a variable untouched in a state, "
                    "omit it from that state instead."
                )
    return states


class TriggerVariant(SchemaModel):
    """A named operating condition that tweaks a few writes of the profile.

    A variant lists only what *differs* from the base profile — for example
    a ``laser_off`` variant that flips the trigger source to internal while
    everything else stays as the base defines it.
    """

    states: dict[TriggerState, StateWrites] = Field(
        description=(
            "Only the writes that differ from the base profile, per state. "
            "Anything not listed here keeps its base value."
        )
    )
    description: str = Field(
        "",
        description="Optional note about when to use this variant.",
    )

    _fix_off_key = field_validator("states", mode="before")(_normalize_state_keys)
    _no_empty = field_validator("states")(_reject_empty_values)


class TriggerProfile(VersionedSchemaModel):
    """The writes that drive one trigger device through its named states.

    Names the device that controls shots and, for each state (OFF, STANDBY,
    SCAN, SINGLESHOT, ARMED), lists exactly which variables to set and to
    what.  Edit it when timing hardware or its settings change; add a
    variant when you need a named alternative condition (e.g. laser off)
    instead of a copy of the whole file.

    Notes
    -----
    Values are sent verbatim over the GEECS wire protocol.  Use
    :meth:`writes_for` / :meth:`defines_state` instead of digging the dicts —
    they implement the variant overlay and the "state with no writes is not
    defined" rule.
    """

    name: str = Field(
        description="The name scans use to refer to this trigger profile."
    )
    device: str = Field(
        min_length=1,
        description=(
            "The device that controls the shot trigger, e.g. "
            "'U_DG645_ShotControl'. Any settable device works — nothing here "
            "is specific to one hardware type."
        ),
    )
    states: dict[TriggerState, StateWrites] = Field(
        default_factory=dict,
        description=(
            "For each trigger state, the variable values that put the device "
            "into it. Omit a variable from a state to leave it untouched."
        ),
    )
    variants: dict[str, TriggerVariant] = Field(
        default_factory=dict,
        description=(
            "Named alternative operating conditions (e.g. 'laser_off'), each "
            "listing only the writes that differ from the base states."
        ),
    )
    description: str = Field(
        "",
        description="Optional note about what setup this profile is for.",
    )

    _fix_off_key = field_validator("states", mode="before")(_normalize_state_keys)
    _no_empty = field_validator("states")(_reject_empty_values)

    def writes_for(
        self, state: "TriggerState | str", variant: str | None = None
    ) -> StateWrites:
        """Return the ``{variable: value}`` writes that drive *state*.

        Parameters
        ----------
        state : TriggerState or str
            The target state.
        variant : str, optional
            Name of a variant to overlay on the base writes.

        Returns
        -------
        dict of str to str
            The merged writes (base state writes, overridden/extended by the
            variant's writes for that state).

        Raises
        ------
        KeyError
            If *variant* is given but not defined on this profile.
        """
        key = TriggerState(state)
        writes = dict(self.states.get(key, {}))
        if variant is not None:
            if variant not in self.variants:
                raise KeyError(
                    f"Trigger profile {self.name!r} has no variant "
                    f"{variant!r}. Known variants: {sorted(self.variants)}"
                )
            writes.update(self.variants[variant].states.get(key, {}))
        return writes

    def defines_state(
        self, state: "TriggerState | str", variant: str | None = None
    ) -> bool:
        """Whether driving to *state* would write anything at all.

        Parameters
        ----------
        state : TriggerState or str
            The state to query.
        variant : str, optional
            Name of a variant to overlay before deciding.

        Returns
        -------
        bool
            ``True`` if at least one write exists for the state (matching the
            legacy ``ShotControlConfig.defines_state`` semantics).
        """
        return bool(self.writes_for(state, variant))
