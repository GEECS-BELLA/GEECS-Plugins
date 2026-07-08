"""TriggerProfile — how the machine is driven through its named trigger states.

A trigger profile describes *machine* states — OFF, STANDBY, SCAN,
SINGLESHOT, ARMED — and exactly which device writes put the machine into
each one.  A state transition may touch several devices (the delay generator
that fires the trigger, a gas-jet controller, a shutter …); each state lists
its writes **in the order they are sent**.  You would edit a profile when
timing hardware changes, when a state needs an extra write, or when a new
device joins the transition.  A scan picks a profile by name in its
:class:`~geecs_schemas.scan_request.ScanRequest`.

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

- The layout pivots from one implicit device with per-variable
  ``{variable: {state: value}}`` tables to per-state **ordered write lists**
  ``[{device, variable, value}, ...]``.  The old single-device shape was an
  accident of the DG645 carrying everything (gas jet gated by its Ch AB
  amplitude); conceptually the states are machine states and a transition
  may write several devices.
- **Order matters within a transition**: writes are applied top to bottom
  (e.g. raise an amplitude before switching a trigger source).  A state may
  write each (device, variable) at most once.
- Values are **verbatim wire strings** ("4.0", "on", enum labels); nothing is
  coerced.
- The legacy empty-string "no-op for this state" convention is retired:
  a no-op is expressed by *omitting* the write from the state (matching
  ``ShotControlConfig.values_for_state``, which skipped empty strings).
  Empty-string values are rejected.
- A state with no writes is "not defined" for this profile, exactly like
  ``ShotControlConfig.defines_state``.

The model is deliberately **device-agnostic**: shot-control devices are
usually DG645 delay generators, but nothing here assumes that — any devices
with settable variables can take part in a transition.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, field_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel


class TriggerState(str, Enum):
    """The named states the machine trigger can be driven to.

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


class TriggerWrite(SchemaModel):
    """One device variable set during a state transition.

    A transition is an ordered list of these; they are sent top to bottom.
    """

    device: str = Field(
        min_length=1,
        description=(
            "The device to write to, e.g. 'U_DG645_ShotControl' or a gas-jet "
            "controller — any settable device can take part in a transition."
        ),
    )
    variable: str = Field(
        min_length=1,
        description="Which variable on the device to set, e.g. 'Trigger.Source'.",
    )
    value: str = Field(
        description=(
            "The value to send, exactly as the device expects it — a number "
            "as text ('4.0'), a word ('on'), or a device option name "
            "('External rising edges')."
        ),
    )

    @field_validator("value")
    @classmethod
    def _no_empty_value(cls, value: str) -> str:
        """Reject empty-string writes (legacy no-op convention is retired).

        Parameters
        ----------
        value : str
            The proposed wire value.

        Returns
        -------
        str
            The validated value, unchanged.

        Raises
        ------
        ValueError
            If the value is an empty string.
        """
        if value == "":
            raise ValueError(
                "A write's value must not be empty. To leave a variable "
                "untouched in a state, omit the write from that state "
                "instead."
            )
        return value


# One state's transition: the writes to send, in order, top to bottom.
StateWrites = list[TriggerWrite]


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


def _reject_duplicate_targets(states: dict) -> dict:
    """Reject a state that writes the same (device, variable) twice.

    Parameters
    ----------
    states : dict
        Mapping of state → ordered write list (already model-validated).

    Returns
    -------
    dict
        The validated mapping, unchanged.

    Raises
    ------
    ValueError
        If any state lists two writes to the same device variable — the
        intended value would be ambiguous.
    """
    for state, writes in states.items():
        seen: set[tuple[str, str]] = set()
        for write in writes or []:
            target = (write.device, write.variable)
            if target in seen:
                state_name = getattr(state, "value", state)
                raise ValueError(
                    f"State {state_name!r} writes "
                    f"{write.device}:{write.variable} more than once — keep "
                    "one write per device variable per state."
                )
            seen.add(target)
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
            "A write here replaces the base write to the same device "
            "variable; writes to new device variables are added after the "
            "base ones. Anything not listed keeps its base value."
        )
    )
    description: str = Field(
        "",
        description="Optional note about when to use this variant.",
    )

    _fix_off_key = field_validator("states", mode="before")(_normalize_state_keys)
    _no_duplicates = field_validator("states")(_reject_duplicate_targets)


class TriggerProfile(VersionedSchemaModel):
    """The device writes that drive the machine through its trigger states.

    For each state (OFF, STANDBY, SCAN, SINGLESHOT, ARMED) list the writes —
    possibly to several devices — that put the machine into it, **in the
    order they should be sent**.  Edit it when timing hardware or its
    settings change; add a variant when you need a named alternative
    condition (e.g. laser off) instead of a copy of the whole file.

    Notes
    -----
    Values are sent verbatim over the GEECS wire protocol.  Use
    :meth:`writes_for` / :meth:`defines_state` instead of digging the lists —
    they implement the variant overlay and the "state with no writes is not
    defined" rule.  :attr:`devices` lists every device the profile touches.
    """

    name: str = Field(
        description="The name scans use to refer to this trigger profile."
    )
    states: dict[TriggerState, StateWrites] = Field(
        default_factory=dict,
        description=(
            "For each trigger state, the writes that put the machine into "
            "it, applied in order from top to bottom. A transition may "
            "write several devices. Omit a device variable from a state to "
            "leave it untouched."
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
    _no_duplicates = field_validator("states")(_reject_duplicate_targets)

    @property
    def devices(self) -> list[str]:
        """Every device this profile writes, in order of first appearance.

        Returns
        -------
        list of str
            Distinct device names across all base states and variants.
        """
        seen: dict[str, None] = {}
        for writes in self.states.values():
            for write in writes:
                seen.setdefault(write.device)
        for variant in self.variants.values():
            for writes in variant.states.values():
                for write in writes:
                    seen.setdefault(write.device)
        return list(seen)

    def writes_for(
        self, state: "TriggerState | str", variant: str | None = None
    ) -> StateWrites:
        """Return the ordered writes that drive *state*.

        Parameters
        ----------
        state : TriggerState or str
            The target state.
        variant : str, optional
            Name of a variant to overlay on the base writes.

        Returns
        -------
        list of TriggerWrite
            The merged transition: base writes in their declared order, with
            variant writes replacing same-(device, variable) entries in
            place and any additional variant writes appended after them.

        Raises
        ------
        KeyError
            If *variant* is given but not defined on this profile.
        """
        key = TriggerState(state)
        writes = list(self.states.get(key, []))
        if variant is not None:
            if variant not in self.variants:
                raise KeyError(
                    f"Trigger profile {self.name!r} has no variant "
                    f"{variant!r}. Known variants: {sorted(self.variants)}"
                )
            overlay = {
                (write.device, write.variable): write
                for write in self.variants[variant].states.get(key, [])
            }
            writes = [
                overlay.pop((write.device, write.variable), write) for write in writes
            ]
            writes.extend(overlay.values())
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
