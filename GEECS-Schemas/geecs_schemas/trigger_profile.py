"""TriggerProfile — how the machine is driven through its named trigger states.

A trigger profile describes *machine* states — OFF, STANDBY, SCAN,
SINGLESHOT, ARMED — and exactly which device writes put the machine into
each one.  A state transition may touch several devices (the delay generator
that fires the trigger, a gas-jet controller, a shutter …); each state lists
its writes **in the order they are sent**.  You would edit a profile when
timing hardware changes, when a state needs an extra write, or when a new
device joins the transition.  A scan picks a profile by name in its
:class:`~geecs_schemas.scan_request.ScanRequest`.

One operating condition, one profile
------------------------------------
Operating conditions (the classic laser-on / laser-off pair, no-gas, …) are
separate profile files, each complete on its own, and a scan names the one
it wants.  Format v1 also allowed named *variants* inside a profile (an
overlay of a few writes); nobody adopted them — every experiment kept
separate files — so format v2 removed them.  A v1 file with an empty
``variants`` block still loads (the block is dropped); one that actually
defines a variant is refused with the remedy: split it into its own
profile.

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

from pydantic import model_validator, Field, field_validator

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


class TriggerProfile(VersionedSchemaModel):
    """The device writes that drive the machine through its trigger states.

    For each state (OFF, STANDBY, SCAN, SINGLESHOT, ARMED) list the writes —
    possibly to several devices — that put the machine into it, **in the
    order they should be sent**.  Edit it when timing hardware or its
    settings change; make a separate profile for a different operating
    condition (e.g. laser off).

    Notes
    -----
    Values are sent verbatim over the GEECS wire protocol.  Use
    :meth:`writes_for` / :meth:`defines_state` instead of digging the lists —
    they implement the "state with no writes is not defined" rule.
    :attr:`devices` lists every device the profile touches.

    Format v2 removed the v1 ``variants`` overlay (see the module
    docstring); a before-validator drops an empty v1 block, refuses a
    populated one, and normalizes ``schema_version`` ≤ 1 to 2.
    """

    schema_version: int = Field(
        2,
        description=(
            "Format version of this config file. Leave at 2 — tools update "
            "this automatically when the file format changes."
        ),
    )

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
    description: str = Field(
        "",
        description="Optional note about what setup this profile is for.",
    )

    _fix_off_key = field_validator("states", mode="before")(_normalize_state_keys)
    _no_duplicates = field_validator("states")(_reject_duplicate_targets)

    @model_validator(mode="before")
    @classmethod
    def _lift_v1_layout(cls, data: object) -> object:
        """Drop the removed v1 ``variants`` block; normalize the version.

        Parameters
        ----------
        data : object
            The raw input; non-mapping input passes through untouched.

        Returns
        -------
        object
            The (copied) mapping in v2 layout, or *data* unchanged.

        Raises
        ------
        ValueError
            If the document defines a variant — format v2 has no overlay;
            the remedy is a separate profile per operating condition.
        """
        if not isinstance(data, dict):
            return data
        version = data.get("schema_version")
        if isinstance(version, str) and version.isdigit():
            version = int(version)
        stale = isinstance(version, int) and version <= 1
        if "variants" not in data and not stale:
            return data
        lifted = dict(data)
        variants = lifted.pop("variants", None)
        if variants:
            raise ValueError(
                f"trigger profile {lifted.get('name', '?')!r} defines variant(s) "
                f"{sorted(variants)} — profile variants were removed in format "
                "v2; save each operating condition as its own profile file "
                "and name that profile in the scan request."
            )
        if stale:
            lifted["schema_version"] = 2
        return lifted

    @property
    def devices(self) -> list[str]:
        """Every device this profile writes, in order of first appearance.

        Returns
        -------
        list of str
            Distinct device names across all states.
        """
        seen: dict[str, None] = {}
        for writes in self.states.values():
            for write in writes:
                seen.setdefault(write.device)
        return list(seen)

    def writes_for(self, state: "TriggerState | str") -> StateWrites:
        """Return the ordered writes that drive *state*.

        Parameters
        ----------
        state : TriggerState or str
            The target state.

        Returns
        -------
        list of TriggerWrite
            The transition's writes in their declared order (empty when the
            state is not defined).
        """
        return list(self.states.get(TriggerState(state), []))

    def defines_state(self, state: "TriggerState | str") -> bool:
        """Whether driving to *state* would write anything at all.

        Parameters
        ----------
        state : TriggerState or str
            The state to query.

        Returns
        -------
        bool
            ``True`` if at least one write exists for the state (matching the
            legacy ``ShotControlConfig.defines_state`` semantics).
        """
        return bool(self.writes_for(state))
