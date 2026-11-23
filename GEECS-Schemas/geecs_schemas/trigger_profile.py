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
it wants.  Format v2 is the only format: a profile has no ``variants``
block, and a document declaring any other ``schema_version`` is refused.

Scan mode
---------
``use_scan_mode: true`` runs every scan under the profile in GEECS **scan
mode**, the way Master Control drives the devices: the worker announces
the scan to the experiment's devices (``ScanStart``), pushes each shot's
save path to them before the fire, and ends the scan (``ScanEnd``) when the
run closes; a device then acquires once per announced shot, saves
shot-numbered files and applies its own database start/end values.  The
profile's writes stay what they are — the box's states — and a profile
with the flag and **no writes at all** is the box-less case (the shot
message is the only trigger).  Off by default; strict acquisition only.
The worker side is GeecsBluesky's ``devices/scan_mode.py``.

Shot gate
---------
``shot_gate`` names a facility's software interlock server: a TCP
server that checks hardware and broadcasts its verdict one way.  Under
such a profile the worker waits, before every SINGLESHOT it issues, for
a safe verdict on data taken after the previous fire, and refuses a run
whose gate does not connect.  A
permission to fire, never a safety system.  Absent by default; a profile
for a condition that needs no gate (laser off, a bench) leaves it out.
The worker side is GeecsBluesky's ``shot_gate.py``.

Developer notes
---------------
Successor of the shot-control YAML once validated by the retired
``ShotControlConfig`` (the engine now consumes this profile as
``geecs_bluesky.devices.shot_control.ShotControlWrites``).  Semantics are
kept, not contradicted:

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

from typing import ClassVar, Literal

from enum import Enum

from pydantic import model_validator, Field, field_validator

from geecs_schemas._base import SchemaModel, Tier, VersionedSchemaModel

#: Where an experiment's trigger profiles live: ``shot_control_configurations/<name>.yaml``.
TRIGGER_PROFILES_FOLDER = "shot_control_configurations"


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


class ShotGateSpec(SchemaModel):
    """A software interlock server the worker waits on before every shot.

    The server broadcasts its interlocks' verdict one way over TCP, a frame
    on connect and then every ~0.1 s.  Every SINGLESHOT waits for a safe
    verdict on data taken after the previous fire (each interlock's
    ``data_stamp``); a verdict carrying no data stamp counts once it is
    received after the fire.  The server accepts more than one client (the
    worker and the web scanner's chip each hold a connection).
    """

    host: str = Field(
        min_length=1,
        description="The service's host name or IP address on the facility network.",
    )
    port: int = Field(ge=1, le=65535, description="The service's TCP port.")
    hold_pause_s: float = Field(
        default=60.0,
        gt=0,
        description=(
            "Seconds a shot waits for permission before the scan pauses with "
            "the service's reason (resuming waits again; a pause never fires)."
        ),
    )
    holdoff_s: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Seconds added to the time a shot is fired before a safe verdict "
            "counts as the next shot's permission: its data must be taken "
            "that much after the fire (or, for a verdict with no data stamp, "
            "received that much after it). Covers a device clock ahead of "
            "the worker's, and, for interlocks with no data stamp, the time "
            "the server takes to see a shot. A device clock behind the "
            "worker's cannot be covered here: the clocks must agree."
        ),
    )

    @property
    def key(self) -> tuple[str, int]:
        """``(host, port)``: what makes two profiles name the same server."""
        return (self.host, self.port)

    @property
    def address(self) -> str:
        """``host:port``."""
        return "{}:{}".format(*self.key)


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

    Format v2 is the only format accepted: ``schema_version`` must be 2 (or
    absent), and an unknown key such as the removed ``variants`` overlay
    is refused like any other.
    """

    tier: ClassVar[Tier] = "operator"

    schema_version: Literal[2] = Field(
        default=2,
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
        default="",
        description="Optional note about what setup this profile is for.",
    )
    use_scan_mode: bool = Field(
        default=False,
        description=(
            "Run scans under this profile in GEECS scan mode, as Master "
            "Control drives the devices: the worker announces the scan to "
            "the experiment's devices (ScanStart), pushes each shot's save "
            "path to them before the fire, and ends the scan (ScanEnd) when "
            "the run closes. A device then acquires once per announced shot, "
            "saves shot-numbered files and applies its own database "
            "start/end values. Off by default. Strict acquisition only: a "
            "fly run and the calibration plans refuse such a profile. A "
            "profile with this flag and no writes at all is the box-less "
            "case — the shot message is the only trigger."
        ),
    )

    shot_gate: ShotGateSpec | None = Field(
        default=None,
        description=(
            "A permission-to-fire service the worker waits on before every "
            "shot: a 'safe' received after the previous shot permits the "
            "next; unreachable or not safe holds it, and a long hold pauses "
            "the scan. Leave empty for no gate."
        ),
    )

    _fix_off_key = field_validator("states", mode="before")(_normalize_state_keys)
    _no_duplicates = field_validator("states")(_reject_duplicate_targets)

    @model_validator(mode="after")
    def _gate_needs_a_fire(self) -> "TriggerProfile":
        """A shot gate holds the SINGLESHOT put, so the profile must define one."""
        if self.shot_gate is not None and not self.defines_state(
            TriggerState.SINGLESHOT
        ):
            raise ValueError(
                f"trigger profile {self.name!r} names a shot gate but defines "
                "no SINGLESHOT writes — the gate holds the single-shot fire, "
                "and this profile issues none (a box-less profile's shot is "
                "the scan-mode announce)."
            )
        return self

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
