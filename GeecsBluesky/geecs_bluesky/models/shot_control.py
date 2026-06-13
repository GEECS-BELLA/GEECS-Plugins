"""ShotControlConfig — validated DG645 / shot-controller configuration.

The GEECS shot controller (a DG645 delay generator) is configured by a small
YAML document — historically passed around the scanner as a bare, untyped
``dict``::

    device: U_DG645_ShotControl
    variables:
      Trigger.Source:
        OFF: "Single shot external rising edges"
        SCAN: "External rising edges"
        STANDBY: "External rising edges"
        SINGLESHOT: ""
      Trigger.ExecuteSingleShot:
        OFF: ""
        SCAN: ""
        STANDBY: ""
        SINGLESHOT: "on"

Each named **state** maps a set of device variables to the values that put the
controller into that state.  An empty-string value means "no-op for this
state" (leave the variable untouched) — matching the legacy
``TriggerController`` convention.

This module gives that document a typed home so callers validate once and then
ask structured questions (``defines_state``, ``values_for_state``) instead of
digging raw nested dicts.  It is pure data — no hardware, no GEECS engine
imports — so both :class:`~geecs_bluesky.scanner_bridge.BlueskyScanner` and any
future GUI/editor can share it without dragging in the legacy engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ShotControlState(str, Enum):
    """Named states a shot controller can be driven to.

    ``OFF`` / ``SCAN`` / ``STANDBY`` are the legacy trigger-window states;
    ``SINGLESHOT`` is the plan-owned single-shot fire used by strict
    acquisition.  Which states a given controller actually implements depends
    on its YAML — query with :meth:`ShotControlConfig.defines_state`.
    """

    OFF = "OFF"
    SCAN = "SCAN"
    STANDBY = "STANDBY"
    SINGLESHOT = "SINGLESHOT"


class ShotControlConfig(BaseModel):
    """A shot-controller device plus its per-variable state→value table.

    Parameters
    ----------
    device:
        GEECS device name (e.g. ``"U_DG645_ShotControl"``).
    variables:
        ``{variable_name: {state_name: value}}``.  Values are sent verbatim
        over the GEECS wire protocol (they may be words like ``"on"`` or
        device-enum strings).  An empty string means "no-op for this state".
    """

    model_config = ConfigDict(extra="forbid")

    device: str
    variables: dict[str, dict[str, str]] = Field(default_factory=dict)

    @classmethod
    def from_information(
        cls, information: "ShotControlConfig | dict[str, Any] | None"
    ) -> "ShotControlConfig | None":
        """Coerce the legacy ``shot_control_information`` dict (or ``None``).

        Accepts an existing :class:`ShotControlConfig` (returned as-is), the
        ``{"device": ..., "variables": ...}`` dict the GUI/YAML produces, or
        ``None`` (no shot control configured → returns ``None``).
        """
        if information is None:
            return None
        if isinstance(information, ShotControlConfig):
            return information
        return cls.model_validate(information)

    @staticmethod
    def _state_name(state: "ShotControlState | str") -> str:
        return state.value if isinstance(state, ShotControlState) else str(state)

    def defines_state(self, state: "ShotControlState | str") -> bool:
        """Whether any variable has a non-empty value for *state*.

        A state listed everywhere as ``""`` (all no-ops) counts as *not*
        defined — driving to it would do nothing.
        """
        name = self._state_name(state)
        return any(values.get(name) for values in self.variables.values())

    def values_for_state(self, state: "ShotControlState | str") -> dict[str, str]:
        """Return the ``{variable: value}`` writes that drive *state*.

        Variables whose value for this state is missing or empty are omitted
        (no-op), so the result contains only the writes that should actually
        be sent.
        """
        name = self._state_name(state)
        return {
            var: values[name]
            for var, values in self.variables.items()
            if values.get(name)
        }
