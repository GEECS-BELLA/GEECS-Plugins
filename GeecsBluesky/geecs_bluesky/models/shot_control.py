"""ShotControlWrites — the trigger box's per-state ordered write lists.

The engine-side shape of a configs-repo
``geecs_schemas.trigger_profile.TriggerProfile``: a state transition is an
ordered list of ``(device, variable, value)`` writes, applied top to bottom
(order is schema-documented — e.g. raise an amplitude before switching a
trigger source), possibly spanning several devices.  Values are verbatim
wire strings; a state with no writes is "not defined" for this box.  The
adapter lives beside the device that consumes it
(:func:`~geecs_bluesky.devices.shot_control.trigger_writes_from_profile`).

The five state names are the schema's :class:`TriggerState`
(``Planning/native_bluesky/03_clean_room_rebuild.md`` §11.1): OFF is the
only quiet state, STANDBY and SCAN pass external edges, ARMED is the
single-shot source for strict acquisition, SINGLESHOT is the momentary fire.
"""

from __future__ import annotations

from geecs_schemas.trigger_profile import TriggerState
from pydantic import BaseModel, ConfigDict, Field

#: Standing states in which external edges reach the devices, so a RunEngine
#: pause must drive OFF (SCAN and STANDBY both pass edges — STANDBY is the
#: machine's idle state, not a quiet one).  ARMED/OFF are quiescent by
#: construction.  Consumed by the ShotControl device's ``pause()``.
QUIESCE_FROM: frozenset[str] = frozenset(
    {TriggerState.SCAN.value, TriggerState.STANDBY.value}
)


class ShotControlWrites(BaseModel):
    """Per-state **ordered** multi-device write lists.

    Parameters
    ----------
    name:
        Profile name, used in log/error messages (e.g. ``"HTU-NoGas"``).
    states:
        ``{state_name: [(device, variable, value), ...]}`` — the ordered
        writes per state.  Empty-string values are not expected here (the
        TriggerProfile schema rejects them; omission is the no-op).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    states: dict[str, list[tuple[str, str, str]]] = Field(default_factory=dict)

    @staticmethod
    def _state_name(state: TriggerState | str) -> str:
        return state.value if isinstance(state, TriggerState) else str(state)

    @property
    def devices(self) -> list[str]:
        """Every device written, in order of first appearance."""
        seen: dict[str, None] = {}
        for writes in self.states.values():
            for device, _variable, _value in writes:
                seen.setdefault(device)
        return list(seen)

    def defines_state(self, state: TriggerState | str) -> bool:
        """Whether driving to *state* would write anything at all."""
        return bool(self.states.get(self._state_name(state)))

    def writes_for_state(self, state: TriggerState | str) -> list[tuple[str, str, str]]:
        """Return the ordered ``(device, variable, value)`` writes for *state*."""
        return list(self.states.get(self._state_name(state), []))
