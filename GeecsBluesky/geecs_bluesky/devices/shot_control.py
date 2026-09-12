"""ShotControl — the trigger box as an ophyd-async device.

One device, the protocols Bluesky already has for it
(``Planning/native_bluesky/03_clean_room_rebuild.md`` §4.A):

- **Movable** over the profile's named states: ``bps.mv(shot_control,
  "ARMED")`` replays that state's ordered ``(device, variable, value)``
  writes, each completing before the next (the TriggerProfile semantics).
  ``"SINGLESHOT"`` is the momentary fire — a move that never becomes the
  standing state.
- **Pausable**, keyed on the standing state (§10.3): ``ARMED`` (strict) is
  quiescent by construction — the single-shot source cannot free-run — so
  the RunEngine pausing simply stops the plan firing and ``pause()`` does
  nothing.  ``SCAN`` and ``STANDBY`` both pass external edges
  (:data:`~geecs_bluesky.models.shot_control.QUIESCE_FROM`; §11.1 — STANDBY is the machine's idle state, not a
  quiet one), so a pause there drives ``OFF`` and ``resume()`` restores
  what the plan had.  The RunEngine calls both on every Pausable it has
  seen in a message (bluesky 1.15.0 ``run_engine.py``), so being the
  ``set`` target is enough to be paused.  Neither notification ever raises:
  an exception out of ``pause()`` aborts the run the operator meant to
  pause, and out of ``resume()`` lands after the RunEngine has already
  rewound — failures are logged loudly instead.

The writes go through one cached gateway ``:SP`` put per distinct
``(device, variable)`` target (:class:`~geecs_bluesky.devices.ca.gateway_put.CaPutSetter` — the hardware-proven
stringified-wire convention); each state's list replays in declared order,
every put completing before the next (the TriggerProfile semantics: raise
an amplitude before switching a source).  The ``state`` config signal
mirrors the standing state so every descriptor records which state the box
was in.  :func:`trigger_writes_from_profile` adapts the configs-repo
``TriggerProfile`` into :class:`~geecs_bluesky.models.shot_control.ShotControlWrites`.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

from geecs_schemas.trigger_profile import TriggerProfile, TriggerState
from ophyd_async.core import (
    AsyncStatus,
    StandardReadable,
    StandardReadableFormat,
    soft_signal_r_and_setter,
)

from geecs_bluesky.devices.ca.gateway_put import CaPutSetter
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import QUIESCE_FROM, ShotControlWrites
from geecs_core.pv_naming import pv_name, setpoint_pv

logger = logging.getLogger(__name__)


def _state_write_triples(
    profile: TriggerProfile, state: TriggerState
) -> list[tuple[str | None, str, str]]:
    """Normalize one state's writes to ``(device, variable, value)`` triples.

    Handles both TriggerProfile generations (single-device dict shape and
    multi-device ordered write lists); order is preserved exactly
    (schema-documented: writes apply top to bottom).
    """
    writes = profile.writes_for(state)
    if isinstance(writes, dict):
        device = getattr(profile, "device", None)
        return [(device, variable, value) for variable, value in writes.items()]
    triples: list[tuple[str | None, str, str]] = []
    for write in writes:
        if isinstance(write, dict):
            triples.append((write["device"], write["variable"], write["value"]))
        else:
            triples.append((write.device, write.variable, write.value))
    return triples


def trigger_writes_from_profile(profile: TriggerProfile) -> ShotControlWrites:
    """Adapt a TriggerProfile into the controller's ``ShotControlWrites``.

    Each state becomes the profile's **ordered** write list (possibly
    spanning several devices); the controller replays them sequentially,
    each write completing before the next.

    Raises
    ------
    GeecsConfigurationError
        The profile writes no device at all.
    """
    states: dict[str, list[tuple[str, str, str]]] = {}
    any_device = False
    for state in TriggerState:
        triples: list[tuple[str, str, str]] = []
        for device, variable, value in _state_write_triples(profile, state):
            if device is None:
                raise GeecsConfigurationError(
                    f"trigger profile {profile.name!r} has a write to "
                    f"{variable!r} with no device — it cannot be sent"
                )
            triples.append((device, variable, value))
            any_device = True
        if triples:
            states[state.value] = triples
    if not any_device:
        raise GeecsConfigurationError(
            f"trigger profile {profile.name!r} names no trigger device — "
            "it cannot drive a scan's trigger"
        )
    name = getattr(profile, "name", "") or ""
    return ShotControlWrites(name=name, states=states)


class ShotControl(StandardReadable):
    """The trigger box: ``Movable`` over its named states, ``Pausable``.

    Parameters
    ----------
    writes :
        Per-state ordered write lists (the adapted TriggerProfile).
    experiment :
        Experiment PV-namespace prefix for the gateway ``:SP`` puts.
    name :
        ophyd-async device name.
    put_timeout :
        CA put budget per write (each put completes when GEECS accepts the
        set).
    setter_factory :
        ``factory(device, variable) → object with async put(value)``
        override — tests inject recording setters; hardware uses the
        default gateway puts.
    """

    def __init__(
        self,
        writes: ShotControlWrites,
        *,
        experiment: str | None = None,
        name: str = "",
        put_timeout: float = 10.0,
        setter_factory: Callable[[str, str], Any] | None = None,
    ) -> None:
        self._writes = writes
        factory = setter_factory or (
            lambda device, variable: CaPutSetter(
                setpoint_pv(pv_name(experiment, device, variable)), timeout=put_timeout
            )
        )
        # One setter per distinct target, cached across states; each state's
        # transition is its ordered (setter, value) list.
        setters: dict[tuple[str, str], Any] = {}
        self._transitions: dict[str, list[tuple[Any, str]]] = {}
        for state_name, state_writes in writes.states.items():
            ordered: list[tuple[Any, str]] = []
            for device, variable, value in state_writes:
                key = (device, variable)
                if key not in setters:
                    setters[key] = factory(device, variable)
                ordered.append((setters[key], value))
            if ordered:
                self._transitions[state_name] = ordered
        #: The last *standing* state driven — never the momentary SINGLESHOT
        #: fire (recording it would make a later re-assert refire a shot).
        self._standing: str | None = None
        with self.add_children_as_readables(StandardReadableFormat.CONFIG_SIGNAL):
            # Mirrors the standing state; "" until the first move.
            self.state, self._set_state = soft_signal_r_and_setter(str, "")
        self._resume_to: str | None = None
        #: How many RunEngine pauses this box has seen — a gated step reads it
        #: before and after its batch to learn it was interrupted (§4.2:
        #: an immediate pause mid-batch means the step is retaken).
        self.pause_count = 0
        super().__init__(name=name)

    @classmethod
    def from_profile(
        cls, profile: TriggerProfile, *, experiment: str | None = None, **kwargs: Any
    ) -> ShotControl:
        """Build from a TriggerProfile (the configs-repo document)."""
        return cls(
            trigger_writes_from_profile(profile), experiment=experiment, **kwargs
        )

    @property
    def profile_name(self) -> str:
        """The adapted profile's name (log/error messages)."""
        return self._writes.name

    def defines(self, state: str | TriggerState) -> bool:
        """Whether the profile writes anything for *state* (``False`` for unknown names)."""
        try:
            name = _state(state).value
        except GeecsConfigurationError:
            return False
        return bool(self._transitions.get(name))

    @property
    def standing_state(self) -> str:
        """The last standing state driven, ``""`` before the first move."""
        return self._standing or ""

    @AsyncStatus.wrap
    async def set(self, value: str | TriggerState) -> None:
        """Drive the box to *value*: that state's writes, in order.

        A standing state the plan drives supersedes any pause bookkeeping:
        a run stopped while paused must not make a later, unrelated resume
        re-assert the state it was paused from.
        """
        state = _state(value)
        await self._drive(state)
        if state is not TriggerState.SINGLESHOT:
            self._resume_to = None

    async def _drive(self, state: TriggerState) -> None:
        setters = self._transitions.get(state.value, [])
        if not setters:
            raise GeecsConfigurationError(
                f"trigger profile {self.profile_name!r} defines no writes for "
                f"{state.value}; it cannot drive the box there"
            )
        for setter, value in setters:
            await setter.put(value)
        if state is not TriggerState.SINGLESHOT:
            self._standing = state.value
        self._set_state(self.standing_state)

    async def pause(self) -> None:
        """Stop edges on a RunEngine pause if the standing state lets them flow."""
        self.pause_count += 1
        try:
            standing = self.standing_state
            if standing not in QUIESCE_FROM:
                logger.debug("%s: pause — %r needs no quiesce", self.name, standing)
                return
            if not self.defines(TriggerState.OFF):
                logger.warning(
                    "%s: trigger profile %r defines no OFF writes — paused with "
                    "the trigger still free-running (add an OFF state)",
                    self.name,
                    self.profile_name,
                )
                return
            self._resume_to = standing
            await self._drive(TriggerState.OFF)
            logger.info("%s: pause — %s → OFF", self.name, standing)
        except Exception:
            logger.exception(
                "%s: pause quiesce failed — the scan is paused but the trigger "
                "may still be running",
                self.name,
            )

    async def resume(self) -> None:
        """Restore, on RunEngine resume, the state ``pause`` left."""
        standing, self._resume_to = self._resume_to, None
        if standing is None:
            return
        try:
            await self._drive(TriggerState(standing))
            logger.info("%s: resume — → %s", self.name, standing)
        except Exception:
            logger.exception(
                "%s: could not re-assert %s on resume — check the trigger",
                self.name,
                standing,
            )


def _state(value: str | TriggerState) -> TriggerState:
    if isinstance(value, Enum):
        value = value.value
    try:
        return TriggerState(str(value).upper())
    except ValueError:
        names = ", ".join(s.value for s in TriggerState)
        raise GeecsConfigurationError(
            f"{value!r} is not a trigger state (one of {names})"
        ) from None
