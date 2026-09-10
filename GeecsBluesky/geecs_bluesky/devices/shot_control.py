"""ShotControl — the trigger box as an ophyd-async device.

One device, the protocols Bluesky already has for it
(``Planning/native_bluesky/03_clean_room_rebuild.md`` §4.A):

- **Movable** over the profile's named states: ``bps.mv(shot_control,
  "ARMED")`` replays that state's ordered ``(device, variable, value)``
  writes, each completing before the next (the TriggerProfile semantics).
  ``"SINGLESHOT"`` is the momentary fire — a move that never becomes the
  standing state.
- **Pausable**, state-dependent (§10.3): in strict mode the RunEngine
  pausing simply stops the plan firing and the box stays ``ARMED``, so
  ``pause()`` does nothing; in gated mode edges flow on their own, so
  ``pause()`` drives ``OFF`` and ``resume()`` restores ``SCAN``.  The
  RunEngine calls both on every Pausable it has seen in a message
  (bluesky 1.15.0 ``run_engine.py``), so being the ``set`` target is
  enough to be paused.

The write machinery is :class:`~geecs_bluesky.shot_controller.ShotController`
(``from_writes``: one cached gateway ``:SP`` put per distinct target, the
hardware-proven stringified-wire convention) — composed, not copied.  The
standing state is a config signal, so every descriptor records which state
the box was in.
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

from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.models.shot_control import ShotControlWrites
from geecs_bluesky.shot_controller import ShotController

logger = logging.getLogger(__name__)


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
        self._controller = ShotController.from_writes(
            writes,
            experiment=experiment,
            put_timeout=put_timeout,
            setter_factory=setter_factory,
        )
        with self.add_children_as_readables(StandardReadableFormat.CONFIG_SIGNAL):
            # The last *standing* state driven (never SINGLESHOT); "" until
            # the first move.
            self.state, self._set_state = soft_signal_r_and_setter(str, "")
        self._resume_to: TriggerState | None = None
        super().__init__(name=name)

    @classmethod
    def from_profile(
        cls, profile: TriggerProfile, *, experiment: str | None = None, **kwargs: Any
    ) -> ShotControl:
        """Build from a TriggerProfile (the configs-repo document)."""
        # Deferred: the adapter lives with the request runner until the plan
        # layer relocates it (phase 1); importing it there is the one copy.
        from geecs_bluesky.scan_request_runner import trigger_writes_from_profile

        return cls(
            trigger_writes_from_profile(profile), experiment=experiment, **kwargs
        )

    @property
    def profile_name(self) -> str:
        """The adapted profile's name (log/error messages)."""
        return self._writes.name

    def defines(self, state: str | TriggerState) -> bool:
        """Whether the profile writes anything for *state*."""
        return self._controller.defines_state(_state(state).value)

    @property
    def standing_state(self) -> str:
        """The last standing state driven, ``""`` before the first move."""
        return self._standing

    _standing: str = ""

    @AsyncStatus.wrap
    async def set(self, value: str | TriggerState) -> None:
        """Drive the box to *value*: that state's writes, in order."""
        await self._drive(_state(value))

    async def _drive(self, state: TriggerState) -> None:
        setters = self._controller.state_setters(state.value)
        if not setters:
            raise GeecsConfigurationError(
                f"trigger profile {self.profile_name!r} defines no writes for "
                f"{state.value}; it cannot drive the box there"
            )
        for setter, value in setters:
            await setter.put(value)
        if state is not TriggerState.SINGLESHOT:
            self._standing = state.value
            self._set_state(state.value)

    async def pause(self) -> None:
        """Stop edges on a RunEngine pause only if they flow on their own (SCAN)."""
        if self._standing == TriggerState.SCAN.value:
            self._resume_to = TriggerState.SCAN
            logger.info("%s: pause — SCAN → OFF", self.name)
            await self._drive(TriggerState.OFF)

    async def resume(self) -> None:
        """Restore, on RunEngine resume, the state ``pause`` left."""
        if self._resume_to is not None:
            state, self._resume_to = self._resume_to, None
            logger.info("%s: resume — → %s", self.name, state.value)
            await self._drive(state)


def _state(value: str | TriggerState) -> TriggerState:
    if isinstance(value, Enum):
        return TriggerState(value.value)
    return TriggerState(str(value).upper())
