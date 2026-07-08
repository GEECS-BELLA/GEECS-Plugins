"""CaActionSignalFactory — the production SettableFactory for action plans.

The ActionPlan compiler (:mod:`geecs_bluesky.plans.action_compiler`) is
transport-ignorant: it asks an injected factory for signals by GEECS
``(device, variable)`` name.  This module is the CA-backed production
implementation:

- ``get_settable(device, variable)`` → a Movable whose ``set()`` puts the
  value (coerced to its wire string) to the variable's gateway setpoint PV
  (``[Experiment:]Device:Variable:SP`` via :func:`~geecs_bluesky.devices.ca._pv.ca_pv`).
  The gateway forwards ``:SP`` puts to GEECS's blocking UDP set and only
  completes the CA put when GEECS accepts (or rejects) it — so CA
  put-completion *is* the legacy ``wait_for_execution`` semantics: an
  ``abs_set(..., wait=True)`` blocks exactly as ``device.set(var, value,
  sync=True)`` did.
- ``get_readable(device, variable)`` → a string-typed read signal on the
  streamed readback PV.  Reading as a string matches the legacy check
  pipeline (:func:`~geecs_bluesky.plans.action_compiler.values_match` floats
  numeric-looking strings, exactly like ``GeecsDevice.interpret_value``).

Signals are created lazily, **cached per (device, variable)** so repeated
steps and per-step plans reuse one CA channel, and connected through the
session's RE-loop connector at creation time.  Because plan generators
execute *inside* the RunEngine's event loop (where a blocking connect would
deadlock), callers must pre-connect every signal a compiled plan will touch
before handing the plan to the RE — see
:func:`~geecs_bluesky.scan_request_runner.prefetch_action_signals`.

The factory rides the same per-scan cleanup path as devices: it exposes an
``async disconnect()`` so ``session.disconnect(factory)`` treats it
uniformly (the signals hold no persistent monitor subscriptions, so there is
nothing to tear down beyond dropping the cache).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ophyd_async.epics.core import epics_signal_r, epics_signal_rw

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)

__all__ = ["CaActionSignalFactory"]


class _WireSettable:
    """Movable adapter: coerces action values to wire strings before the put.

    Action YAML values are ``str | float | int``; the gateway's ``:SP``
    channels accept the string form of any of them (enum PVs take labels,
    numeric PVs coerce numeric strings — the same convention as
    :class:`~geecs_bluesky.shot_controller.CaPutSetter`).  The underlying
    signal is string-typed, so every value is stringified here, once.
    """

    def __init__(self, signal: Any) -> None:
        self._signal = signal

    @property
    def name(self) -> str:
        """Signal name (used by Bluesky logging/message repr)."""
        return self._signal.name

    def set(self, value: Any):
        """Put ``str(value)``; the returned status completes with the GEECS set."""
        return self._signal.set(str(value))


class CaActionSignalFactory:
    """Hands out cached, connected CA signals for action set/check steps.

    Parameters
    ----------
    experiment : str or None
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    connect : callable
        ``connect(signal)`` — connects an ophyd-async signal in the
        RunEngine's persistent event loop (mock-aware); typically the
        session's ``_connect``.  Called once per new signal, at creation.
    """

    def __init__(self, experiment: str | None, connect: Callable[[Any], Any]) -> None:
        self._experiment = experiment
        self._connect = connect
        self._settables: dict[tuple[str, str], _WireSettable] = {}
        self._readables: dict[tuple[str, str], Any] = {}

    def get_settable(self, device: str, variable: str) -> _WireSettable:
        """Return the (cached) setpoint writer for ``(device, variable)``.

        Parameters
        ----------
        device : str
            GEECS device name (e.g. ``"U_148_PLC"``).
        variable : str
            Writable variable on that device (e.g. ``"DO.Ch9"``).

        Returns
        -------
        _WireSettable
            A Movable putting wire strings to the variable's ``:SP`` PV;
            put-completion rides the GEECS blocking set.
        """
        key = (device, variable)
        settable = self._settables.get(key)
        if settable is None:
            pv = f"{ca_pv(self._experiment, device, variable)}:SP"
            signal = epics_signal_rw(str, pv, name=safe_name(f"{device}_{variable}_sp"))
            self._connect(signal)
            settable = _WireSettable(signal)
            self._settables[key] = settable
            logger.debug("action settable created: %s -> %s", key, pv)
        return settable

    def get_readable(self, device: str, variable: str) -> Any:
        """Return the (cached) string readback signal for ``(device, variable)``.

        Parameters
        ----------
        device : str
            GEECS device name (e.g. ``"U_GaiaSVEReader"``).
        variable : str
            Variable to read (e.g. ``"InternalShutterA"``).

        Returns
        -------
        SignalR
            A string-typed read signal on the streamed readback PV,
            accepted by ``bps.rd``.
        """
        key = (device, variable)
        signal = self._readables.get(key)
        if signal is None:
            pv = ca_pv(self._experiment, device, variable)
            signal = epics_signal_r(str, pv, name=safe_name(f"{device}_{variable}"))
            self._connect(signal)
            self._readables[key] = signal
            logger.debug("action readable created: %s -> %s", key, pv)
        return signal

    async def disconnect(self) -> None:
        """Per-scan teardown hook (rides ``session.disconnect`` uniformly).

        The factory's signals hold no persistent monitor subscriptions —
        aioca manages the underlying CA channels globally — so teardown is
        just dropping the cache.
        """
        self._settables.clear()
        self._readables.clear()
