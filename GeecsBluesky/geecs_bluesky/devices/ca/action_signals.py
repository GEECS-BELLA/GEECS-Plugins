"""CaActionSignalFactory — the production SettableFactory for action plans.

The ActionPlan compiler (:mod:`geecs_bluesky.plans.action_compiler`) is
transport-ignorant: it asks an injected factory for signals by GEECS
``(device, variable)`` name.  This module is the CA-backed production
implementation:

- ``get_settable(device, variable)`` → a Movable whose ``set()`` puts the
  value's **wire string via raw CA** (the hardware-proven
  :class:`~geecs_bluesky.shot_controller.CaPutSetter` convention: CA
  converts the string to the PV's native type server-side, so the put works
  on float, enum, and string ``:SP`` channels alike — a *typed* ophyd
  signal connect-fails when the PV's inferred type disagrees).  The gateway
  forwards ``:SP`` puts to GEECS's blocking UDP set and only completes the
  CA put when GEECS accepts (or rejects) it — so put-completion *is* the
  legacy ``wait_for_execution`` semantics.  A dtype-inferred probe signal
  is still connected at creation, preserving the pre-claim fail-fast (a
  missing PV fails before any scan number is claimed).
- ``get_readable(device, variable)`` → a **native-typed** read signal
  (dtype inferred from the PV, like telemetry) on the streamed readback.
  :func:`~geecs_bluesky.plans.action_compiler.values_match` handles both
  string and numeric actuals, quirks preserved.

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

from ophyd_async.core import AsyncStatus
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)

__all__ = ["CaActionSignalFactory"]


class _WireSettable:
    """Movable: puts the wire string to the ``:SP`` PV via raw CA.

    Action YAML values are ``str | float | int``; CA converts the string
    form to the PV's native type (enum labels, numeric strings) — the same
    convention as :class:`~geecs_bluesky.shot_controller.CaPutSetter`, and
    the reason this is a raw ``caput`` rather than a typed ophyd signal
    (which connect-fails on a float ``:SP``).  In mock mode the put is
    recorded on ``last_mock_put`` and completes immediately.
    """

    def __init__(
        self, pv: str, name: str, *, mock: bool = False, timeout: float = 30.0
    ) -> None:
        self._pv = pv
        self.name = name
        self._mock = mock
        self._timeout = timeout
        self.last_mock_put: str | None = None

    def set(self, value: Any) -> AsyncStatus:
        """Put ``str(value)``; the returned status completes with the GEECS set."""
        wire = str(value)

        if self._mock:

            async def _record() -> None:
                self.last_mock_put = wire

            return AsyncStatus(_record())

        async def _do() -> None:
            from aioca import caput  # deferred: needs the `ca` extra

            await caput(self._pv, wire, wait=True, timeout=self._timeout)

        return AsyncStatus(_do())


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

    def __init__(
        self,
        experiment: str | None,
        connect: Callable[[Any], Any],
        *,
        mock: bool = False,
    ) -> None:
        self._experiment = experiment
        self._connect = connect
        self._mock = mock
        self._settables: dict[tuple[str, str], _WireSettable] = {}
        self._readables: dict[tuple[str, str], Any] = {}
        self._probes: dict[tuple[str, str], Any] = {}

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
            name = safe_name(f"{device}_{variable}_sp")
            # dtype-inferred probe: proves the PV exists/connects (pre-claim
            # fail-fast) without imposing a type the PV may not have.
            probe = epics_signal_r(None, pv, name=f"{name}_probe")
            self._connect(probe)
            self._probes[key] = probe
            settable = _WireSettable(pv, name, mock=self._mock)
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
            A native-typed (dtype-inferred) read signal on the streamed
            readback PV, accepted by ``bps.rd``.
        """
        key = (device, variable)
        signal = self._readables.get(key)
        if signal is None:
            pv = ca_pv(self._experiment, device, variable)
            signal = epics_signal_r(None, pv, name=safe_name(f"{device}_{variable}"))
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
        self._probes.clear()
