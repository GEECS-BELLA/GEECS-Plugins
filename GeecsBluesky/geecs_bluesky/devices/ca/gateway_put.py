"""GatewaySetpointPut — the one blessed gateway ``:SP`` put primitive.

Owns the ca://-vs-bare addressing rule (:func:`bare_pv` — a schemed name
hangs raw aioca; issue #490), the wire-value conventions, the timeout policy,
the ``AsyncStatus`` wrapping, and mock support.  Every setpoint pathway
delegates here: ``CaSettable``/``CaMotor``'s Layer-1 signal put,
``ShotController``'s ``CaPutSetter``, and the action factory's wire settable.

Two transports, one policy owner:

- **raw CA** (``setpoint_pv=…``) — ``aioca.caput(bare_name, wire, wait=True,
  timeout=…)``.  Names are normalized by :func:`bare_pv`: the ophyd ``ca://``
  scheme is stripped (aioca treats a scheme as part of the PV name), any other
  scheme is rejected.  The gateway forwards ``:SP`` puts to GEECS's blocking
  UDP set and only completes the CA put when GEECS accepts (or rejects) it —
  so put-completion *is* the legacy ``wait_for_execution`` semantics.
- **ophyd signal** (``signal=…``) — delegates to ``signal.set(...)`` on a
  typed ``epics_signal_rw`` built from the ``ca://`` URI form (ophyd parses
  the scheme itself; see :mod:`geecs_bluesky.devices.ca._pv`).  Used by
  ``CaSettable``/``CaMotor`` Layer 1, whose typed signal doubles as the
  connect-time dtype check and the mock-backend seam
  (``tests/ca_mock_helpers.follow_setpoint``).

Wire-value conventions (the ``coerce`` parameter — each consumer's pinned,
hardware-proven convention; do not "unify" them without live verification):

- ``str`` — everything stringified: the ShotController convention (enum
  labels pass through; the gateway's typed channel coerces numeric strings).
- :func:`wire_value` — native numerics, wire string otherwise: the
  action-plan convention (a *string* put-with-callback to a float gateway
  channel can hang; observed live 2026-07-10, issue #490).
- ``None`` — pass through untouched: the typed-signal (motor) convention.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ophyd_async.core import AsyncStatus

from geecs_bluesky.devices.ca._pv import CA_TRANSPORT_PREFIX

logger = logging.getLogger(__name__)

__all__ = ["GatewaySetpointPut", "bare_pv", "wire_value"]


def bare_pv(pv: str) -> str:
    """Normalize *pv* for raw-aioca use: strip ``ca://``, reject other schemes.

    ophyd strips the ``ca://`` scheme before its backend stores the PV; raw
    aioca does **not** — it treats the scheme as part of the name, so a
    schemed put CA-searches for a PV nothing serves and hangs for the full
    timeout (issue #490).  This is the one place that rule lives.

    Parameters
    ----------
    pv : str
        A gateway PV name, bare (``expt:dev:var:SP``) or in the ophyd
        signal-URI form (``ca://expt:dev:var:SP``).

    Returns
    -------
    str
        The bare EPICS name.

    Raises
    ------
    ValueError
        If a scheme other than ``ca://`` remains — a raw CA put can never
        address it.
    """
    name = pv.removeprefix(CA_TRANSPORT_PREFIX)
    if "://" in name:
        raise ValueError(
            f"raw CA put needs a bare EPICS name or a ca:// URI, got {pv!r} "
            "(aioca treats a scheme as part of the PV name — issue #490)"
        )
    return name


def wire_value(value: Any) -> Any:
    """Action-plan wire convention: native numerics, wire string otherwise.

    Numbers go natively (DBR_DOUBLE — a string put-with-callback to a float
    gateway channel can hang; observed live 2026-07-10); strings (enum
    labels, ``'on'``/``'off'``) go as the wire string, CA-converted to the
    PV's native type server-side.
    """
    return value if isinstance(value, (int, float)) else str(value)


class GatewaySetpointPut:
    """Movable putting one value to a gateway setpoint PV, GEECS-blocking.

    Parameters
    ----------
    setpoint_pv : str, optional
        Raw-CA transport: the ``:SP`` PV name — bare or in the ``ca://`` URI
        form, normalized by :func:`bare_pv`.  Exactly one of ``setpoint_pv``
        / ``signal``.
    signal : SignalRW, optional
        Ophyd-signal transport: an already-built typed ``:SP`` signal; puts
        delegate to ``signal.set()`` (mock-aware via ``connect(mock=True)``).
    coerce : callable, optional
        ``value → wire value``, applied once per put; ``None`` passes the
        value through untouched.  See the module docstring for the pinned
        conventions.
    timeout : float, optional
        Default per-put budget in seconds.  ``None`` (signal transport only)
        defers to the signal's own default.
    name : str
        Movable name (Bluesky logging / message repr).
    mock : bool
        Raw transport only: record puts on ``last_mock_put`` (as the string
        form of the wire value) instead of touching CA.
    """

    def __init__(
        self,
        setpoint_pv: str | None = None,
        *,
        signal: Any = None,
        coerce: Callable[[Any], Any] | None = None,
        timeout: float | None = 10.0,
        name: str = "",
        mock: bool = False,
    ) -> None:
        if (setpoint_pv is None) == (signal is None):
            raise ValueError("exactly one of setpoint_pv / signal is required")
        if signal is not None and mock:
            raise ValueError(
                "mock puts belong to the raw-CA transport; a signal-backed "
                "put mocks through the signal's own connect(mock=True)"
            )
        if signal is None and timeout is None:
            raise ValueError("the raw-CA transport requires a put timeout")
        self._pv = bare_pv(setpoint_pv) if setpoint_pv is not None else None
        self._signal = signal
        self._coerce = coerce
        self._timeout = timeout
        self.name = name
        self._mock = mock
        self.last_mock_put: str | None = None

    async def put(self, value: Any, timeout: float | None = None) -> None:
        """Put *value*; returns when the gateway completes the GEECS set.

        Parameters
        ----------
        value : Any
            The value to write; ``coerce`` is applied first.
        timeout : float, optional
            Per-put override of the constructor's budget (e.g. a motor's
            ``move_timeout`` — a slow axis is not a dead one).
        """
        wire = self._coerce(value) if self._coerce is not None else value
        budget = self._timeout if timeout is None else timeout
        if self._mock:
            self.last_mock_put = str(wire)
            return
        if self._signal is not None:
            if budget is None:
                await self._signal.set(wire)
            else:
                await self._signal.set(wire, timeout=budget)
            return
        from aioca import caput  # deferred: needs the `ca` extra

        await caput(self._pv, wire, wait=True, timeout=budget)

    def set(self, value: Any) -> AsyncStatus:
        """Movable: put *value*; the status completes with the GEECS set."""
        return AsyncStatus(self.put(value))
