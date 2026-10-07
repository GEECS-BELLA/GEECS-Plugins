"""GatewaySetpointPut — the one gateway ``:SP`` put primitive.

Owns PV addressing (:func:`bare_pv`), the wire-value conventions, the
timeout policy, the ``AsyncStatus`` wrapping and mock support.  Every
setpoint pathway delegates here: ``CaSettable``/``CaMotor``,
``ShotControl``'s :class:`CaPutSetter`, and the action factory.

Two transports:

- **raw CA** (``setpoint_pv=…``): ``aioca.caput(bare, wire, wait=True,
  timeout=…)``.  The gateway completes the put only when GEECS accepts or
  rejects the set, so put-completion is the blocking-set semantics.
- **ophyd signal** (``signal=…``): a put through the connected backend of
  a typed ``epics_signal_rw`` (see :mod:`geecs_bluesky.devices.ca._pv`),
  whose type is the connect-time dtype check and the mock seam.  Never
  ``signal.set()``: ophyd-async 0.19 reports a refused put as success
  (see :func:`_backend_put`).

Wire-value conventions (``coerce``), each hardware-proven; do not unify
them without live verification:

- ``str`` — everything stringified (shot control).
- :func:`wire_value` — native numerics, strings otherwise (action plans);
  ``None`` — untouched (the typed-signal motor path).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from ophyd_async.core import DEFAULT_TIMEOUT, AsyncStatus

from geecs_bluesky.devices.ca._pv import CA_TRANSPORT_PREFIX

logger = logging.getLogger(__name__)

__all__ = ["GatewaySetpointPut", "bare_pv", "wire_value"]


def bare_pv(pv: str) -> str:
    """Normalize *pv* for raw-aioca use: strip ``ca://``, reject other schemes.

    ophyd strips the ``ca://`` scheme before its backend stores the PV; raw
    aioca does **not** — it treats the scheme as part of the name, so a
    schemed put CA-searches for a PV nothing serves and hangs for the full
    timeout.  This is the one place that rule lives.

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
    gateway channel can hang); strings (enum labels, ``'on'``/``'off'``) go
    as the wire string, CA-converted to the PV's native type server-side.
    """
    return value if isinstance(value, (int, float)) else str(value)


async def _backend_put(signal: Any, value: Any, timeout: float | None) -> None:
    """Put *value* through *signal*'s connected backend, bounded by *timeout*.

    The signal's own ``set()`` (ophyd-async 0.19.3) re-raises a stored
    failure only if it is truthy, and a failed ``aioca.CANothing`` is
    **falsy**, so ``set()`` reports a refused put as success.  The backend's
    put is the coroutine ``set()`` awaits, with the same bounded wait (mock
    backend included) and ophyd's default budget when *timeout* is ``None``.

    Parameters
    ----------
    signal :
        A connected typed ``:SP`` signal (``epics_signal_rw``).
    value :
        The wire value.
    timeout :
        Seconds; ``None`` means ophyd-async's ``DEFAULT_TIMEOUT``.
    """
    backend = signal._connector.backend
    budget = DEFAULT_TIMEOUT if timeout is None else timeout
    try:
        await asyncio.wait_for(backend.put(value), budget)
    except asyncio.TimeoutError as exc:
        source = backend.source(signal.name, read=False)
        raise TimeoutError(
            f"{source}: put of {value!r} did not complete within {budget} s"
        ) from exc


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
        go through its connected backend (:func:`_backend_put` — mock-aware
        via ``connect(mock=True)``).
    coerce : callable, optional
        ``value → wire value``, applied once per put; ``None`` passes the
        value through untouched.  See the module docstring for the pinned
        conventions.
    timeout : float, optional
        Default per-put budget in seconds.  ``None`` (signal transport only)
        means ophyd-async's ``DEFAULT_TIMEOUT`` (10 s).
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
            ``reply_ceiling`` — minutes: a slow axis is not a dead one).
        """
        wire = self._coerce(value) if self._coerce is not None else value
        budget = self._timeout if timeout is None else timeout
        if self._mock:
            self.last_mock_put = str(wire)
            return
        if self._signal is not None:
            await _backend_put(self._signal, wire, budget)
            return
        from aioca import caput  # deferred: needs the `ca` extra

        await caput(self._pv, wire, wait=True, timeout=budget)

    def set(self, value: Any) -> AsyncStatus:
        """Movable: put *value*; the status completes with the GEECS set."""
        return AsyncStatus(self.put(value))


class CaPutSetter(GatewaySetpointPut):
    """One value to one gateway setpoint PV, as its wire string.

    The gateway's ``:SP`` write forwards to the GEECS UDP set and completes
    only when GEECS accepts (or rejects) it, so put-completion carries the
    same semantics as the direct UDP ACK.  Values go as strings (labels for
    enum PVs; numeric strings are coerced by the gateway's typed channel) —
    the hardware-proven shot-control convention, 10 s default budget,
    pinned byte-for-byte by ``tests/test_gateway_put.py``.
    """

    def __init__(self, setpoint_pv: str, timeout: float = 10.0) -> None:
        super().__init__(setpoint_pv, coerce=str, timeout=timeout)
