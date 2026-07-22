"""Device panel seam (R7): live gateway readback + ``:SP`` setpoint puts.

The protocol and a no-op stub live here alongside the real
:class:`GatewayDevicePanel`: persistent ``aioca.camonitor`` streams on the
readback PV(s) — one per target; ``subscribe_many`` backs the movable
panel's composite selections (values flow to the GUI through a queued Qt
signal owned by ``MovablePanelController``) — and setpoint puts through
:class:`~geecs_bluesky.devices.ca.gateway_put.GatewaySetpointPut` — the one
blessed gateway ``:SP`` put primitive, riding GEECS's native blocking set.
Catalog-level moves (composites, motors, confirm variables) do NOT go
through this backend: the controller routes them to the engine's
``Submitter.move_variable``.

Threading model (mirrors the health-probe precedent — **no QThread**, ever):
``aioca`` is asyncio-based, so the real backend owns ONE persistent asyncio
event loop running ``run_forever`` in a single daemon ``threading.Thread``,
created lazily on first use.  ``camonitor`` / subscription close / ``caput``
are submitted onto that loop via ``asyncio.run_coroutine_threadsafe``.
Nothing here ever blocks the GUI thread: ``subscribe`` / ``unsubscribe``
return immediately, and the blocking ``set`` is dispatched by the window to
a short-lived daemon thread.  Teardown never joins — the loop thread is
daemonic and dies with the process.

Every real dependency (``aioca``, the geecs-bluesky PV helpers) is imported
lazily inside methods, keeping this module import-safe with zero network and
without the ``ca`` extra.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: Default per-put budget (seconds) — the gateway completes the CA put only
#: when GEECS accepts the set, so this is the whole blocking-set budget.
_PUT_TIMEOUT_S = 10.0

#: Extra slack on top of the put budget for the cross-thread future wait.
_DISPATCH_SLACK_S = 5.0


@runtime_checkable
class DevicePanelBackend(Protocol):
    """Anything that can drive the R7 device panel (readback + set)."""

    def subscribe(
        self,
        experiment: str,
        device: str,
        variable: str,
        on_value: Callable[[Any], None],
    ) -> None:
        """Start a readback stream; *on_value* may fire on any thread."""
        ...

    def subscribe_many(
        self,
        experiment: str,
        targets: list[tuple[str, str]],
        on_value: Callable[[int, Any], None],
    ) -> None:
        """Start one readback stream per ``(device, variable)`` target.

        *on_value* is called with ``(target_index, value)`` and may fire on
        any thread.  Replaces any previous subscription (single or many).
        Backs the movable panel's composite (pseudo) selections — one live
        readback per component.
        """
        ...

    def unsubscribe(self) -> None:
        """Stop the current readback stream(s) (no-op when none is active)."""
        ...

    def set(self, experiment: str, device: str, variable: str, value: Any) -> None:
        """Write *value* to the setpoint, blocking until done; raise on failure."""
        ...


class StubDevicePanel:
    """The no-network default: readback never updates, sets report unwired."""

    def subscribe(
        self,
        experiment: str,
        device: str,
        variable: str,
        on_value: Callable[[Any], None],
    ) -> None:
        """No-op: the stub has no readback source.

        Parameters
        ----------
        experiment, device, variable : str
            Ignored.
        on_value : callable
            Never called.
        """
        return None

    def subscribe_many(
        self,
        experiment: str,
        targets: list[tuple[str, str]],
        on_value: Callable[[int, Any], None],
    ) -> None:
        """No-op: the stub has no readback source.

        Parameters
        ----------
        experiment : str
            Ignored.
        targets : list of tuple
            Ignored.
        on_value : callable
            Never called.
        """
        return None

    def unsubscribe(self) -> None:
        """No-op: nothing was subscribed."""
        return None

    def set(self, experiment: str, device: str, variable: str, value: Any) -> None:
        """Refuse the set so the window surfaces a clear offline message.

        Raises
        ------
        RuntimeError
            Always — the stub has no gateway to write to.
        """
        raise RuntimeError("device panel backend not wired (offline stub)")


# ----------------------------------------------------------------------
# Pure helpers (unit-testable without Qt or CA)
# ----------------------------------------------------------------------


def parse_device_variable(text: str) -> Optional[tuple[str, str]]:
    """Split a ``device:variable`` combo entry into its two GEECS names.

    Parameters
    ----------
    text : str
        The R7 combo text, e.g. ``"U_ESP_JetXYZ:Position.Axis 1"``.  The
        split is on the *first* colon — GEECS variable names may contain
        dots and spaces but device names never contain colons.

    Returns
    -------
    tuple of (str, str) or None
        ``(device, variable)`` with surrounding whitespace stripped, or
        ``None`` when the text is not a valid ``device:variable`` pair.
    """
    device, sep, variable = text.partition(":")
    device, variable = device.strip(), variable.strip()
    if not sep or not device or not variable:
        return None
    return device, variable


def parse_set_value(text: str) -> float | str:
    """Interpret the R7 set-field text as a scalar wire value.

    Parameters
    ----------
    text : str
        The raw field text.

    Returns
    -------
    float or str
        A float when the text parses as one (numeric setpoints go natively
        — the :func:`~geecs_bluesky.devices.ca.gateway_put.wire_value`
        convention), otherwise the stripped string (enum labels,
        ``'on'``/``'off'``).
    """
    raw = text.strip()
    try:
        return float(raw)
    except ValueError:
        return raw


def format_readback(value: Any) -> str:
    """Render one readback value for the R7 label, width-stable.

    Parameters
    ----------
    value : Any
        A monitor callback value (aioca augmented values subclass their
        native python type, so plain ``isinstance`` checks apply).

    Returns
    -------
    str
        Floats with **fixed** decimals (4) in the ordinary magnitude range,
        scientific with fixed mantissa digits outside it, everything else
        stringified.  Fixed-width rendering is deliberate: noise in a
        streamed readback's final digits must not change the string length
        (the old 6-significant-digits format made the window width jitter
        at ~1 Hz — owner report, 0.19.1).
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value == 0.0 or 1e-3 <= abs(value) < 1e5:
            return f"{value:.4f}"
        return f"{value:.3e}"
    return str(value)


def readback_pv(experiment: str, device: str, variable: str) -> str:
    """Bare readback PV name for ``(experiment, device, variable)``.

    Built through the blessed geecs-bluesky helpers (lazy import): ``ca_pv``
    applies the gateway naming contract, ``bare_pv`` strips the ``ca://``
    transport scheme raw aioca must never see (issue #490).

    Parameters
    ----------
    experiment : str
        Experiment prefix ("" drops out of the name).
    device, variable : str
        Raw GEECS names; CA-normalized by the naming contract.

    Returns
    -------
    str
        The bare EPICS name, e.g. ``"htu:u_hexapod:ypos"`` (PV components
        are lowercase — the naming contract case-folds, so operator-typed
        case never matters).
    """
    from geecs_bluesky.devices.ca._pv import ca_pv
    from geecs_bluesky.devices.ca.gateway_put import bare_pv

    return bare_pv(ca_pv(experiment or None, device, variable))


def setpoint_pv(experiment: str, device: str, variable: str) -> str:
    """Bare setpoint (``:SP``) PV name for ``(experiment, device, variable)``.

    Parameters
    ----------
    experiment : str
        Experiment prefix ("" drops out of the name).
    device, variable : str
        Raw GEECS names.

    Returns
    -------
    str
        The bare ``…:SP`` EPICS name.
    """
    from geecs_bluesky.devices.ca._pv import setpoint_pv as sp

    return sp(readback_pv(experiment, device, variable))


def format_target_readbacks(targets: list[tuple[str, str]], values: list[Any]) -> str:
    """Render per-target readbacks for a composite selection, compactly.

    Parameters
    ----------
    targets : list of tuple
        The ``(device, variable)`` pairs of a composite's components.
    values : list
        The latest monitor value per target (``None`` = not yet delivered).

    Returns
    -------
    str
        One ``device value`` chunk per target joined with a middle dot —
        device names alone when they are unique (the common composite
        shape), ``device:variable`` when a device appears twice.
    """
    em_dash = "\u2014"
    devices = [device for device, _ in targets]
    labels = [
        device if devices.count(device) == 1 else f"{device}:{variable}"
        for device, variable in targets
    ]
    chunks = [
        f"{label} {format_readback(value) if value is not None else em_dash}"
        for label, value in zip(labels, values)
    ]
    return " \u00b7 ".join(chunks)


# ----------------------------------------------------------------------
# The real backend
# ----------------------------------------------------------------------


class GatewayDevicePanel:
    """Live R7 backend: CA monitor readback + gateway ``:SP`` blocking set.

    One persistent asyncio event loop in one daemon thread hosts the aioca
    machinery (see the module docstring).  At most one readback monitor is
    active at a time — subscribing closes the previous monitor first, and a
    monotonically increasing generation counter drops any straggler callback
    from a closed monitor.

    Parameters
    ----------
    put_timeout : float, optional
        Per-put budget in seconds (the gateway completes the CA put when
        GEECS accepts the set, so this bounds the whole blocking set).
    """

    def __init__(self, put_timeout: float = _PUT_TIMEOUT_S) -> None:
        self._put_timeout = put_timeout
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_lock = threading.Lock()
        self._generation = 0
        self._pending_subscription: Any = None  # list[Future -> Subscription]

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Return the persistent CA event loop, starting it on first use.

        Returns
        -------
        asyncio.AbstractEventLoop
            A loop running ``run_forever`` in one daemon thread — never
            joined; it dies with the process.
        """
        with self._loop_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(
                    target=loop.run_forever,
                    name="console-device-panel-ca",
                    daemon=True,
                ).start()
                self._loop = loop
        return self._loop

    def subscribe(
        self,
        experiment: str,
        device: str,
        variable: str,
        on_value: Callable[[Any], None],
    ) -> None:
        """Open a CA monitor on the readback PV; returns immediately.

        Parameters
        ----------
        experiment, device, variable : str
            Raw GEECS names; the PV comes from :func:`readback_pv`.
        on_value : callable
            Called with each monitor value **on the CA loop thread** — the
            caller passes a queued Qt signal's ``emit``, so values land on
            the GUI thread.
        """
        self.subscribe_many(
            experiment, [(device, variable)], lambda _index, value: on_value(value)
        )

    def subscribe_many(
        self,
        experiment: str,
        targets: list[tuple[str, str]],
        on_value: Callable[[int, Any], None],
    ) -> None:
        """Open one CA monitor per target; returns immediately.

        Parameters
        ----------
        experiment : str
            Experiment prefix for every target PV.
        targets : list of tuple
            ``(device, variable)`` pairs (a composite's components).
        on_value : callable
            Called with ``(target_index, value)`` **on the CA loop
            thread** — route through a queued Qt signal.
        """
        self.unsubscribe()
        generation = self._generation
        loop = self._ensure_loop()
        pending: list[Any] = []
        for index, (device, variable) in enumerate(targets):
            pv = readback_pv(experiment, device, variable)

            def _dispatch(value: Any, index: int = index) -> None:
                # Drop stragglers from monitors unsubscribe() already retired.
                if generation == self._generation:
                    on_value(index, value)

            async def _start(pv: str = pv, dispatch: Any = _dispatch) -> Any:
                import aioca  # deferred: needs the `ca` extra

                return aioca.camonitor(pv, dispatch)

            pending.append(asyncio.run_coroutine_threadsafe(_start(), loop))
        self._pending_subscription = pending

    def unsubscribe(self) -> None:
        """Close the active monitor(s); returns immediately, never joins."""
        self._generation += 1
        pending, self._pending_subscription = self._pending_subscription, None
        if not pending or self._loop is None:
            return

        async def _close() -> None:
            for item in pending:
                try:
                    subscription = await asyncio.wrap_future(item)
                    subscription.close()
                except Exception as exc:  # noqa: BLE001 — teardown must not raise
                    logger.debug("closing readback monitor failed: %s", exc)

        asyncio.run_coroutine_threadsafe(_close(), self._loop)

    def set(self, experiment: str, device: str, variable: str, value: Any) -> None:
        """Blocking setpoint put through :class:`GatewaySetpointPut`.

        Called from a short-lived daemon thread (never the GUI thread — the
        window dispatches).  Completion means GEECS accepted the set (the
        gateway forwards ``:SP`` puts to GEECS's native blocking set).

        Parameters
        ----------
        experiment, device, variable : str
            Raw GEECS names; the ``:SP`` PV comes from the naming contract.
        value : Any
            The scalar to write; wire-coerced by
            :func:`~geecs_bluesky.devices.ca.gateway_put.wire_value`
            (native numerics, wire string otherwise).

        Raises
        ------
        Exception
            Whatever the put raises (timeout, GEECS rejection, no CA) — the
            window renders it in the status bar.
        """
        from geecs_bluesky.devices.ca._pv import ca_pv
        from geecs_bluesky.devices.ca._pv import setpoint_pv as sp
        from geecs_bluesky.devices.ca.gateway_put import GatewaySetpointPut, wire_value

        put = GatewaySetpointPut(
            setpoint_pv=sp(ca_pv(experiment or None, device, variable)),
            coerce=wire_value,
            timeout=self._put_timeout,
            name=f"{device}:{variable}",
        )
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(put.put(value), loop)
        future.result(timeout=self._put_timeout + _DISPATCH_SLACK_S)
