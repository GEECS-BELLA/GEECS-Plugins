"""GeecsDevice — the entry-level synchronous client for one GEECS device.

The successor to the legacy GEECS-PythonAPI ``GeecsDevice``: the same familiar
shape (``get`` / ``set`` / ``subscribe`` / ``state`` / ``close``) with the
transport, correlation, and parsing delegated to :mod:`geecs_core.transport`
and device lookup to :class:`geecs_core.db.geecs_db.GeecsDb`. Deliberately
absent from the legacy surface: variable aliases, the ``exp_info`` global,
composite devices (GeecsBluesky's pseudo variables are the successor), scan
helpers, and ``None``-returns on failure — errors raise
:mod:`geecs_core.exceptions` types.

Usage::

    from geecs_core import GeecsDevice

    with GeecsDevice("U_Hexapod") as dev:
        x = dev.get("xpos")
        dev.set("xpos", x + 0.05)
        dev.subscribe(["xpos"], on_update=print)
        ...
        print(dev.state)   # {"xpos": 4.2, "shot number": 117, "connected": True}

Threading model: all I/O runs on the shared background loop
(:mod:`geecs_core.client._loop`); ``get``/``set`` block the calling thread
until the device's exe response arrives. Subscription callbacks run **on the
loop thread** — keep them fast and hand heavy work to your own queue/thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable

from geecs_core.db.geecs_db import GeecsDb
from geecs_core.transport.tcp_subscriber import GeecsTcpSubscriber
from geecs_core.transport.udp_client import GeecsUdpClient

from ._loop import run_sync

logger = logging.getLogger(__name__)

OnUpdate = Callable[[dict[str, Any]], None]
"""Subscription callback: called with ``{variable: value}`` per push frame."""

# Reconnect backoff bounds — the same policy the CA and PVA gateway
# supervisors use for their device subscriptions.
_BACKOFF_INITIAL_S = 0.5
_BACKOFF_MAX_S = 30.0

# state keys owned by the client, never by a GEECS variable.
_RESERVED_STATE_KEYS = frozenset({"shot number", "connected"})


class GeecsDevice:
    """A GEECS device over direct UDP/TCP: get, set, subscribe. Nothing else.

    Parameters
    ----------
    name : str
        GEECS device name (e.g. ``"U_Hexapod"``). Looked up in the experiment
        database unless *host*/*port* are given.
    host, port : str, int, optional
        Explicit endpoint, skipping the database entirely (off-network use,
        fake-server tests). Give both or neither.

    Attributes
    ----------
    state : dict
        Last-known values, fed by get/set responses and subscription frames.
        Two reserved keys: ``"connected"`` (subscription stream up) and
        ``"shot number"`` (the device's frame counter, while subscribed).
        Written from the loop thread; reads from any thread see the latest
        completed write (last-write-wins).

    Raises
    ------
    GeecsDeviceNotFoundError
        If *name* is not in the database (DB-lookup construction only).
    """

    GET_TIMEOUT_S: float = 10.0
    SET_TIMEOUT_S: float = 30.0  # GEECS sets block until convergence

    def __init__(
        self,
        name: str,
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        if (host is None) != (port is None):
            raise ValueError("give both host and port, or neither")
        if host is None:
            host, port = GeecsDb.find_device(name)
        self.name = name
        self._host: str = host
        self._port: int = int(port)  # type: ignore[arg-type]

        self.state: dict[str, Any] = {}
        self._udp: GeecsUdpClient | None = None
        # Guards every lifecycle transition (_closed, _udp, _sub, _sub_task)
        # so close() cannot race get()/subscribe() into leaked sockets or an
        # orphaned supervisor. Never taken on the loop thread.
        self._lock = threading.Lock()
        self._sub: GeecsTcpSubscriber | None = None
        self._sub_task: asyncio.Task | None = None
        self._closed = False

    # -- get / set -----------------------------------------------------------

    def get(self, variable: str, timeout: float | None = None) -> Any:
        """Read *variable*, blocking until the device answers.

        Parameters
        ----------
        variable : str
            Raw GEECS variable name (no aliases).
        timeout : float, optional
            Exe-response budget in seconds (default ``GET_TIMEOUT_S``).

        Returns
        -------
        Any
            The value the device reported, numerically coerced
            (int/float/str). Never ``None``-on-failure.

        Raises
        ------
        GeecsCommandRejectedError, GeecsCommandFailedError, GeecsConnectionError
        """
        value = run_sync(
            self._ensure_udp().get(
                variable,
                timeout=timeout if timeout is not None else self.GET_TIMEOUT_S,
            )
        )
        if variable not in _RESERVED_STATE_KEYS:
            self.state[variable] = value
        return value

    def set(self, variable: str, value: Any, timeout: float | None = None) -> Any:
        """Write *variable*, blocking until the device reports convergence.

        Returns
        -------
        Any
            The device's reported readback from the exe response — not an
            echo of *value*, not a cache read.

        Raises
        ------
        GeecsCommandRejectedError, GeecsCommandFailedError, GeecsConnectionError
        """
        result = run_sync(
            self._ensure_udp().set(
                variable,
                value,
                timeout=timeout if timeout is not None else self.SET_TIMEOUT_S,
            )
        )
        if variable not in _RESERVED_STATE_KEYS:
            self.state[variable] = result
        return result

    def _ensure_udp(self) -> GeecsUdpClient:
        """Create and connect the UDP client on first use (thread-safe).

        The closed check lives *inside* the lock: checking it outside would
        let a racing ``close()`` complete between check and creation, leaving
        live sockets on a closed device.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError(f"GeecsDevice({self.name!r}) is closed")
            if self._udp is None:
                client = GeecsUdpClient(self._host, self._port, device_name=self.name)
                run_sync(client.connect())
                self._udp = client
            return self._udp

    # -- subscribe -----------------------------------------------------------

    def subscribe(
        self,
        variables: list[str] | None = None,
        on_update: OnUpdate | None = None,
        *,
        reconnect: bool = True,
        text_variables: set[str] | None = None,
    ) -> None:
        """Start the device's TCP push stream into ``state``.

        Parameters
        ----------
        variables : list of str, optional
            Variables to subscribe. ``None`` subscribes the device's full
            database-declared variable set (one ``GeecsDb`` lookup).
        on_update : callable, optional
            Called with each parsed frame ``{variable: value, "shot number":
            int}`` after ``state`` is updated. Runs on the loop thread;
            exceptions are logged and the stream continues.
        reconnect : bool
            When True (default) a supervisor keeps the stream alive through
            device restarts (0.5 s → ×2 → 30 s backoff, the gateways'
            policy). When False the stream simply ends on the first drop.
        text_variables : set of str, optional
            Variables whose values must stay exact raw wire text (paths,
            labels) — everything else gets numeric coercion, which is lossy
            for text that *looks* numeric (``'007'`` → ``7``).

        Raises
        ------
        RuntimeError
            If already subscribed (call :meth:`unsubscribe` first).
        OSError, TimeoutError
            If the initial connection fails — surfaced synchronously in both
            reconnect modes so a bad endpoint is loud at the call site.
        """
        if variables is None:
            variables = [m["name"] for m in GeecsDb.get_device_variables(self.name)]
        reserved = _RESERVED_STATE_KEYS.intersection(variables)
        if reserved:
            raise ValueError(
                f"variable name(s) {sorted(reserved)} collide with reserved state keys"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError(f"GeecsDevice({self.name!r}) is closed")
            if self._sub_task is not None and not self._sub_task.done():
                raise RuntimeError(
                    f"GeecsDevice({self.name!r}) is already subscribed — "
                    "unsubscribe() first"
                )
            run_sync(
                self._start_subscription(
                    list(variables),
                    frozenset(text_variables or ()),
                    on_update,
                    reconnect,
                )
            )

    async def _start_subscription(
        self,
        variables: list[str],
        text_variables: frozenset[str],
        on_update: OnUpdate | None,
        reconnect: bool,
    ) -> None:
        """Connect + subscribe (raising on failure), then start the supervisor."""
        sub = await self._connect_subscriber(variables, text_variables, on_update)
        self._sub = sub
        self._sub_task = asyncio.create_task(
            self._supervise(sub, variables, text_variables, on_update, reconnect),
            name=f"geecs-device-sub[{self.name}]",
        )

    async def _connect_subscriber(
        self,
        variables: list[str],
        text_variables: frozenset[str],
        on_update: OnUpdate | None,
    ) -> GeecsTcpSubscriber:
        """Open one subscriber and send the Wait command.

        Owns the subscriber until it is handed back: any failure (or a
        cancellation landing mid-connect) closes the half-established socket
        before propagating — the discipline both gateway supervisors follow.
        """
        sub = GeecsTcpSubscriber(self._host, self._port)
        try:
            await sub.connect()
            await sub.subscribe(
                variables,
                self._make_dispatch(on_update),
                text_variables=text_variables,
                include_shot=True,
            )
        except BaseException:
            await sub.close()
            raise
        self.state["connected"] = True
        return sub

    def _make_dispatch(self, on_update: OnUpdate | None) -> Callable[[dict], None]:
        """Build the per-frame callback: update state, then the user's hook."""

        def dispatch(frame: dict[str, Any]) -> None:
            self.state.update(frame)
            self.state["connected"] = True
            if on_update is not None:
                on_update(frame)  # exceptions logged by the transport listener

        return dispatch

    async def _supervise(
        self,
        sub: GeecsTcpSubscriber,
        variables: list[str],
        text_variables: frozenset[str],
        on_update: OnUpdate | None,
        reconnect: bool,
    ) -> None:
        """Keep the stream alive: await drops, back off, reconnect."""
        backoff = _BACKOFF_INITIAL_S
        try:
            while True:
                await sub.wait_disconnected()
                await sub.close()
                self.state["connected"] = False
                if not reconnect:
                    logger.info(
                        "subscription to %s dropped (reconnect=False)", self.name
                    )
                    return
                logger.warning("subscription to %s dropped — reconnecting", self.name)
                while True:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_MAX_S)
                    try:
                        sub = await self._connect_subscriber(
                            variables, text_variables, on_update
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # Broad on purpose (gateway-supervisor parity): any
                        # narrower filter lets an unexpected error kill the
                        # supervisor silently, freezing state["connected"].
                        logger.debug(
                            "reconnect attempt to %s failed",
                            self.name,
                            exc_info=True,
                        )
                        continue
                    self._sub = sub
                    backoff = _BACKOFF_INITIAL_S
                    logger.info("subscription to %s restored", self.name)
                    break
        except asyncio.CancelledError:
            await sub.close()
            raise

    def unsubscribe(self) -> None:
        """Stop the push stream (idempotent); get/set stay usable."""
        with self._lock:
            if self._sub_task is None and self._sub is None:
                return
            run_sync(self._stop_subscription())
        self.state["connected"] = False

    async def _stop_subscription(self) -> None:
        """Cancel the supervisor and close the live subscriber."""
        task, self._sub_task = self._sub_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        sub, self._sub = self._sub, None
        if sub is not None:
            await sub.close()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Release the subscription and both UDP sockets. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._sub_task is not None or self._sub is not None:
                run_sync(self._stop_subscription())
                self.state["connected"] = False
            udp, self._udp = self._udp, None
            if udp is not None:
                run_sync(udp.close())

    def __enter__(self) -> "GeecsDevice":
        """Return self."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the device."""
        self.close()

    def __repr__(self) -> str:
        """Endpoint-bearing repr for logs."""
        return (
            f"GeecsDevice({self.name!r}, host={self._host!r}, port={self._port}"
            f"{', closed' if self._closed else ''})"
        )
