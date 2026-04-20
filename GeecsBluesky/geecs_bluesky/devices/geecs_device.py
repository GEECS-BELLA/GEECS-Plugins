"""ophyd-async Device base class for GEECS hardware.

Usage pattern::

    from geecs_bluesky.devices.geecs_device import GeecsDevice
    from geecs_bluesky.signals import geecs_signal_rw, geecs_signal_r
    from geecs_bluesky.transport.udp_client import GeecsUdpClient

    class JetStage(GeecsDevice):
        def __init__(self, host: str, port: int, name: str = ""):
            dev = "U_ESP_JetXYZ"
            udp = GeecsUdpClient(host, port)
            with self.add_children_as_readables():
                self.x = geecs_signal_rw(float, dev, "Position.Axis 1", host, port,
                                         units="mm", shared_udp=udp)
            super().__init__(name=name, shared_udp=udp)

    stage = JetStage("192.168.8.198", 65158, name="jet")
    await stage.connect()   # connects UDP, pre-populates cache, starts TCP subscriber
    reading = await stage.read()
    await stage.disconnect()

DB-resolved construction (requires ``mysql-connector-python``)::

    stage = JetStage.from_db("U_ESP_JetXYZ", name="jet")
    await stage.connect()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from bluesky.protocols import Reading
from ophyd_async.core import Signal, StandardReadable

from geecs_bluesky.backends.geecs_signal_backend import GeecsSignalBackend
from geecs_bluesky.transport.tcp_subscriber import GeecsTcpSubscriber
from geecs_bluesky.transport.udp_client import GeecsUdpClient

logger = logging.getLogger(__name__)


class GeecsDevice(StandardReadable):
    """Thin ``StandardReadable`` subclass for GEECS devices.

    Provides:

    * Shared-UDP-client lifecycle — pass ``shared_udp`` to ``super().__init__``.
    * **Shared TCP subscriber** — one TCP connection per device that populates
      ``_shot_cache`` and ``_shot_queue`` atomically on every 5-Hz push frame.
      All child signal backends read from this cache via ``get_value()``, and
      registered monitoring callbacks are forwarded on each push.
    * :meth:`from_db` — construct from GEECS MySQL database lookup.
    * :meth:`disconnect` — cancels TCP task, closes signal backends, closes UDP.
    """

    def __init__(
        self,
        *args: Any,
        shared_udp: GeecsUdpClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._shared_udp = shared_udp
        self._shot_cache: dict[str, Any] = {}
        self._shot_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=32)
        self._tcp_task: asyncio.Task | None = None
        self._signal_backends: list[GeecsSignalBackend] = []

    async def connect(
        self,
        mock: bool = False,
        timeout: float = 10.0,
        force_reconnect: bool = False,
    ) -> None:
        """Connect UDP, pre-populate cache, then connect signals, then start TCP."""
        if self._shared_udp is not None:
            if force_reconnect and self._shared_udp._cmd_transport is not None:
                await self._shared_udp.close()
            if self._shared_udp._cmd_transport is None:
                await self._shared_udp.connect()
                logger.debug("GeecsDevice shared UDP connected")

        # Reset state for clean (re)connect
        self._shot_cache.clear()
        self._signal_backends.clear()
        while not self._shot_queue.empty():
            self._shot_queue.get_nowait()

        # Pre-populate cache via UDP GET and inject cache into backends BEFORE
        # super().connect(), because super() may call backend.set_callback() which
        # needs the cache populated to deliver an immediate reading to ophyd-async.
        if self._shared_udp is not None:
            for _name, child in self.children():
                if isinstance(child, Signal):
                    backend = child._connector.backend
                    if isinstance(backend, GeecsSignalBackend):
                        self._signal_backends.append(backend)
                        backend._shot_cache = self._shot_cache
                        try:
                            raw = await self._shared_udp.get(backend._variable)
                            self._shot_cache[backend._variable] = raw
                        except Exception:
                            logger.debug(
                                "initial UDP GET failed for %s/%s",
                                backend._device_name,
                                backend._variable,
                            )

            # GeecsTriggerable devices also need acq_timestamp in the subscription
            from geecs_bluesky.devices.triggerable import GeecsTriggerable

            if isinstance(self, GeecsTriggerable):
                ts_var = self._acq_timestamp_variable  # type: ignore[attr-defined]
                if ts_var not in self._shot_cache:
                    try:
                        raw = await self._shared_udp.get(ts_var)
                        self._shot_cache[ts_var] = raw
                    except Exception:
                        logger.debug("initial UDP GET failed for acq_timestamp")

        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )

        # Start the shared TCP subscriber
        variables = [b._variable for b in self._signal_backends]
        from geecs_bluesky.devices.triggerable import GeecsTriggerable

        if isinstance(self, GeecsTriggerable):
            ts_var = self._acq_timestamp_variable  # type: ignore[attr-defined]
            if ts_var not in variables:
                variables.append(ts_var)

        if variables and self._shared_udp is not None:
            host = self._shared_udp._host
            port = self._shared_udp._port
            # Cancel any previous TCP task before starting a new one
            if self._tcp_task is not None and not self._tcp_task.done():
                self._tcp_task.cancel()
                try:
                    await self._tcp_task
                except asyncio.CancelledError:
                    pass
            self._tcp_task = asyncio.create_task(
                self._run_tcp_subscription(host, port, variables),
                name=f"tcp-sub[{self.name}]",
            )
            logger.debug(
                "GeecsDevice TCP subscriber started for %s: %s", self.name, variables
            )

    async def _run_tcp_subscription(
        self, host: str, port: int, variables: list[str]
    ) -> None:
        """Background task: maintain TCP subscription and update shot cache."""
        try:
            async with GeecsTcpSubscriber(host, port) as sub:
                await sub.subscribe(variables, self._on_tcp_push)
                await asyncio.sleep(float("inf"))
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("TCP subscription failed for %s", self.name)

    def _on_tcp_push(self, frame: dict[str, Any]) -> None:
        """Callback invoked on each 5-Hz TCP push frame."""
        self._shot_cache.update(frame)
        try:
            self._shot_queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # trigger() drains stale frames; a full queue means nobody is consuming

        # Forward new values to any registered monitoring callbacks on signal backends
        for backend in self._signal_backends:
            if backend._callback is not None and backend._variable in frame:
                raw = frame[backend._variable]
                value = backend._cast(raw)
                reading = Reading(
                    value=value, timestamp=time.monotonic(), alarm_severity=0
                )
                try:
                    backend._callback(reading)
                except Exception:
                    logger.exception(
                        "signal callback raised in %s/%s",
                        backend._device_name,
                        backend._variable,
                    )

    async def disconnect(self) -> None:
        """Stop TCP subscriber, disconnect signal backends, close shared UDP."""
        if self._tcp_task is not None and not self._tcp_task.done():
            self._tcp_task.cancel()
            try:
                await self._tcp_task
            except asyncio.CancelledError:
                pass
            self._tcp_task = None

        for _name, child in self.children():
            if isinstance(child, Signal):
                backend = child._connector.backend
                if isinstance(backend, GeecsSignalBackend):
                    await backend.disconnect()

        if self._shared_udp is not None:
            await self._shared_udp.close()
            logger.debug("GeecsDevice shared UDP closed")

    @classmethod
    def from_db(
        cls,
        device_name: str,
        name: str = "",
        **kwargs: Any,
    ) -> "GeecsDevice":
        """Construct a device by looking up ``(host, port)`` from the GEECS database.

        Reads credentials from the standard GEECS configuration files
        (``~/.config/geecs_python_api/config.ini`` → ``Configurations.INI``).
        Requires ``mysql-connector-python``::

            pip install mysql-connector-python

        Parameters
        ----------
        device_name:
            Device name to resolve (e.g. ``"U_ESP_JetXYZ"``).
        name:
            ophyd-async device name.
        **kwargs:
            Additional keyword arguments forwarded to ``cls.__init__``.
        """
        from geecs_bluesky.db.geecs_db import GeecsDb

        host, port = GeecsDb.find_device(device_name)
        logger.info("DB resolved %s → %s:%s", device_name, host, port)
        return cls(host=host, port=port, name=name, **kwargs)
