"""systemd's notify protocol: ``READY=1`` once serving, ``WATCHDOG=1`` from the evaluation loop.

Under a unit with ``Type=notify`` and ``WatchdogSec=``, systemd restarts the
server when the watchdog goes unfed for that long. The monitor feeds it after
every evaluation (``Monitor(on_tick=)``), so a stalled loop, not only a dead
process, is restarted. Outside systemd (no ``NOTIFY_SOCKET``) every call does
nothing. The protocol is one datagram per message on a Unix socket; no
dependency is needed for it.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time

logger = logging.getLogger(__name__)

#: Watchdog pings sent per second at most (the loop ticks at 10 Hz).
WATCHDOG_HZ = 2.0


class Notifier:
    """Sends notify messages to ``NOTIFY_SOCKET``; inert when it is unset."""

    def __init__(self, address: str | None = None) -> None:
        address = os.environ.get("NOTIFY_SOCKET") if address is None else address
        if address and address.startswith("@"):
            address = "\0" + address[1:]  # an abstract socket
        self._address = address or None
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._last_ping = 0.0

    @property
    def active(self) -> bool:
        """Whether a notify socket was given (the process runs under systemd)."""
        return self._address is not None

    def send(self, message: str) -> None:
        """Send *message*; a failure is logged, never raised."""
        if self._address is None:
            return
        with self._lock:
            try:
                if self._sock is None:
                    self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                self._sock.sendto(message.encode(), self._address)
            except OSError as exc:
                logger.warning("systemd notify failed: %s", exc)

    def ready(self) -> None:
        """Tell systemd the server is up."""
        self.send("READY=1")

    def stopping(self) -> None:
        """Tell systemd the server is stopping."""
        self.send("STOPPING=1")

    def watchdog(self) -> None:
        """Feed the watchdog (at most :data:`WATCHDOG_HZ` a second)."""
        if self._address is None:
            return
        now = time.monotonic()
        if now - self._last_ping < 1.0 / WATCHDOG_HZ:
            return
        self._last_ping = now
        self.send("WATCHDOG=1")

    def close(self) -> None:
        """Close the socket."""
        with self._lock:
            if self._sock is not None:
                self._sock.close()
                self._sock = None


__all__ = ["WATCHDOG_HZ", "Notifier"]
