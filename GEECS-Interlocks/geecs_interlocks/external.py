"""External signals: UDP status packets from programs outside GEECS.

A program that is not a GEECS device — a motion controller on another
computer, say — sends one short text packet to the interlock server whenever
its state changes, and nothing in between. A packet whose **first word** is
``safe`` (any case) means safe: it is OK to fire a shot. Whatever follows
the first word is information for operators only (``safe stage1 parked``).
Every other packet is **unsafe** (``stage1 safe``, ``safety``, ``moving``,
an empty one), whatever it says: no keyword is needed to report a hazard,
only the absence of a leading ``safe``. Before the first packet the signal
is unknown, which every reader treats as unsafe too.

The listeners belong to the engine, not to a monitor, and outlive a reload:
a sender that only transmits on a change would otherwise leave a reloaded
server unsafe until its next change. :class:`ExternalSignals` opens a
listener per port the interlocks name and closes the ones no longer named.
"""

from __future__ import annotations

import logging
import re
import socket
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

#: How often a listener wakes to notice :meth:`UdpListener.stop`.
RECV_POLL_S = 0.5
#: The longest packet text kept for display.
RAW_KEPT = 200
#: A safe packet: its first word is ``safe`` (a whole word: ``safety`` is not).
_SAFE_FIRST = re.compile(r"^safe\b")


@dataclass(frozen=True)
class SignalReading:
    """One external signal's latest state.

    Attributes
    ----------
    state :
        ``True`` unsafe (any packet not starting with ``safe``), ``False``
        safe, ``None`` before the first packet.
    received :
        ``time.monotonic()`` of the last packet of any kind; ``None`` before
        the first.
    raw :
        The last packet's text (trimmed), for operators.
    sender :
        The address the last packet came from.
    """

    state: bool | None = None
    received: float | None = None
    raw: str = ""
    sender: str = ""


#: The reading of a signal nothing has been heard from.
NO_SIGNAL = SignalReading()


class SignalSource(Protocol):
    """What the monitor reads: the latest reading per UDP port, from any thread."""

    def signal(self, port: int, sender: str = "") -> SignalReading:
        """The latest reading on *port* (from *sender* only, when given; :data:`NO_SIGNAL` if none yet)."""


def decode(payload: bytes) -> bool:
    """A packet's state: ``False`` safe (its first word is ``safe``), ``True`` unsafe (anything else).

    Only the first word decides; the rest is information for operators.
    """
    text = payload.decode("utf-8", errors="replace").strip().lower()
    return _SAFE_FIRST.match(text) is None


class UdpListener:
    """Listen on one UDP port and keep the state its last packet said.

    Parameters
    ----------
    port :
        The UDP port (``0`` picks a free one: tests).
    host :
        The local address to bind: one of THIS machine's interfaces, never
        the sender's address (``0.0.0.0``: every interface).
    """

    def __init__(self, port: int, host: str) -> None:
        self._requested = (host, port)
        self._lock = threading.Lock()
        self._reading = NO_SIGNAL
        self._by_sender: dict[str, SignalReading] = {}
        self._socket: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        """The bound ``(host, port)``; the requested one before :meth:`start`."""
        if self._socket is not None:
            host, port = self._socket.getsockname()[:2]
            return host, port
        return self._requested

    def reading(self, sender: str = "") -> SignalReading:
        """The latest state (any thread): from any address, or from *sender* only.

        Each sender keeps its own reading, so a packet from another host never
        overwrites the named sender's last state.
        """
        with self._lock:
            if sender:
                return self._by_sender.get(sender, NO_SIGNAL)
            return self._reading

    def start(self) -> None:
        """Bind and start listening.

        Raises
        ------
        OSError
            The port is taken, or *host* is not an address of this machine.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(self._requested)
        except OSError as exc:
            sock.close()
            host, port = self._requested
            raise OSError(
                f"cannot listen for external signals on {host}:{port}: {exc}. "
                "The host must be an address of this machine (0.0.0.0 for "
                "every interface), not the sender's."
            ) from exc
        sock.settimeout(RECV_POLL_S)
        self._socket = sock
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"interlock-udp-{self.address[1]}", daemon=True
        )
        self._thread.start()
        logger.info("external signals on UDP %s:%d", *self.address)

    def stop(self) -> None:
        """Stop listening and release the port."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(RECV_POLL_S * 4)
            self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def _loop(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                payload, sender = self._socket.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:
                if not self._stop.is_set():
                    logger.exception("UDP %d: receive failed", self.address[1])
                return
            self.receive(payload, sender[0])

    def receive(self, payload: bytes, sender: str = "") -> None:
        """Take one packet: its state, its text and its time."""
        state = decode(payload)
        raw = payload.decode("utf-8", errors="replace").strip()[:RAW_KEPT]
        with self._lock:
            old = self._reading
            self._reading = SignalReading(state, time.monotonic(), raw, sender)
            self._by_sender[sender] = self._reading
        port = self.address[1]
        if state != old.state:
            label = "UNSAFE" if state else "SAFE"
            logger.info("UDP %d: %s from %s (%r)", port, label, sender, raw)


class ExternalSignals:
    """The engine's UDP listeners, one per port, kept across reloads.

    Parameters
    ----------
    host :
        The local address every listener binds.
    """

    def __init__(self, host: str) -> None:
        self.host = host
        self._lock = threading.Lock()
        self._listeners: dict[int, UdpListener] = {}

    def ports(self) -> list[int]:
        """The ports listened on now."""
        with self._lock:
            return sorted(self._listeners)

    def listener(self, port: int) -> UdpListener | None:
        """The listener on *port*, if one is open."""
        with self._lock:
            return self._listeners.get(port)

    def signal(self, port: int, sender: str = "") -> SignalReading:
        """The latest reading on *port*, from *sender* only when given (:data:`NO_SIGNAL` if none)."""
        listener = self.listener(port)
        return NO_SIGNAL if listener is None else listener.reading(sender)

    def open(self, ports: Iterable[int]) -> None:
        """Listen on every port in *ports* not listened on yet.

        All or nothing: if one port cannot be bound, the ones this call
        opened are closed again before the error is raised.

        Raises
        ------
        OSError
            A port could not be bound.
        """
        opened: list[UdpListener] = []
        try:
            for port in sorted(set(ports)):
                if self.listener(port) is not None:
                    continue
                listener = UdpListener(port, self.host)
                listener.start()
                opened.append(listener)
        except OSError:
            for listener in opened:
                listener.stop()
            raise
        with self._lock:
            for listener in opened:
                self._listeners[listener.address[1]] = listener

    def retain(self, ports: Iterable[int]) -> None:
        """Close every listener whose port is not in *ports*."""
        keep = set(ports)
        with self._lock:
            gone = [p for p in self._listeners if p not in keep]
            closing = [self._listeners.pop(p) for p in gone]
        for listener in closing:
            listener.stop()

    def stop(self) -> None:
        """Close every listener."""
        self.retain(())


__all__ = [
    "NO_SIGNAL",
    "ExternalSignals",
    "SignalReading",
    "SignalSource",
    "UdpListener",
    "decode",
]
