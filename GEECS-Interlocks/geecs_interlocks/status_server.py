"""The TCP stream Master Control reads: one :func:`~geecs_interlocks.wire.render_status` frame per client every interval.

One accept thread, one sender thread per client. Every sender reads the
monitor's current snapshot and sends the message rendered for it, rendered
once per snapshot however many clients are connected. A client that stops
reading is dropped after :data:`SEND_TIMEOUT_S`.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from geecs_interlocks.monitor import Snapshot
from geecs_interlocks.present import stall_age
from geecs_interlocks.wire import (
    DEFAULT_WIRE_FORMAT,
    SEND_INTERVAL_S,
    WireFormat,
    frame,
    render_stalled,
    render_status,
)

logger = logging.getLogger(__name__)

#: A client that stops reading for this long is dropped (``sendall`` would
#: otherwise block its thread for good once the socket buffer fills).
SEND_TIMEOUT_S = 5.0
#: How often the accept loop wakes to notice :meth:`StatusServer.stop`.
ACCEPT_POLL_S = 0.5


@dataclass(frozen=True)
class StreamClient:
    """One client connected to the stream."""

    address: str
    #: Wall-clock seconds (``time.time()``) when it connected.
    since: float


class StatusServer:
    """Serve the status stream of *snapshot* on ``(host, port)``.

    Parameters
    ----------
    snapshot :
        Returns the current snapshot (``Monitor.snapshot``).
    host, port :
        Where to listen; port ``0`` picks a free one (tests).
    interval :
        Seconds between frames to one client.
    wire_format :
        The message format (``summary``: one keyword per frame, as Master Control needs).
    """

    def __init__(
        self,
        snapshot: Callable[[], Snapshot],
        host: str,
        port: int,
        *,
        interval: float = SEND_INTERVAL_S,
        wire_format: WireFormat = DEFAULT_WIRE_FORMAT,
    ) -> None:
        self.wire_format = wire_format
        self._snapshot = snapshot
        self._host, self._port = host, port
        self._interval = interval
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._clients: dict[int, StreamClient] = {}
        self._frames_sent = 0
        self._rendered: tuple[Snapshot | None, bytes] = (None, b"")

    @property
    def address(self) -> tuple[str, int]:
        """The bound ``(host, port)``; the requested one before :meth:`start`."""
        if self._socket is not None:
            host, port = self._socket.getsockname()[:2]
            return host, port
        return self._host, self._port

    @property
    def frames_sent(self) -> int:
        """Frames sent so far, to every client together."""
        with self._lock:
            return self._frames_sent

    def clients(self) -> list[StreamClient]:
        """The clients connected now, oldest first."""
        with self._lock:
            return sorted(self._clients.values(), key=lambda c: c.since)

    def current_message(self) -> str:
        """The message a frame sent now would carry (the server window shows it)."""
        return self._frame()[4:].decode("utf-8")

    def start(self) -> None:
        """Bind and start accepting (``OSError`` if the port is taken)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self._host, self._port))
            sock.listen()
        except OSError:
            sock.close()
            raise
        sock.settimeout(ACCEPT_POLL_S)
        self._socket = sock
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._accept_loop, name="interlock-status", daemon=True
        )
        self._thread.start()
        logger.info("Master Control stream on %s:%d", *self.address)

    def stop(self) -> None:
        """Stop accepting and close every client (each within one interval)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(ACCEPT_POLL_S * 4)
        if self._socket is not None:
            self._socket.close()
            self._socket = None  # address falls back to the requested one

    def _frame(self) -> bytes:
        snapshot = self._snapshot()
        stalled = stall_age(snapshot)
        if stalled is not None:
            # The loop has stopped replacing the snapshot: never re-send its verdict.
            return frame(render_stalled(stalled, self.wire_format))
        with self._lock:
            rendered_for, data = self._rendered
            if rendered_for is not snapshot:
                data = frame(render_status(snapshot, self.wire_format))
                self._rendered = (snapshot, data)
            return data

    def _accept_loop(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                if not self._stop.is_set():
                    logger.exception("status stream: accept failed")
                return
            threading.Thread(
                target=self._serve,
                args=(conn, addr),
                name=f"interlock-status-{addr[0]}",
                daemon=True,
            ).start()

    def _serve(self, conn: socket.socket, addr: tuple) -> None:
        logger.info("status stream: client %s connected", addr[0])
        key = id(conn)
        with self._lock:
            self._clients[key] = StreamClient(f"{addr[0]}:{addr[1]}", time.time())
        conn.settimeout(SEND_TIMEOUT_S)
        try:
            with conn:
                while not self._stop.is_set():
                    conn.sendall(self._frame())
                    with self._lock:
                        self._frames_sent += 1
                    self._stop.wait(self._interval)
        except OSError:
            pass  # the client went away (or stopped reading): drop it
        finally:
            with self._lock:
                self._clients.pop(key, None)
        logger.info("status stream: client %s disconnected", addr[0])


__all__ = ["ACCEPT_POLL_S", "SEND_TIMEOUT_S", "StatusServer", "StreamClient"]
