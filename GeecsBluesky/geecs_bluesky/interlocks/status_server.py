"""The interlock status stream: the TCP feed the LabVIEW side reads.

The retired ``geecs_python_api.controls.interlocks`` server's framing (a
4-byte big-endian length, then UTF-8; every :data:`SEND_INTERVAL_S` to
each client; parts joined with ``" | "``), with the message led by the
overall verdict, because the master-control VI decides by looking for the
keyword ``SAFE``::

    SAFE | OK: <name> | OK: <name>                          every interlock safe
    WARNING! TRIPPED | TRIPPED: <name> - <reason> | OK: <name>
    WARNING! TRIPPED | NOT ARMED: <name> - <why>            could not be armed
    WARNING! TRIPPED | NO INTERLOCKS                        none registered

**The one rule** (pinned by the tests over every mix of states): the word
"safe", in any case, appears only as the leading ``SAFE`` of an all-clear
frame — so any keyword search, however the VI does it, is right.  The
retired server's ``SAFE: <name>`` / ``<name> UNSAFE`` parts broke that
(``UNSAFE`` contains ``SAFE``; a mixed frame still carried ``SAFE:``),
which is why the parts changed.  Inside a part a "safe" in a name or
reason is sent as ``sa-fe``, and a ``|`` as ``/``.

It reads the registry's cached verdicts, never a PV, for the worker
environment's lifetime; ``config.ini [interlocks] status_host /
status_port`` say where (no port, no server).
"""

from __future__ import annotations

import logging
import re
import socket
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from geecs_bluesky.interlocks.install import InterlockRegistry, InterlockStatus
from geecs_bluesky.interlocks.wire import (
    ALL_SAFE,
    NO_INTERLOCKS,
    SEND_INTERVAL_S,
    TRIPPED,
    frame,
)

logger = logging.getLogger(__name__)

#: A client that stops reading for this long is dropped (``sendall`` would
#: otherwise block its thread for good once the socket buffer fills).
SEND_TIMEOUT_S = 5.0
#: How often the accept loop wakes to notice :meth:`InterlockStatusServer.stop`.
ACCEPT_POLL_S = 0.5
#: The bind address when ``status_host`` is not set: every interface.
DEFAULT_HOST = "0.0.0.0"

_SAFE_WORD = re.compile("safe", re.IGNORECASE)
_NOT_ARMED_PREFIX = "not armed: "


def _clean(text: str) -> str:
    """*text* safe to put in a part: no separator, no keyword ("Safety" → "Sa-fety")."""
    text = text.replace("|", "/")
    return _SAFE_WORD.sub(lambda m: f"{m.group(0)[:2]}-{m.group(0)[2:]}", text)


def _part(status: InterlockStatus) -> str:
    name = _clean(status.name)
    if not status.unsafe:
        return f"OK: {name}"
    if not status.armed:
        why = status.diagnostic.removeprefix(_NOT_ARMED_PREFIX)
        return f"NOT ARMED: {name} - {_clean(why)}"
    if status.diagnostic:
        return f"TRIPPED: {name} - {_clean(status.diagnostic)}"
    return f"TRIPPED: {name}"


def render_status(statuses: Sequence[InterlockStatus]) -> str:
    """The message one frame carries: the overall verdict, then one part per interlock."""
    if not statuses:
        return f"{TRIPPED} | {NO_INTERLOCKS}"
    verdict = TRIPPED if any(s.unsafe for s in statuses) else ALL_SAFE
    return " | ".join([verdict, *(_part(s) for s in statuses)])


@dataclass(frozen=True)
class StreamClient:
    """One client connected to a stream."""

    address: str
    #: Wall-clock seconds (``time.time()``) when it connected.
    since: float


class FrameServer:
    """Send ``frame(message())`` to every connected client, every *interval*.

    Parameters
    ----------
    message :
        Returns the text one frame carries (called once per frame, per client).
    host, port :
        Where to listen; port ``0`` picks a free one (tests).
    interval :
        Seconds between frames to one client.
    label :
        What the log lines and thread names call this stream.
    """

    def __init__(
        self,
        message: Callable[[], str],
        host: str = DEFAULT_HOST,
        port: int = 0,
        *,
        interval: float = SEND_INTERVAL_S,
        label: str = "interlock status",
    ) -> None:
        self._message = message
        self._host = host
        self._port = port
        self._interval = interval
        self._label = label
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._clients: dict[int, StreamClient] = {}
        self._frames_sent = 0

    @property
    def address(self) -> tuple[str, int]:
        """The bound ``(host, port)``; the requested one before :meth:`start`."""
        if self._socket is not None:
            host, port = self._socket.getsockname()[:2]
            return host, port
        return self._host, self._port

    @property
    def interval(self) -> float:
        """Seconds between two frames to one client."""
        return self._interval

    @property
    def frames_sent(self) -> int:
        """Frames sent so far, to every client together."""
        with self._lock:
            return self._frames_sent

    def clients(self) -> list[StreamClient]:
        """The clients connected now, oldest first."""
        with self._lock:
            return sorted(self._clients.values(), key=lambda c: c.since)

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
        self._thread = threading.Thread(
            target=self._accept_loop, name=self._thread_name(), daemon=True
        )
        self._thread.start()
        logger.info("%s stream on %s:%d", self._label, *self.address)

    def stop(self) -> None:
        """Stop accepting and close every client (each within one interval)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(ACCEPT_POLL_S * 4)
        if self._socket is not None:
            self._socket.close()

    def _thread_name(self, suffix: str = "") -> str:
        name = self._label.replace(" ", "-")
        return f"{name}-{suffix}" if suffix else name

    def _accept_loop(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                if not self._stop.is_set():
                    logger.exception("%s stream: accept failed", self._label)
                return
            threading.Thread(
                target=self._serve,
                args=(conn, addr),
                name=self._thread_name(addr[0]),
                daemon=True,
            ).start()

    def _serve(self, conn: socket.socket, addr: tuple) -> None:
        logger.info("%s stream: client %s connected", self._label, addr[0])
        key = id(conn)
        with self._lock:
            self._clients[key] = StreamClient(f"{addr[0]}:{addr[1]}", time.time())
        conn.settimeout(SEND_TIMEOUT_S)
        try:
            with conn:
                while not self._stop.is_set():
                    conn.sendall(frame(self._message()))
                    with self._lock:
                        self._frames_sent += 1
                    self._stop.wait(self._interval)
        except OSError:
            pass  # the client went away (or stopped reading): drop it
        finally:
            with self._lock:
                self._clients.pop(key, None)
        logger.info("%s stream: client %s disconnected", self._label, addr[0])


class InterlockStatusServer(FrameServer):
    """Serve :func:`render_status` of *snapshot* to every connected client.

    Parameters
    ----------
    snapshot :
        Returns the current statuses (the registry's ``snapshot``).
    host, port :
        Where to listen; port ``0`` picks a free one (tests).
    interval :
        Seconds between frames to one client.
    """

    def __init__(
        self,
        snapshot: Callable[[], Sequence[InterlockStatus]],
        host: str = DEFAULT_HOST,
        port: int = 0,
        *,
        interval: float = SEND_INTERVAL_S,
    ) -> None:
        super().__init__(
            lambda: render_status(snapshot()),
            host,
            port,
            interval=interval,
            label="interlock status",
        )


def listen_address(
    port_key: str, *, config_path: Path | None = None
) -> tuple[str, int] | None:
    """``(host, port)`` from ``config.ini [interlocks]``; ``None`` when off.

    *port_key* names the port entry (``status_port``, ``detail_port``);
    the host is ``status_host`` for both, every interface by default.
    Never raises: a malformed port is logged at ERROR and reads as off.
    """
    from geecs_core.configs_repo import read_config_entry

    raw_port = read_config_entry("interlocks", port_key, config_path)
    if raw_port is None:
        logger.info("[interlocks] %s not set in config.ini: stream off", port_key)
        return None
    host = read_config_entry("interlocks", "status_host", config_path) or DEFAULT_HOST
    if not raw_port.isdigit():
        logger.error("stream off: [interlocks] %s=%r is not a port", port_key, raw_port)
        return None
    return host, int(raw_port)


def start_listening(server: FrameServer) -> FrameServer | None:
    """``server.start()``; ``None`` (logged at ERROR) when the port cannot be bound."""
    failure: OSError | None = None
    try:
        server.start()
    except OSError as exc:
        failure = exc
    if failure is not None:
        host, port = server.address
        logger.error("stream off: cannot listen on %s:%d (%s)", host, port, failure)
        return None
    return server


def start_status_server(
    registry: InterlockRegistry | None, *, config_path: Path | None = None
) -> InterlockStatusServer | None:
    """Start the stream from ``config.ini [interlocks]``; ``None`` when not configured.

    Never raises: a malformed port or a taken one is logged at ERROR and
    the worker runs without the stream.  With no registry (no
    ``interlocks.yaml``) the stream still runs and says ``NO INTERLOCKS``
    under a ``WARNING! TRIPPED`` verdict — never ``SAFE``.
    """
    address = listen_address("status_port", config_path=config_path)
    if address is None:
        return None
    server = InterlockStatusServer(
        registry.snapshot if registry is not None else list, *address
    )
    started = start_listening(server)
    return server if started is not None else None


__all__ = [
    "ALL_SAFE",
    "NO_INTERLOCKS",
    "SEND_INTERVAL_S",
    "TRIPPED",
    "FrameServer",
    "InterlockStatusServer",
    "StreamClient",
    "frame",
    "listen_address",
    "render_status",
    "start_listening",
    "start_status_server",
]
