"""UDP listener for one-shot, unicast status signals from non-GEECS sources.

Some external processes (e.g. a motion controller on a separate computer) report
their status not by continuously updating a GEECS device variable but by
unicasting a single UDP packet whenever their state changes - one packet when
motion starts, one packet when it's done, and nothing in between. There is no
GeecsDevice or DeviceMonitorGroup involved, so this listener stands in for that
role: it owns the socket, caches the last received state, and lets an
ExternalStatusCheck (see monitor_conditions.py) read it like any other check.

BUILT-IN FAIL-SAFE: before the first packet is ever received, the state is
unknown and treated as unsafe - same "no data yet means unsafe" rule the rest
of this library follows. There is intentionally NO staleness timeout after
that: the sender only transmits on change, so silence is expected and does not
by itself mean the sender died. If the sender is later changed to also
heartbeat periodically, staleness can be layered on top by checking
`time.time() - listener.get_last_received_at()` in a custom predicate.
"""

import logging
import socket
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def default_decode(payload: bytes) -> bool | None:
    """Decode a packet's payload into a state: True=unsafe, False=safe, None=unrecognized.

    This assumes a plain text payload containing "safe" or "warning"
    (case-insensitive). Replace this by passing a custom ``decode`` callable to
    ExternalStatusListener once you know the sender's exact payload format -
    turn on DEBUG logging first to see the raw bytes it actually sends.
    """
    try:
        text = payload.decode("utf-8", errors="replace").strip().lower()
    except Exception:  # noqa: BLE001
        return None
    if "warning" in text:
        return True
    if "safe" in text:
        return False
    return None


class ExternalStatusListener:
    """Background UDP listener that tracks the last-known state of an external signal.

    Usage:
        listener = ExternalStatusListener(port=6001, name="stage-motion")
        listener.start()
        ...
        state = listener.get_state()  # True=unsafe, False=safe, None=no packet yet
        listener.stop()

    Typically created via InterlockConstructor.add_external_listener() so its
    start()/stop() lifecycle is tied to the interlock server's, rather than
    managed by hand.
    """

    def __init__(
        self,
        port: int,
        host: str = "",
        decode: Callable[[bytes], bool | None] = default_decode,
        name: str | None = None,
    ):
        """
        Initialize the listener. Does not open a socket until start() is called.

        Args:
        ----
            port: UDP port to listen on.
            host: Local interface to bind to (default '' = all interfaces on
                this machine). Must be an address assigned to one of THIS
                machine's own network interfaces, not the sending computer's
                IP - binding to a non-local address fails (WSAEADDRNOTAVAIL /
                WinError 10049 on Windows).
            decode: Function mapping a raw packet payload to True (unsafe),
                False (safe), or None (unrecognized - previous state is kept).
            name: Label used in log messages and diagnostics; defaults to the port.
        """
        self.port = port
        self.host = host
        self.decode = decode
        self.name = name or f"external-status:{port}"

        self._lock = threading.Lock()
        self._state: bool | None = None
        self._last_raw: bytes = b""
        self._last_received_at: float | None = None
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Bind the UDP socket and start the background receive loop."""
        if self._running:
            logger.warning(f"[{self.name}] Listener is already running")
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._socket.bind((self.host, self.port))
        except OSError as e:
            self._socket.close()
            self._socket = None
            raise OSError(
                f"[{self.name}] Could not bind UDP socket to "
                f"host={self.host!r} port={self.port}: {e}. "
                "The host must be a local address on THIS machine (an IP "
                "assigned to one of its own network interfaces) - not the "
                "sending computer's IP. Use '' or '0.0.0.0' to listen on all "
                "local interfaces, which is almost always what you want."
            ) from e
        self._socket.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"[{self.name}] Listening for UDP status packets on {self.host}:{self.port}"
        )

    def stop(self) -> None:
        """Stop the background thread and close the socket."""
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        if self._socket:
            self._socket.close()
            self._socket = None
        logger.info(f"[{self.name}] Stopped listening")

    def _listen_loop(self) -> None:
        while self._running:
            try:
                raw, _addr = self._socket.recvfrom(4096)
            except socket.timeout:  # noqa: UP041
                continue
            except OSError:
                if self._running:
                    logger.error(f"[{self.name}] Socket error while listening")
                break

            try:
                state = self.decode(raw)
            except Exception as e:  # noqa: BLE001
                logger.error(f"[{self.name}] Error decoding packet {raw!r}: {e}")
                state = None

            with self._lock:
                self._last_raw = raw
                self._last_received_at = time.time()
                if state is None:
                    logger.warning(
                        f"[{self.name}] Received unrecognized packet {raw!r} "
                        "- keeping previous state"
                    )
                else:
                    self._state = state
                    logger.info(
                        f"[{self.name}] State updated to "
                        f"{'WARNING (unsafe)' if state else 'SAFE'} from {raw!r}"
                    )

    def get_state(self) -> bool | None:
        """Return the last known state: True=unsafe, False=safe, None=no packet received yet."""
        with self._lock:
            return self._state

    def get_last_received_at(self) -> float | None:
        """Return the time.time() of the last received packet, or None if none yet."""
        with self._lock:
            return self._last_received_at

    def get_diagnostic_info(self) -> str:
        """Describe the last received packet; suitable as a monitor's diagnostic_func."""
        with self._lock:
            if self._last_received_at is None:
                return f"{self.name}: no status packet received yet on port {self.port}"
            age_s = time.time() - self._last_received_at
            if self._state is True:
                label = "WARNING"
            elif self._state is False:
                label = "SAFE"
            else:
                label = "unrecognized"
            return f"{self.name}: last packet {age_s:.1f}s ago: {self._last_raw!r} -> {label}"

    def __enter__(self) -> "ExternalStatusListener":  # noqa: D105, PYI034
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: D105
        self.stop()
        return False
