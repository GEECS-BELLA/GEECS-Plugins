"""InterlockServer: TCP server that broadcasts interlock status flags to clients."""

import logging
import signal
import socket
import struct
import threading
import time
from typing import Callable, Dict, List

from .base_interlock import BaseInterlock

logger = logging.getLogger(__name__)


class InterlockServer:
    """TCP server that broadcasts interlock status flags to clients.

    Two registration paths are supported.  Prefer ``register_interlock``
    for new code:

    >>> server = InterlockServer(host="0.0.0.0", port=9999)
    >>> server.register_interlock(
    ...     CameraThresholdInterlock("CAM-PL1-LC_Film", "MeanCounts", 2.0)
    ... )
    >>> server.run_forever()

    The older callable-based path is kept for back-compat with the
    original example notebook:

    >>> server.register_monitor("Camera MaxCounts Check", my_check_fn)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9999):
        self.host = host
        self.port = port
        self.interlock_flags: Dict[str, bool] = {}
        self.flags_lock = threading.Lock()
        self.server_running = False
        self._server_thread = None
        self._monitor_threads: List[threading.Thread] = []

    def set_interlock(self, name: str, is_active: bool):
        """Set the state of an interlock flag and log status transitions.

        Thread-safe; callable from monitor threads or external code.

        Parameters
        ----------
        name : str
            Identifier for the interlock.
        is_active : bool
            True if the interlock condition is unsafe (consumer should
            hold off), False if safe.
        """
        with self.flags_lock:
            old_state = self.interlock_flags.get(name, False)
            self.interlock_flags[name] = is_active
            if is_active != old_state:
                status = "ACTIVE" if is_active else "NOT ACTIVE"
                logger.info(f"[{name}] Interlock {status}")

    def register_interlock(self, interlock: BaseInterlock) -> None:
        """Register a :class:`BaseInterlock` instance.

        The server polls :meth:`BaseInterlock.check` at the interlock's
        ``poll_interval``.  On uncaught exceptions from ``check`` the
        flag is forced to active (unsafe) — same policy as
        :meth:`register_monitor`.
        """

        def monitor_loop():
            self.set_interlock(interlock.name, False)
            while self.server_running:
                try:
                    result = interlock.check()
                    self.set_interlock(interlock.name, result)
                except Exception as e:
                    logger.error(f"Error in interlock '{interlock.name}': {e}")
                    self.set_interlock(interlock.name, True)
                time.sleep(interlock.poll_interval)

        thread = threading.Thread(
            target=monitor_loop, daemon=True, name=f"interlock-{interlock.name}"
        )
        self._monitor_threads.append(thread)
        if self.server_running:
            thread.start()

    def register_monitor(
        self, name: str, check_func: Callable[[], bool], interval: float = 0.5
    ):
        """Register a plain callable as an interlock check (legacy path).

        Kept for backward compatibility with the original example
        notebook.  New code should subclass :class:`BaseInterlock` and
        use :meth:`register_interlock` instead — it gives YAML-driven
        configuration and freshness tracking for free.

        Parameters
        ----------
        name : str
            Identifier for the interlock.
        check_func : Callable[[], bool]
            Returns True when the interlock should trip (unsafe).
        interval : float, optional
            Seconds between checks (default 0.5).
        """

        def monitor_loop():
            self.set_interlock(name, False)
            while self.server_running:
                try:
                    result = check_func()
                    self.set_interlock(name, result)
                except Exception as e:
                    logger.error(f"Error in monitor '{name}': {e}")
                    self.set_interlock(name, True)
                time.sleep(interval)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_threads.append(thread)
        if self.server_running:
            thread.start()

    def get_interlock(self, name: str) -> bool:
        """Get the current state of an interlock flag."""
        with self.flags_lock:
            return self.interlock_flags.get(name, False)

    def get_all_interlocks(self) -> Dict[str, bool]:
        """Get all interlock flags."""
        with self.flags_lock:
            return self.interlock_flags.copy()

    # ----- TCP server internals (wire protocol intentionally unchanged) -----

    def _handle_client(self, conn, addr):
        """Handle a connected client and send status updates."""
        logger.debug(f"Client connected: {addr}")
        try:
            while self.server_running:
                with self.flags_lock:
                    status_lines = []
                    for name, flag in self.interlock_flags.items():
                        status = (
                            "WARNING! Interlock conditions not met." if flag else "SAFE"
                        )
                        status_lines.append(f"{name}: {status}")
                    message = (
                        " | ".join(status_lines)
                        if status_lines
                        else "No monitors active"
                    )

                message_bytes = message.encode("utf-8")
                length_prefix = struct.pack(">I", len(message_bytes))
                conn.sendall(length_prefix + message_bytes)

                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            conn.close()
            logger.debug(f"Client disconnected: {addr}")

    def _server_loop(self):
        """Main server loop."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            logger.info(f"Interlock server listening on {self.host}:{self.port}")

            while self.server_running:
                try:
                    s.settimeout(1.0)
                    conn, addr = s.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client, args=(conn, addr), daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.server_running:
                        logger.error(f"Server error: {e}")

    def start(self):
        """Start the interlock server and monitor threads (non-blocking)."""
        if self.server_running:
            logger.warning("Server is already running")
            return

        self.server_running = True

        for thread in self._monitor_threads:
            thread.start()

        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()

    def stop(self):
        """Stop the interlock server."""
        logger.info("Stopping interlock server...")
        self.server_running = False
        if self._server_thread:
            self._server_thread.join(timeout=2)
        logger.info("Interlock server stopped")

    def run_forever(self) -> None:
        """Start the server and block until SIGINT or SIGTERM.

        Convenience wrapper for CLI / script use — replaces the
        ``while True: time.sleep(1)`` dance after :meth:`start`.  Falls
        back to ``KeyboardInterrupt`` handling on platforms where
        ``signal.signal`` is not available from the calling thread
        (e.g. inside a Jupyter cell).
        """
        stop_event = threading.Event()

        def _on_signal(signum, _frame):
            logger.info(f"Received signal {signum}; shutting down")
            stop_event.set()

        try:
            signal.signal(signal.SIGINT, _on_signal)
            signal.signal(signal.SIGTERM, _on_signal)
        except ValueError:
            # signal.signal only works from the main thread of the main
            # interpreter; fall back to KeyboardInterrupt below.
            pass

        self.start()
        try:
            while not stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
