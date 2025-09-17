"""
Centralized logging utilities for geecs_scanner.

This module configures a single, thread-safe logging pipeline for the entire
process. It uses a QueueHandler → QueueListener to ensure that all producer
threads can safely emit log records without blocking. A MultiplexingHandler
fan-outs those records to global sinks (e.g., console, rotating file) and
temporary per-scan log files.

Key functions
-------------
ensure_logging
    Initialize logging once per process (safe, idempotent).
update_context
    Inject scan_id / shot_id into all log records.
scan_log
    Context manager to attach/detach a per-scan log file in the scan folder.

Typical usage
-------------
>>> from geecs_scanner.logging_setup import ensure_logging, scan_log, update_context
>>> ensure_logging(log_dir="/tmp/geecs_logs")
>>> with scan_log("Scan001", "/tmp/data/Scan001"):
...     update_context(scan_id="Scan001", shot_id="0")
...     logger.info("scan started")
"""

from __future__ import annotations
import logging
import logging.handlers
from pathlib import Path
from threading import Lock
from contextlib import contextmanager
from queue import Queue
from typing import Optional
import atexit


# --- internal globals ---
_QUEUE: Optional[Queue] = None
_LISTENER: Optional[logging.handlers.QueueListener] = None
_MUX: Optional[MultiplexingHandler] = None
_CTX_FILTER: Optional[ContextFilter] = None
_INIT = False
_LOCK = Lock()


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


class ContextFilter(logging.Filter):
    """
    Filter that injects stable keys (scan_id, shot_id) into every record.

    Ensures that formatters referencing %(scan_id)s and %(shot_id)s never
    raise KeyError, even if context has not yet been set.

    Parameters
    ----------
    **base : dict
        Default context values (e.g., {"scan_id": "-", "shot_id": "-"}).
    """

    def __init__(self, **base):
        super().__init__()
        self.base = base

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach context attributes to the record if missing."""
        for k, v in self.base.items():
            if not hasattr(record, k):
                setattr(record, k, v)
        return True


class ScanFilter(logging.Filter):
    """
    Filter that allows only records matching a specific scan_id.

    Useful for per-scan log files so only messages from the current scan
    are written.

    Parameters
    ----------
    scan_id : str
        The scan identifier to filter on.
    """

    def __init__(self, scan_id: str):
        super().__init__()
        self.scan_id = str(scan_id)

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if record.scan_id matches the filter's scan_id."""
        return getattr(record, "scan_id", None) == self.scan_id


# --------------------------------------------------------------------------
# Multiplexing handler (fan-out target for QueueListener)
# --------------------------------------------------------------------------


class MultiplexingHandler(logging.Handler):
    """
    Thread-safe fan-out handler used by the QueueListener to dispatch to sinks.

    Maintains a list of real handlers (file, console, etc.) and forwards each
    record to them, applying their individual filters.
    """

    def __init__(self):
        """Initialize with an empty handler list and a thread lock."""
        super().__init__()
        self._handlers: list[logging.Handler] = []
        self._hlock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Forward record to all registered handlers that accept it."""
        with self._hlock:
            for h in tuple(self._handlers):
                try:
                    # handle() applies handler-level filters + level threshold
                    h.handle(record)
                except Exception:
                    self.handleError(record)

    def add(self, h: logging.Handler) -> None:
        """Register a new handler to receive log records."""
        with self._hlock:
            self._handlers.append(h)

    def remove(self, h: logging.Handler) -> None:
        """Remove and close a previously registered handler."""
        with self._hlock:
            if h in self._handlers:
                self._handlers.remove(h)
        try:
            h.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def init_logging(
    *,
    log_dir: str,
    level: int = logging.INFO,
    console: bool = True,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    base_context: Optional[dict] = None,
) -> None:
    """
    Configure process-wide logging once.

    This sets up:
    - Root logger with a QueueHandler (non-blocking).
    - A QueueListener that consumes from the queue in a background thread.
    - A MultiplexingHandler that fans out to sinks:
        - Rotating file at <log_dir>geecs_scanner.log
        - Console (optional)

    Safe to call multiple times; subsequent calls are ignored.

    Parameters
    ----------
    log_dir : str
        Directory where global logs should be written.
    level : int, default=logging.INFO
        Logging level for all sinks.
    console : bool, default=True
        If True, attach a console StreamHandler.
    max_bytes : int, default=10_000_000
        Maximum file size before rotation.
    backup_count : int, default=5
        Number of rotated log files to keep.
    base_context : dict, optional
        Initial context values (defaults to {"scan_id": "-", "shot_id": "-"}).
    """
    global _INIT, _QUEUE, _LISTENER, _MUX
    if _INIT:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Root logger with QueueHandler
    _QUEUE = Queue(-1)
    root = logging.getLogger()
    root.setLevel(level)
    qh = logging.handlers.QueueHandler(_QUEUE)
    root.handlers[:] = [qh]
    # One shared ContextFilter; apply at handler level so *all* records get i
    global _CTX_FILTER
    _CTX_FILTER = ContextFilter(**(base_context or {"scan_id": "-", "shot_id": "-"}))
    qh.addFilter(_CTX_FILTER)  # inject before enqueue

    # Multiplexer for real sinks
    _MUX = MultiplexingHandler()
    _MUX.addFilter(_CTX_FILTER)  # inject in listener thread before fan-out

    # Global rotating file
    gf = logging.handlers.RotatingFileHandler(
        Path(log_dir, "geecs_scanner.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,
    )
    gf.setLevel(level)
    gf.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s [%(threadName)s] "
            "scan=%(scan_id)s shot=%(shot_id)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _MUX.add(gf)

    # Optional console output
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s - %(message)s", datefmt="%H:%M:%S"
            )
        )
        _MUX.add(ch)

    _LISTENER = logging.handlers.QueueListener(_QUEUE, _MUX, respect_handler_level=True)
    _LISTENER.start()
    atexit.register(_shutdown_logging)
    _INIT = True


def _shutdown_logging() -> None:
    """Stop the QueueListener gracefully at process exit."""
    global _LISTENER
    try:
        if _LISTENER:
            _LISTENER.stop()
    except Exception:
        pass


def ensure_logging(log_dir: Optional[str] = None, **kwargs) -> None:
    """
    Ensure logging is initialized once.

    Useful in entry points (GUI launcher, scripts, notebooks).
    Idempotent: if logging is already configured, this call is a no-op.

    Parameters
    ----------
    log_dir : str, optional
        Directory for global logs. Defaults to $GEECS_LOG_DIR or ~/.geecs/logs.
    **kwargs
        Forwarded to init_logging().
    """
    import os

    if not _INIT:
        log_dir = (
            log_dir
            or os.getenv("GEECS_LOG_DIR")
            or str(Path.home() / ".geecs" / "logs")
        )
        init_logging(log_dir=log_dir, **kwargs)


def update_context(context: dict[str, object]) -> None:
    """
    Update logging context values (e.g., scan_id, shot_id).

    Parameters
    ----------
    context : dict
        Key/value pairs to inject into all future log records.
    """
    global _CTX_FILTER
    if _CTX_FILTER is None:
        # Late init safety: install a new one with defaults + provided context
        _CTX_FILTER = ContextFilter(**{"scan_id": "-", "shot_id": "-", **context})
        logging.getLogger().addFilter(_CTX_FILTER)
    else:
        _CTX_FILTER.base.update(context)


def attach_scan_log(
    scan_id: str, scan_dir: str, filename: str = "scan.log"
) -> logging.Handler:
    """
    Attach a per-scan log file in the scan directory.

    A FileHandler is created at <scan_dir><filename>. A ScanFilter ensures
    only records with the given scan_id are written to it.

    Parameters
    ----------
    scan_id : str
        Identifier of the scan (used for filtering).
    scan_dir : str
        Directory where the scan log file should be created.
    filename : str, default="scan.log"
        File name for the scan log.

    Returns
    -------
    logging.Handler
        The attached FileHandler (must later be detached).
    """
    global _MUX
    Path(scan_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(scan_dir, filename), encoding="utf-8", delay=True)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s [%(threadName)s] "
            "shot=%(shot_id)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    fh.addFilter(ScanFilter(str(scan_id)))
    with _LOCK:
        _MUX.add(fh)
    return fh


def detach_scan_log(h: logging.Handler) -> None:
    """
    Detach and close a previously attached per-scan log handler.

    Parameters
    ----------
    h : logging.Handler
        The handler returned by attach_scan_log().
    """
    global _MUX
    with _LOCK:
        _MUX.remove(h)


@contextmanager
def scan_log(scan_id: str, scan_dir: str, filename: str = "scan.log"):
    """
    Context manager for per-scan logging.

    Attaches a log file <scan_dir>/<filename> that only receives records
    tagged with scan_id. Automatically detaches the handler and resets
    context on exit.
    """
    update_context({"scan_id": scan_id, "shot_id": "-"})
    h = attach_scan_log(scan_id, scan_dir, filename)
    try:
        yield
    finally:
        detach_scan_log(h)
        update_context({"scan_id": "-", "shot_id": "-"})
