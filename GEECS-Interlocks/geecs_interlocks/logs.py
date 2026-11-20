"""The server's log: the terminal, a daily file on disk, and a buffer the window shows.

Every record goes to all three, so the window's log pane shows exactly what
the terminal prints and the file keeps it after the window closes:

- the file is ``interlocks-YYYY-MM-DD.log`` in ``config.ini [interlocks]
  log_dir`` (default :func:`default_log_dir`), a new one each day, the ones
  older than :data:`KEEP_DAYS` deleted at each change of day;
- :class:`LogBuffer` keeps the last :data:`BUFFER_LINES` records in memory
  and tells its listeners about each new one (the window's pane).

Nothing here imports Qt.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

#: Days of log files kept.
KEEP_DAYS = 30
#: Records the in-memory buffer keeps.
BUFFER_LINES = 5000
#: Every handler's line format (the terminal's, the file's, the pane's).
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
FILE_PREFIX = "interlocks-"


def default_log_dir() -> Path:
    """Where the log files go when ``config.ini`` names no folder: beside the config."""
    return Path.home() / ".config" / "geecs_python_api" / "interlock_logs"


class DailyFileHandler(logging.Handler):
    """Write records to ``<folder>/interlocks-<date>.log``, a new file each day.

    Parameters
    ----------
    folder :
        Created if missing.
    keep_days :
        Files older than this are deleted when the day changes (and at start).
    today :
        ``datetime.date.today`` stand-in for tests.
    """

    def __init__(
        self,
        folder: Path,
        keep_days: int = KEEP_DAYS,
        today: Callable[[], dt.date] = dt.date.today,
    ) -> None:
        super().__init__()
        self.folder = folder
        self._keep_days = keep_days
        self._today = today
        self._day: dt.date | None = None
        self._stream = None
        folder.mkdir(parents=True, exist_ok=True)

    def path_for(self, day: dt.date) -> Path:
        """The file of *day*."""
        return self.folder / f"{FILE_PREFIX}{day.isoformat()}.log"

    def emit(self, record: logging.LogRecord) -> None:
        """Append *record* to today's file (opening it, and pruning, on a new day)."""
        try:
            day = self._today()
            if day != self._day:
                self._open(day)
            assert self._stream is not None
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except (OSError, ValueError, TypeError):
            self.handleError(record)

    def _open(self, day: dt.date) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = self.path_for(day).open("a", encoding="utf-8")
        self._day = day
        self.prune(day)

    def prune(self, day: dt.date) -> list[Path]:
        """Delete the files older than ``keep_days`` before *day*; the ones deleted."""
        oldest = day - dt.timedelta(days=self._keep_days)
        gone = []
        for path in self.folder.glob(f"{FILE_PREFIX}*.log"):
            try:
                stamp = dt.date.fromisoformat(path.stem[len(FILE_PREFIX) :])
            except ValueError:
                continue  # not one of ours
            if stamp < oldest:
                path.unlink(missing_ok=True)
                gone.append(path)
        return gone

    def close(self) -> None:
        """Close today's file."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


class LogBuffer(logging.Handler):
    """The last *size* records in memory, and a callback per new one.

    Listeners are called on the logging thread; a window hands the record to
    its own thread (a queued signal) before touching a widget. A listener
    that raises is dropped, and its error logged once.
    """

    def __init__(self, size: int = BUFFER_LINES) -> None:
        super().__init__()
        self._records: deque[logging.LogRecord] = deque(maxlen=size)
        self._listeners: list[Callable[[logging.LogRecord], None]] = []
        self._buffer_lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """Keep *record* and pass it to every listener."""
        with self._buffer_lock:
            self._records.append(record)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(record)
            except Exception:
                # Drop it first: the report below passes through this buffer too.
                self.unsubscribe(listener)
                logging.getLogger(__name__).exception("log listener dropped")

    def records(self) -> list[logging.LogRecord]:
        """Every record kept, oldest first."""
        with self._buffer_lock:
            return list(self._records)

    def clear(self) -> None:
        """Forget every record kept."""
        with self._buffer_lock:
            self._records.clear()

    def subscribe(self, listener: Callable[[logging.LogRecord], None]) -> None:
        """Call *listener* with every new record."""
        with self._buffer_lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[logging.LogRecord], None]) -> None:
        """Stop calling *listener*."""
        with self._buffer_lock:
            if listener in self._listeners:
                self._listeners.remove(listener)


def configure_logging(
    level: str = "INFO", log_dir: Path | None = None
) -> tuple[LogBuffer, DailyFileHandler | None]:
    """Send every record to the terminal, the daily file and a :class:`LogBuffer`.

    A folder that cannot be created is reported on the terminal and the file
    is skipped: the server still runs.
    """
    formatter = logging.Formatter(FORMAT)
    root = logging.getLogger()
    root.setLevel(level.upper())
    terminal = logging.StreamHandler()
    buffer = LogBuffer()
    handlers: list[logging.Handler] = [terminal, buffer]
    file_handler = None
    folder = log_dir or default_log_dir()
    try:
        file_handler = DailyFileHandler(folder)
        handlers.append(file_handler)
    except OSError as exc:
        root.addHandler(terminal)
        logging.getLogger(__name__).warning("no log file (%s): %s", folder, exc)
        root.removeHandler(terminal)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    if file_handler is not None:
        logging.getLogger(__name__).info("log file folder: %s", folder)
    return buffer, file_handler


__all__ = [
    "BUFFER_LINES",
    "FORMAT",
    "KEEP_DAYS",
    "DailyFileHandler",
    "LogBuffer",
    "configure_logging",
    "default_log_dir",
]
