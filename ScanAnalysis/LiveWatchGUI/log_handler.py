"""Custom logging handler that bridges Python logging to Qt signals.

Captures log records from the ``scan_analysis`` loggers and emits them as
Qt signals so the GUI can display live log output without blocking the UI
thread.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from PyQt5.QtCore import QObject, pyqtSignal


class _LogSignalEmitter(QObject):
    """Thin QObject whose only job is to own a signal."""

    log_record = pyqtSignal(str, str)  # (formatted_message, level_name)


class QtLogHandler(logging.Handler):
    """Logging handler that forwards records to a Qt signal.

    Attach this handler to any Python logger; each emitted record is
    forwarded as a ``(formatted_message, level_name)`` signal that the
    GUI can connect to for display.

    Parameters
    ----------
    level : int
        Minimum log level to forward (default ``logging.DEBUG``).

    Examples
    --------
    >>> handler = QtLogHandler()
    >>> handler.emitter.log_record.connect(my_slot)
    >>> logging.getLogger("scan_analysis").addHandler(handler)
    """

    def __init__(self, level: int = logging.DEBUG):
        super().__init__(level)
        self.emitter = _LogSignalEmitter()
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Format *record* and emit it via the Qt signal."""
        try:
            msg = self.format(record)
            self.emitter.log_record.emit(msg, record.levelname)
        except Exception:  # pragma: no cover – never let logging crash the app
            self.handleError(record)
