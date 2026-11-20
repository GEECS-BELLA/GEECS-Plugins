"""The server window (PyQt6, the ``gui`` extra): the Monitor and Builder tabs.

:func:`run_window` blocks on Qt's event loop on the calling (main) thread;
the engine's own threads keep running underneath it.
"""

from __future__ import annotations

import logging
import signal
import sys
from types import TracebackType

from geecs_interlocks.catalog import Catalog
from geecs_interlocks.engine import Engine
from geecs_interlocks.logs import LogBuffer
from geecs_interlocks.sources import PreviewFactory

logger = logging.getLogger(__name__)


def log_uncaught(
    kind: type[BaseException], error: BaseException, trace: TracebackType | None
) -> None:
    """Log an exception that escaped a Qt slot instead of letting PyQt6 abort the process.

    PyQt6 ends the process on such an exception; in the interlock server
    that would end the evaluation and Master Control's stream with it, for
    what is at worst a drawing bug.
    """
    if issubclass(kind, KeyboardInterrupt):
        sys.__excepthook__(kind, error, trace)
        return
    logger.error(
        "window error (the server keeps running)", exc_info=(kind, error, trace)
    )


def run_window(
    engine: Engine,
    buffer: LogBuffer,
    catalog: Catalog,
    make_preview: PreviewFactory | None = None,
    *,
    experiment: str | None = None,
    web_url: str | None = None,
) -> int:
    """Show the server window until it is closed (or Ctrl-C); Qt's exit code."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from geecs_interlocks.gui.theme import STYLESHEET
    from geecs_interlocks.gui.window import ServerWindow

    sys.excepthook = log_uncaught
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = ServerWindow(
        engine, buffer, catalog, make_preview, experiment=experiment, web_url=web_url
    )
    window.show()
    signal.signal(signal.SIGINT, lambda *_: window.close())
    tick = QTimer()  # wakes the interpreter so Ctrl-C reaches the handler
    tick.timeout.connect(lambda: None)
    tick.start(250)
    return app.exec()


__all__ = ["log_uncaught", "run_window"]
