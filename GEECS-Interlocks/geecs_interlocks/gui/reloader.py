"""Reloading the engine off the window's thread.

A reload stops the old monitor and its device source, which can wait out a
device lookup that is stuck (a DB out of reach blocks for over a minute):
run on the window's thread, it would freeze the window. :class:`Reloader`
runs it on a worker thread and hands the result back as a Qt signal, which
arrives on the window's thread.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from geecs_interlocks.engine import Engine


class Reloader(QObject):
    """Run ``engine.reload(path)`` on a worker thread; its callback runs on the window's thread."""

    # (callback, message): queued from the worker, delivered on the window's thread.
    _done = pyqtSignal(object, str)

    def __init__(self, engine: Engine, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._done.connect(self._deliver)

    def reload(
        self, path: Path | None = None, then: Callable[[str], None] | None = None
    ) -> None:
        """Reload (onto *path*, if given); call *then* with the message on the window's thread."""

        def work() -> None:
            self._done.emit(then, self._engine.reload(path))

        threading.Thread(target=work, name="interlock-reload", daemon=True).start()

    def _deliver(self, then: Callable[[str], None] | None, message: str) -> None:
        if then is not None:
            then(message)


__all__ = ["Reloader"]
