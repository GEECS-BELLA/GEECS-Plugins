"""The engine: the interlocks file, its source, its monitor, and Master Control's stream, as one unit.

The stream outlives a reload: it reads :meth:`Engine.snapshot`, which goes to
whichever monitor is current, so Master Control stays connected while the
file is re-read. A reload that does not validate keeps the old interlocks
running and says why.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from geecs_schemas import InterlockSet

from geecs_interlocks.conditions import watched
from geecs_interlocks.config import Settings, load_interlocks
from geecs_interlocks.monitor import Event, Monitor, Snapshot
from geecs_interlocks.sources import DeviceSource, Source
from geecs_interlocks.status_server import StatusServer

logger = logging.getLogger(__name__)

#: Builds the source for one interlocks document (a reload builds a new one).
SourceFactory = Callable[[InterlockSet], Source]


def device_variables(interlocks: InterlockSet) -> dict[str, set[str]]:
    """``device → variables`` every armed interlock in *interlocks* reads."""
    out: dict[str, set[str]] = {}
    for interlock in interlocks.interlocks:
        if not interlock.enabled:
            continue
        for device, variable in watched(interlock.condition):
            variables = out.setdefault(device, set())
            if variable is not None:
                variables.add(variable)
    return out


def geecs_sources(
    endpoints: dict[str, tuple[str, int]] | None = None,
) -> SourceFactory:
    """The real source factory: ``DeviceSource`` over the devices each document reads."""

    def make(interlocks: InterlockSet) -> Source:
        return DeviceSource(device_variables(interlocks), endpoints)

    return make


class Engine:
    """Run the interlocks at *path* over sources from *make_source*.

    Parameters
    ----------
    path :
        The interlocks file (``None``: no configs repo — reports ``NO INTERLOCKS``).
    make_source :
        Builds the source for a document.
    settings :
        Where the stream listens.
    """

    def __init__(
        self, path: Path | None, make_source: SourceFactory, settings: Settings
    ) -> None:
        self.path = path
        self.settings = settings
        self._make_source = make_source
        self._lock = threading.Lock()
        interlocks, origin = load_interlocks(path)  # invalid at start: raise
        self._monitor = Monitor(interlocks, make_source(interlocks), origin=origin)
        self.stream = StatusServer(
            self.snapshot, settings.status_host, settings.status_port
        )

    def start(self) -> None:
        """Start the monitor, then the stream (``OSError``: the port is taken)."""
        self._monitor.start()
        self.stream.start()

    def stop(self) -> None:
        """Stop the stream and the monitor."""
        self.stream.stop()
        with self._lock:
            self._monitor.stop()

    @property
    def monitor(self) -> Monitor:
        """The current monitor (replaced by :meth:`reload`)."""
        with self._lock:
            return self._monitor

    def snapshot(self) -> Snapshot:
        """The current monitor's latest snapshot."""
        return self.monitor.snapshot()

    def events(self) -> list[Event]:
        """The current monitor's state changes."""
        return self.monitor.events()

    def reload(self) -> str:
        """Re-read the file and swap in a fresh monitor; a line saying what happened.

        An invalid file leaves the running interlocks untouched. The new
        monitor starts with every armed interlock waiting, so the verdict is
        unsafe until fresh readings arrive — a reload never carries an old
        OK over.
        """
        try:
            interlocks, origin = load_interlocks(self.path)
        except Exception as exc:
            logger.exception("reload refused")
            return f"reload refused, old interlocks kept: {exc}"
        fresh = Monitor(interlocks, self._make_source(interlocks), origin=origin)
        fresh.start()
        with self._lock:
            old, self._monitor = self._monitor, fresh
        old.stop()
        message = f"reloaded {len(interlocks.interlocks)} interlock(s) from {origin}"
        logger.info(message)
        return message


__all__ = ["Engine", "SourceFactory", "device_variables", "geecs_sources"]
