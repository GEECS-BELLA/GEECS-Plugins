"""The engine: the interlocks file, its source, its monitor, and Master Control's stream, as one unit.

The stream outlives a reload: it reads :meth:`Engine.snapshot`, which goes to
whichever monitor is current, so Master Control stays connected while the
file is re-read. So do the external-signal listeners, which keep the last
state each sender reported. A reload that does not validate, or names a UDP
port that cannot be bound, keeps the old interlocks running and says why.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from geecs_schemas import ExternalCondition, InterlockSet

from geecs_interlocks.conditions import watched
from geecs_interlocks.config import Settings, load_interlocks
from geecs_interlocks.external import ExternalSignals
from geecs_interlocks.monitor import Event, Monitor, Snapshot
from geecs_interlocks.sources import DeviceReading, DeviceSource, Source
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
        for condition in interlock.conditions:
            for device, variable in watched(condition):
                variables = out.setdefault(device, set())
                if variable is not None:
                    variables.add(variable)
    return out


def external_ports(interlocks: InterlockSet) -> set[int]:
    """The UDP ports the armed interlocks in *interlocks* listen on."""
    return {
        c.port
        for i in interlocks.interlocks
        if i.enabled
        for c in i.conditions
        if isinstance(c, ExternalCondition)
    }


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
        Where the stream and the external signals listen, and the wire format.
    """

    def __init__(
        self, path: Path | None, make_source: SourceFactory, settings: Settings
    ) -> None:
        self.path = path
        self.settings = settings
        self._make_source = make_source
        self._lock = threading.Lock()
        self._reloading = (
            threading.Lock()
        )  # one reload at a time (the window runs them on threads)
        interlocks, origin = load_interlocks(path)  # invalid at start: raise
        self._interlocks = interlocks
        self.signals = ExternalSignals(settings.external_host)
        self._monitor = self._build(interlocks, origin)
        self.stream = StatusServer(
            self.snapshot,
            settings.status_host,
            settings.status_port,
            wire_format=settings.wire_format,
        )

    def _build(self, interlocks: InterlockSet, origin: str) -> Monitor:
        return Monitor(
            interlocks,
            self._make_source(interlocks),
            signals=self.signals,
            origin=origin,
        )

    def start(self) -> None:
        """Listen for external signals, start the monitor, then the stream.

        Raises
        ------
        OSError
            A UDP port or the stream's port cannot be bound; nothing is left
            running.
        """
        self.signals.open(external_ports(self._interlocks))
        try:
            self._monitor.start()
            self.stream.start()
        except OSError:
            self._monitor.stop()
            self.signals.stop()
            raise

    def stop(self) -> None:
        """Stop the stream, the monitor and the external-signal listeners."""
        self.stream.stop()
        with self._lock:
            self._monitor.stop()
        self.signals.stop()

    @property
    def interlocks(self) -> InterlockSet:
        """The document the current monitor runs."""
        with self._lock:
            return self._interlocks

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

    def reading(self, device: str) -> DeviceReading:
        """The current monitor's latest reading of *device*."""
        return self.monitor.reading(device)

    def arm(self, name: str | None = None) -> list[str]:
        """Arm the current monitor's ``config_change`` conditions (see :meth:`Monitor.arm`)."""
        return self.monitor.arm(name)

    def reload(self, path: Path | None = None, *, home: Path | None = None) -> str:
        """Re-read the file (or switch to *path*) and swap in a fresh monitor; a line saying what happened.

        An invalid file leaves the running interlocks (and the path) untouched. The new
        monitor starts with every armed interlock waiting, so the verdict is
        unsafe until fresh readings arrive — a reload never carries an old
        OK over.

        *home* is the path a later plain reload reads, when it differs from
        the file loaded now: a Builder save lands in the configurator's inbox
        (which the configurator empties) while the configuration's home is
        the configs repo's copy.
        """
        with self._reloading:
            target = self.path if path is None else path
            return self._reload(target, target if home is None else home)

    def _reload(self, target: Path | None, home: Path | None) -> str:
        before = set(self.signals.ports())
        fresh: Monitor | None = None
        try:
            interlocks, origin = load_interlocks(target)
            ports = external_ports(interlocks)
            self.signals.open(ports)
            fresh = self._build(interlocks, origin)
            fresh.start()
        except Exception as exc:
            logger.exception("reload refused")
            if fresh is not None:
                fresh.stop()
            self.signals.retain(before)  # close only what this reload opened
            return f"reload refused, old interlocks kept: {exc}"
        with self._lock:
            old, self._monitor = self._monitor, fresh
            self._interlocks = interlocks
            self.path = home
        old.stop()
        self.signals.retain(ports)
        message = f"reloaded {len(interlocks.interlocks)} interlock(s) from {origin}"
        logger.info(message)
        return message


__all__ = [
    "Engine",
    "SourceFactory",
    "device_variables",
    "external_ports",
    "geecs_sources",
]
