"""What the routes and the put path share: settings, the devices from data, the policy, the tracker.

Devices come from the family files and the experiment's two device
documents through ``NativeSource`` (the directory's own join), re-read at
most every ``reload_s``; each newly seen PV is subscribed once, so freshness
is judged continuously. The GEECS DB is never read here.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from geecs_core.configs_repo import EXPERIMENTS_SUBDIR
from geecs_schemas import OPERATOR_POLICY_FILENAME, OperatorPolicy

from geecs_canopy.freshness import Tracker, watched_pvs
from geecs_canopy.hare import Hare
from geecs_canopy.journal import Journal
from geecs_canopy.native import NativeSource, Resolved, SourceError

logger = logging.getLogger(__name__)
_READ_TIMEOUT_S = 5.0  # a hung configs share must not hang Hare's event loop


@dataclass(frozen=True)
class ModuleSettings:
    """Env ``GEECS_CANOPY_<FIELD>``: the configs repo's root, the journal's folder, the timings."""

    experiment: str
    configs_root: Path
    state_dir: Path
    stale_after_s: float = 3.0  # a counter still this long reads stale
    put_timeout_s: float = 5.0  # a setting's put-callback
    settle_timeout_s: float = 30.0  # a move's: not Hare's 5 s connect timeout
    reload_s: float = 10.0  # the device documents' re-read interval

    @property
    def folder(self) -> Path:
        """The experiment's configs folder."""
        return self.configs_root.joinpath(*EXPERIMENTS_SUBDIR, self.experiment)

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> ModuleSettings:
        """Read every set ``GEECS_CANOPY_<FIELD>``; the first three are required."""
        values = {}
        for f in fields(cls):
            if (raw := env.get(f"GEECS_CANOPY_{f.name.upper()}")) is not None:
                values[f.name] = {"str": str, "Path": Path}.get(str(f.type), float)(raw)
        return cls(**values)


class DevicesUnavailable(Exception):
    """The device documents cannot be read or do not validate."""


class Module:
    """The module's state, one per process."""

    def __init__(
        self,
        settings: ModuleSettings,
        hare: Hare,
        journal: Journal,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings, self.hare, self.journal = settings, hare, journal
        self.tracker, self._clock = Tracker(clock), clock
        root = settings.configs_root.joinpath(*EXPERIMENTS_SUBDIR)
        self._source = NativeSource(settings.experiment, root)
        self._read_at, self._error = -math.inf, None
        self._devices: dict[str, Resolved] = {}
        self._watched: set[str] = set()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Read and subscribe the devices, so judging starts before the first request."""
        try:
            await self.devices()
        except Exception:  # a bad document or an unbound broker must not stop Canopy
            logger.exception("GEECS device module: the first device read failed")

    async def devices(self) -> dict[str, Resolved]:
        """Return the devices by GEECS name; raise ``DevicesUnavailable`` while unreadable."""
        async with self._lock:
            if self._clock() - self._read_at >= self.settings.reload_s:
                self._read_at = self._clock()
                try:
                    read = asyncio.to_thread(self._source.resolve)
                    listed = await asyncio.wait_for(read, _READ_TIMEOUT_S)
                    self._devices, self._error = {d.name: d for d in listed}, None
                except (SourceError, TimeoutError) as exc:
                    self._devices, self._error = {}, str(exc) or "timed out"
                pvs = {pv for d in self._devices.values() for pv in watched_pvs(d)}
                if new := sorted(pvs - self._watched):
                    await self.hare.watch(new, self.tracker.on_frame)
                    self._watched.update(new)
        if self._error is not None:
            raise DevicesUnavailable(self._error)
        return self._devices

    async def policy(self) -> OperatorPolicy:
        """Read the operator policy (none: the empty one, deny); raise if it does not validate."""
        path = self.settings.folder / OPERATOR_POLICY_FILENAME

        def read() -> OperatorPolicy:
            return OperatorPolicy.from_path(path) if path.exists() else OperatorPolicy()

        return await asyncio.wait_for(asyncio.to_thread(read), _READ_TIMEOUT_S)
