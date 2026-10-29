"""What the routes and the put path share: settings, the devices from data, the policy, the tracker.

Devices come from the family files and the experiment's two device
documents through ``NativeSource`` (the directory's own join), re-read at
most every ``reload_s``; each newly seen PV is subscribed once, so freshness
is judged continuously. With a gateway source (the DB-served devices, given
by the wrapper when ``GEECS_CANOPY_DB_URL`` is set), its devices join them,
read off the loop every ``gateway_reload_s`` and never waited for: the last
good set stands while the DB is down, and the roster says so. A native
device wins over a DB device of the same name. The GEECS DB is never read
here: the source is a callable.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    gateway_reload_s: float = 60.0  # the DB devices' re-read interval

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
        gateway: Callable[[], list[Resolved]] | None = None,
    ) -> None:
        self.settings, self.hare, self.journal = settings, hare, journal
        self.tracker, self._clock = Tracker(clock), clock
        root = settings.configs_root.joinpath(*EXPERIMENTS_SUBDIR)
        self._source = NativeSource(settings.experiment, root)
        self._read_at, self._error = -math.inf, None
        self._devices: dict[str, Resolved] = {}
        self._gateway, self._gateway_at = gateway, -math.inf
        self._gateway_devices: dict[str, Resolved] = {}
        self._gateway_new = self._gateway_busy = False  # new: a read not yet subscribed
        self._gateway_read: dict[str, Any] = {"readAt": None, "error": "not read yet"}
        self._watched: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Read and subscribe the devices (the DB's too), so judging starts before the first request."""
        try:
            await self.read_gateway()
            await self.devices()
        except Exception:  # a bad document or an unbound broker must not stop Canopy
            logger.exception("GEECS device module: the first device read failed")

    @property
    def sources(self) -> dict[str, Any]:
        """Each extra source's last read: ``{gateway: {readAt, error}}`` when there is one."""
        return {"gateway": dict(self._gateway_read)} if self._gateway else {}

    async def read_gateway(self) -> None:
        """Read the DB devices off the loop; on failure keep the last good set and say why."""
        if self._gateway is None:
            return
        self._gateway_at, self._gateway_busy = self._clock(), True
        try:
            listed = await asyncio.to_thread(self._gateway)
        except SourceError as exc:  # the DB is down: the last set stands
            self._gateway_read["error"] = str(exc)
            return
        finally:
            self._gateway_busy = False
        self._gateway_devices, self._gateway_new = {d.name: d for d in listed}, True
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._gateway_read = {"readAt": now, "error": None}

    def _spawn(self, work) -> None:
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def devices(self) -> dict[str, Resolved]:
        """Return the devices by GEECS name; raise ``DevicesUnavailable`` while unreadable."""
        async with self._lock:
            due = self._clock() - self._gateway_at >= self.settings.gateway_reload_s
            if (
                self._gateway and due and not self._gateway_busy
            ):  # one read at a time: it is slow
                self._gateway_busy = True
                self._spawn(self.read_gateway())
            read = self._clock() - self._read_at >= self.settings.reload_s
            if read:
                self._read_at = self._clock()
                try:
                    reading = asyncio.to_thread(self._source.resolve)
                    listed = await asyncio.wait_for(reading, _READ_TIMEOUT_S)
                    self._devices, self._error = {d.name: d for d in listed}, None
                except (SourceError, TimeoutError) as exc:
                    self._devices, self._error = {}, str(exc) or "timed out"
            merged = self._gateway_devices | self._devices
            if read or self._gateway_new:  # what the broker refused is tried on a read
                self._gateway_new = False
                pvs = {pv for d in merged.values() for pv in watched_pvs(d)}
                if new := sorted(
                    pvs - self._watched
                ):  # seeding is slow: not under the lock
                    self._watched.update(new)
                    self._spawn(self._subscribe(new))
        if self._error is not None:
            raise DevicesUnavailable(self._error)
        return merged

    async def _subscribe(self, pvs: list[str]) -> None:
        """Subscribe; what the broker refused (its caps) is tried again on the next read."""
        try:
            accepted = set(await self.hare.watch(pvs, self.tracker.on_frame))
        except Exception:
            logger.exception("GEECS device module: subscribing %d PVs failed", len(pvs))
            accepted = set()
        self._watched -= set(pvs) - accepted

    async def policy(self) -> OperatorPolicy:
        """Read the operator policy (none: the empty one, deny); raise if it does not validate."""
        path = self.settings.folder / OPERATOR_POLICY_FILENAME

        def read() -> OperatorPolicy:
            return OperatorPolicy.from_path(path) if path.exists() else OperatorPolicy()

        return await asyncio.wait_for(asyncio.to_thread(read), _READ_TIMEOUT_S)
