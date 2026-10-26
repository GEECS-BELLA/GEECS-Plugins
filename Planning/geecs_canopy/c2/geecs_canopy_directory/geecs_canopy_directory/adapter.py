"""Raccoon's ``DirectoryAdapter`` over GEECS's device knowledge.

Extras ``experiment``, ``experiments_root``, ``db`` (``on`` | ``off``).  Patterns
are globs (``*`` or ``%`` any run, ``?`` one character): ``list_devices`` and
``list_elements`` glob names, ignoring case; ``list_pvs("device:<glob>")`` /
``("element:<glob>")`` list the PVs of the matching devices / elements in family
order; any other ``list_pvs`` pattern globs PV names case-sensitively, so a
literal name answers itself (Raccoon's ``exists``).  ``element_type`` = ``native``
| ``gateway``; ``tag`` = a family template or DB device type; ``sort_by`` =
``name``.  Anything else, an empty pattern, or ``[ ( |`` is ``InvalidQuery``.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from canopy.raccoon.adapters.base import (
    DEFAULT_PATTERN_METACHARS,
    AdapterHealth,
    AdapterMisconfigured,
    DirectoryUnavailable,
    InvalidQuery,
    RaccoonError,
)
from canopy.raccoon.config import DirectoryInstanceConfig

from geecs_canopy_directory import sources
from geecs_canopy_directory.sources import Device, Family, SourceError


def _matcher(glob: str, *, case: bool) -> re.Pattern[str]:
    if not glob.strip() or any(c in glob for c in "[(|"):
        raise InvalidQuery(f"not a GEECS directory pattern: {glob!r}")
    regex = fnmatch.translate(glob.replace("%", "*"))
    return re.compile(regex, 0 if case else re.IGNORECASE)


class GeecsDirectoryAdapter:
    """Answers Raccoon's three list calls from the registry, families and DB."""

    name: str = "geecs"
    type_name: str = "geecs"
    pattern_metachars = DEFAULT_PATTERN_METACHARS

    def __init__(
        self,
        experiment: str,
        experiments_root: Path,
        *,
        db: bool = True,
        families: dict[str, Family] | None = None,
    ) -> None:
        self._expt, self._root, self._db = experiment, experiments_root, db
        self._families = sources.load_families() if families is None else families
        # GeecsDb is synchronous: an adapter-owned pool (the meme pattern).
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="geecs")

    @classmethod
    def from_config(cls, cfg: DirectoryInstanceConfig) -> GeecsDirectoryAdapter:
        """Build from ``cfg.extras``: experiment, experiments_root, db."""
        extras = {k.lower(): v for k, v in cfg.extras.items()}
        experiment, root = extras.get("experiment"), extras.get("experiments_root")
        db = str(extras.get("db", "on")).lower()
        if not experiment or not root or db not in ("on", "off"):
            raise AdapterMisconfigured(
                f"extras need experiment, experiments_root, db=on|off: {extras}"
            )
        return cls(str(experiment), Path(root), db=db == "on")

    def _load(self, element_type: str | None) -> list[Device]:
        if element_type not in (None, "native", "gateway"):
            raise InvalidQuery(f"element_type is native or gateway: {element_type!r}")
        if element_type == "gateway" and not self._db:
            raise DirectoryUnavailable("the GEECS DB source is off (db=off)")
        native, gateway = [], []
        try:
            if element_type != "gateway":
                native = sources.native_devices(self._expt, self._root, self._families)
            if element_type != "native" and self._db:
                gateway = sources.gateway_devices(self._expt)
        except SourceError as exc:
            raise DirectoryUnavailable(str(exc)) from exc
        return sources.merged(native, gateway)

    async def _names(self, scope: str, pattern: str, tag, sort_by, element_type):
        """One list call: ``scope`` is device, element, or a ``list_pvs`` scope."""
        if sort_by not in (None, "name"):
            raise InvalidQuery(f"sort_by is 'name': {sort_by!r}")
        match = _matcher(pattern, case=scope == "pv")  # PV names keep their case
        loop = asyncio.get_running_loop()
        devices = await loop.run_in_executor(self._pool, self._load, element_type)
        if tag is not None:
            devices = [d for d in devices if d.tag.lower() == tag.lower()]
        if scope == "device":
            names = [d.name for d in devices if match.match(d.name)]
        else:
            names = [
                name
                for d in devices
                if scope != "device:" or match.match(d.name)
                for e in d.elements
                if scope != "element:" or match.match(e.name)
                for name in ([e.name] if scope == "element" else e.pvs)
                if scope not in ("pv", "element") or match.match(name)
            ]
        unique = list(dict.fromkeys(names))
        return sorted(unique) if sort_by == "name" else unique

    async def list_devices(
        self, pattern: str, *, tag=None, sort_by=None, element_type=None
    ) -> list[str]:
        """Device names matching a glob."""
        return await self._names("device", pattern, tag, sort_by, element_type)

    async def list_elements(
        self, pattern: str, *, tag=None, sort_by=None, element_type=None
    ) -> list[str]:
        """Element names matching a glob."""
        return await self._names("element", pattern, tag, sort_by, element_type)

    async def list_pvs(
        self, pattern: str, *, tag=None, sort_by=None, element_type=None
    ) -> list[str]:
        """PV names: of devices, of elements, or matching a PV glob."""
        for scope in ("device:", "element:"):
            if pattern.startswith(scope):
                pattern = pattern[len(scope) :]
                break
        else:
            scope = "pv"
        return await self._names(scope, pattern, tag, sort_by, element_type)

    async def health(self) -> AdapterHealth:
        """Read every enabled source once; any failure is unhealthy."""
        start, error = time.monotonic(), None
        try:
            await self._names("device", "*", None, None, None)
        except RaccoonError as exc:
            error = str(exc)
        return AdapterHealth(
            healthy=error is None,
            latency_ms=(time.monotonic() - start) * 1000,
            last_error=error,
            last_check_ts_ns=time.time_ns(),
        )

    async def close(self) -> None:
        """Release the pool without waiting on a hung DB call."""
        self._pool.shutdown(wait=False, cancel_futures=True)
