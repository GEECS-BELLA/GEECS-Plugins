"""Raccoon's ``DirectoryAdapter`` over GEECS's device knowledge.

Loaded by Canopy by dotted path
(``CANOPY_RACCOON__DIRECTORY__TYPE=geecs_canopy.directory:GeecsDirectoryAdapter``);
Canopy is the host process, never a dependency, so its base module is
imported when the adapter is built.

The pattern grammar: globs (``*`` or ``%`` any run, ``?`` one character).
``list_devices`` and ``list_elements`` glob names ignoring case;
``list_pvs("device:<glob>")`` / ``("element:<glob>")`` list the PVs of the
matching devices / elements in family order; any other ``list_pvs``
pattern globs PV names case-sensitively, so a literal name answers itself
(Raccoon's ``exists``). ``element_type`` = ``native`` | ``gateway``;
``tag`` = a family template or a DB device type; ``sort_by`` = ``name``.
Anything else, an empty pattern, or ``[ ( |`` is ``InvalidQuery``.

Failures: the DB branch raises ``DirectoryUnavailable`` (503), so a query
for ``element_type=native`` survives a DB outage; an unscoped query is
never answered in part, since Raccoon caches answers as complete. A
configs folder that does not validate is ``AdapterMisconfigured`` (500).
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Protocol

from geecs_core.configs_repo import EXPERIMENTS_SUBDIR
from geecs_core.pv_naming import normalize_component

from geecs_canopy.native import Device, NativeSource, SourceError


class InstanceConfig(Protocol):
    """What the adapter reads of Canopy's ``DirectoryInstanceConfig``."""

    extras: dict[str, Any]  # Canopy's own free-form dict, lowercased keys


def canopy_base() -> Any:  # canopy.raccoon.adapters.base, untyped here
    """Canopy's adapter base module, or a clear error outside Canopy's environment."""
    try:
        from canopy.raccoon.adapters import base
    except ImportError as exc:
        raise ImportError(
            "geecs_canopy.directory runs inside Canopy: canopy.raccoon.adapters.base "
            "is not importable in this environment"
        ) from exc
    return base


def _matcher(glob: str, *, case: bool, base: Any) -> re.Pattern[str]:
    if not glob.strip() or any(c in glob for c in "[(|"):
        raise base.InvalidQuery(f"not a GEECS directory pattern: {glob!r}")
    regex = fnmatch.translate(glob.replace("%", "*"))
    return re.compile(regex, 0 if case else re.IGNORECASE)


def merged(native: list[Device], gateway: list[Device]) -> list[Device]:
    """Both lists by name; a native device shadows its DB twin (the worker refuses both)."""
    owned = {normalize_component(d.name) for d in native}
    rest = [d for d in gateway if normalize_component(d.name) not in owned]
    return sorted(native + rest, key=lambda d: d.name.lower())


class GeecsDirectoryAdapter:
    """Answers Raccoon's three list calls from the native documents and the DB."""

    name: str = "geecs"
    type_name: str = "geecs"
    pattern_metachars: ClassVar[frozenset[str]] = frozenset("%*?(|[")

    def __init__(
        self,
        native: NativeSource,
        gateway: Any = None,  # gateway.GatewaySource, or None for db=off
        *,
        cache_ttl_s: float = 10.0,
    ) -> None:
        self._base = canopy_base()
        self._native, self._gateway, self._ttl = native, gateway, cache_ttl_s
        # Raccoon asks per uncached literal: each source is read once per TTL.
        self._cache: dict[str, tuple[float, list[Device] | SourceError]] = {}
        # Both sources block (SMB reads, MySQL): a pool and a lock per source,
        # so a hung DB never holds up a native query.
        keys = ("native", "gateway")
        self._locks = {k: threading.Lock() for k in keys}
        self._pools = {k: ThreadPoolExecutor(2, thread_name_prefix=k) for k in keys}

    @classmethod
    def from_config(cls, cfg: InstanceConfig) -> GeecsDirectoryAdapter:
        """Build from ``extras``: experiment, configs_root, db (on|off), cache_ttl_s."""
        base = canopy_base()
        extras = {k.lower(): v for k, v in cfg.extras.items()}
        experiment, root = extras.get("experiment"), extras.get("configs_root")
        db = str(extras.get("db", "on")).lower()
        if not experiment or not root or db not in ("on", "off"):
            raise base.AdapterMisconfigured(
                f"extras need experiment, configs_root and db=on|off: {sorted(extras)}"
            )
        native = NativeSource(str(experiment), Path(root).joinpath(*EXPERIMENTS_SUBDIR))
        gateway = None
        if db == "on":
            from geecs_canopy.gateway import DB_URL_ENV, GatewaySource

            try:
                gateway = GatewaySource(str(experiment), os.environ.get(DB_URL_ENV, ""))
            except ValueError as exc:
                raise base.AdapterMisconfigured(str(exc)) from exc
        return cls(native, gateway, cache_ttl_s=float(extras.get("cache_ttl_s", 10.0)))

    def _read(self, key: str, read) -> list[Device]:
        with self._locks[key]:
            at, value = self._cache.get(key, (float("-inf"), []))
            if time.monotonic() - at >= self._ttl:
                try:
                    value = read()
                except SourceError as exc:
                    value = exc
                self._cache[key] = (time.monotonic(), value)
        if isinstance(value, SourceError):
            raise SourceError(str(value))
        return value

    async def _source(self, key: str, read) -> list[Device]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pools[key], self._read, key, read)

    async def _load(self, element_type: str | None) -> list[Device]:
        base = self._base
        if element_type not in (None, "native", "gateway"):
            raise base.InvalidQuery(
                f"element_type is native or gateway: {element_type!r}"
            )
        native: list[Device] = []
        gateway: list[Device] = []
        if element_type != "gateway":
            try:
                native = await self._source("native", self._native.devices)
            except SourceError as exc:
                raise base.AdapterMisconfigured(str(exc)) from exc
        if element_type != "native":
            if self._gateway is None and element_type == "gateway":
                raise base.DirectoryUnavailable("the GEECS DB source is off (db=off)")
            if self._gateway is not None:
                try:
                    gateway = await self._source("gateway", self._gateway.devices)
                except SourceError as exc:
                    raise base.DirectoryUnavailable(str(exc)) from exc
        return merged(native, gateway)

    async def _names(self, scope, pattern, tag, sort_by, element_type) -> list[str]:
        if sort_by not in (None, "name"):
            raise self._base.InvalidQuery(f"sort_by is 'name': {sort_by!r}")
        match = _matcher(pattern, case=scope == "pv", base=self._base).match
        devices = await self._load(element_type)
        if tag is not None:
            devices = [d for d in devices if d.tag.lower() == tag.lower()]
        elements = [(d, e) for d in devices for e in d.elements]
        if scope == "device":
            names = [d.name for d in devices if match(d.name)]
        elif scope == "element":
            names = [e.name for _, e in elements if match(e.name)]
        elif scope == "device:":
            names = [pv for d, e in elements if match(d.name) for pv in e.pvs]
        elif scope == "element:":
            names = [pv for _, e in elements if match(e.name) for pv in e.pvs]
        else:
            names = [pv for _, e in elements for pv in e.pvs if match(pv)]
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
        scope = next(
            (s for s in ("device:", "element:") if pattern.startswith(s)), "pv"
        )
        if scope != "pv":
            pattern = pattern[len(scope) :]
        return await self._names(scope, pattern, tag, sort_by, element_type)

    async def health(self) -> Any:  # canopy's AdapterHealth
        """Every enabled source, as last read (at most once per ``cache_ttl_s``)."""
        start, error = time.monotonic(), None
        try:
            await self._names("device", "*", None, None, None)
        except self._base.RaccoonError as exc:
            error = str(exc)
        return self._base.AdapterHealth(
            healthy=error is None,
            latency_ms=(time.monotonic() - start) * 1000,
            last_error=error,
            last_check_ts_ns=time.time_ns(),
        )

    async def close(self) -> None:
        """Release the pool without waiting on a hung DB call."""
        for pool in self._pools.values():
            pool.shutdown(wait=False, cancel_futures=True)
