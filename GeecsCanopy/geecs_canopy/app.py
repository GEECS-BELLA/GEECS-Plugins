"""Canopy with the GEECS device module at ``/geecs``: the derived image's api command.

``uvicorn geecs_canopy.app:app``: ``canopy_surface.self_test`` first, then
Canopy's app, unpatched, with the module mounted, tagged ``geecs`` (only
``@*`` keys reach it: Canopy's ``MODULES`` is fixed) and given Canopy's
error handler. The one ``Hare`` implementation, over ``canopy.hare.gateway``;
the module joins the broker as an internal client (as Vole does). With
``GEECS_CANOPY_DB_URL`` set and ``GEECS_CANOPY_DB`` not ``off`` (the
directory's switch), the module writes the DB-served devices too.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from geecs_canopy import canopy_surface
from geecs_canopy.hare import HareRefused, Wire, WriteValue
from geecs_canopy.journal import Journal
from geecs_canopy.module import Module, ModuleSettings
from geecs_canopy.routes import build_app

logger = logging.getLogger(__name__)


class _Channel:
    """Hare's ``ClientChannel``: frames to the tracker; never raises (the broker would drop it)."""

    def __init__(self, on_frame: Callable[[dict[str, Wire]], None]) -> None:
        self._on_frame = on_frame

    async def send_json(self, data: Any) -> None:
        try:
            if isinstance(data, dict) and data.get("type") in ("initial", "diff"):
                self._on_frame(data.get("data") or {})
        except Exception:
            logger.exception("GEECS module: a Hare frame was not taken")

    async def close(self) -> None:
        pass


class _GatewayHare:
    """The ``Hare`` Protocol over Canopy's in-process gateway: the broker Hare's own routes use."""

    CLIENT_ID = "geecs-module"

    def __init__(self) -> None:
        from canopy.core.errors import CanopyError
        from canopy.hare import gateway

        self._joined, self._error, self._gateway = False, CanopyError, gateway

    async def get(self, pv: str) -> Wire:
        try:
            return (await self._gateway.get_value(pv)).model_dump(by_alias=True)
        except self._error as exc:
            return {"pv": pv, "value": None, "connected": False, "error": exc.detail}

    async def describe(self, pv: str) -> Wire:
        try:
            return (await self._gateway.get_broker().get_describe(pv)).model_dump(
                by_alias=True
            )
        except self._error:
            return {"pv": pv}

    async def put(
        self, pv: str, value: WriteValue, *, wait: bool, timeout: float
    ) -> None:
        try:
            await self._gateway.get_broker().put(pv, value, wait=wait, timeout=timeout)
        except self._error as exc:
            raise HareRefused(
                exc.status_code, exc.error_code.value, exc.detail
            ) from exc

    async def watch(
        self, pvs: list[str], on_frame: Callable[[dict[str, Wire]], None]
    ) -> list[str]:
        broker = self._gateway.get_broker()
        if not self._joined:
            broker.add_client(self.CLIENT_ID, _Channel(on_frame))
            self._joined = True
        return await broker.subscribe(self.CLIENT_ID, pvs)


def _build() -> Any:
    """Check the surface, then import Canopy's app and mount the module on it."""
    canopy_surface.self_test()
    from canopy.core.errors import register_canopy_error_handler
    from canopy.core.modules import tag_module
    from canopy.core.security import require_read_access
    from canopy.main import app

    settings = ModuleSettings.from_env()
    settings.state_dir.mkdir(exist_ok=True)
    journal = Journal(settings.state_dir / "journal.sqlite3")
    from geecs_canopy.gateway import source_from_env

    module = Module(settings, _GatewayHare(), journal, gateway=source_from_env(settings.experiment))  # fmt: skip
    geecs_app = build_app(module, read_access=require_read_access)
    register_canopy_error_handler(geecs_app)
    tag_module(geecs_app, "geecs")
    app.mount("/geecs", geecs_app)
    canopy_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(parent: Any) -> AsyncIterator[Any]:
        """Run Canopy's lifespan (it binds the broker), then the module's first read in the background."""
        async with canopy_lifespan(parent) as state:
            task = asyncio.create_task(module.start())
            try:
                yield state
            finally:
                task.cancel()

    app.router.lifespan_context = lifespan
    return app


app = _build()
