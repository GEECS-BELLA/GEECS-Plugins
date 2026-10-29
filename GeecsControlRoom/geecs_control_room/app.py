"""The app factory: the kit at ``/theme``, the package's static files, the pages, ``/healthz``.

Every page extends ``base.html``, which carries what a page needs to reach
Canopy from the browser: Hare's WebSocket URL and the display's read key,
set here from the settings (condition (a): a read key, never a write key;
every write goes through the module's put route).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from geecs_web_theme.web import (
    ForwardedPrefixMiddleware,
    RevalidatingStaticFiles,
    make_templates,
    mount_theme,
)

from geecs_control_room.module_client import ModuleClient
from geecs_control_room.settings import Settings

_PKG = Path(__file__).resolve().parent


def create_app(settings: Settings, module: ModuleClient | None = None) -> FastAPI:
    """Build the app; ``module`` defaults to a client of ``settings.canopy_url``."""
    module = module or ModuleClient(settings.canopy_url, settings.read_key)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await module.aclose()

    app = FastAPI(title=settings.title, lifespan=lifespan)
    app.add_middleware(ForwardedPrefixMiddleware)
    mount_theme(app)
    static = RevalidatingStaticFiles(directory=str(_PKG / "static"))
    app.mount("/static", static, name="static")
    app.state.settings, app.state.module = settings, module

    def canopy(_: Request) -> dict[str, Any]:
        return {
            "settings": settings,
            "hare_url": settings.browser_url + "/hare/v1/ws/pvs",
        }

    templates = make_templates(_PKG / "templates", context_processors=[canopy])
    app.state.templates = templates

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "experiment": settings.experiment}

    @app.get("/")
    async def home(request: Request) -> Any:
        return templates.TemplateResponse(request, "home.html", {})

    return app
