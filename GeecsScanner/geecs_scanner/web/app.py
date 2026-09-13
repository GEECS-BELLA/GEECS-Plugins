"""The app factory and the router factory.

Kept from GEECS-DataPortal verbatim: the forwarded-prefix middleware
(``X-Forwarded-Prefix`` → ``scope["root_path"]``, path re-prefixed so
Starlette's strip is exact), the ``/theme`` mount from
``geecs_web_theme.static_dir()`` so the kit and the reference page are
served from this origin too, ``/health`` with the version, ``/openapi.json``
on and the docs UI off.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from geecs_scanner import __version__
from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.scanner import ScannerService
from geecs_scanner.web import api, events

#: Requests carrying a proxy mount prefix — the Grafana/JupyterHub
#: convention every reverse proxy speaks.
_FORWARDED_PREFIX_HEADER = b"x-forwarded-prefix"
_PREFIX_RE = re.compile(r"(?:/[A-Za-z0-9._~%@+-]+)+")


def _clean_prefix(raw: str) -> str:
    prefix = raw.strip().rstrip("/")
    if not prefix or _PREFIX_RE.fullmatch(prefix) is None:
        return ""
    return prefix


class ForwardedPrefixMiddleware:
    """Adopt the proxy's ``X-Forwarded-Prefix`` as the ASGI root_path.

    The portal's middleware, unchanged in behaviour: behind
    ``proxy /scan → scanner:8300`` the app never sees its mount point; the
    header names it, and every link and fetch the page builds through the
    one ``root`` value carries it.  The path is re-prefixed so Starlette's
    router strips exactly the prefix and its redirects stay complete.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """Rewrite the scope's root_path and path from the header, then pass through."""
        if scope["type"] == "http":
            for name, value in scope.get("headers", []):
                if name == _FORWARDED_PREFIX_HEADER:
                    prefix = _clean_prefix(value.decode("latin-1"))
                    if prefix:
                        scope["root_path"] = prefix
                        scope["path"] = prefix + scope["path"]
                    break
        await self.app(scope, receive, send)


def create_scanner_router(service: ScannerService) -> APIRouter:
    """The scanner's routes as a router a host could mount under a prefix."""
    router = APIRouter()
    api.register(router, service)
    events.register(router, service)
    return router


def create_app(service: ScannerService, *, root_path: str = "") -> FastAPI:
    """Build the scanner process: the API, the events stream, the theme mount."""
    app = FastAPI(
        title="GEECS Scanner",
        version=__version__,
        root_path=root_path,
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(ForwardedPrefixMiddleware)

    @app.exception_handler(ScannerError)
    async def _scanner_error(_: Request, exc: ScannerError) -> JSONResponse:
        return JSONResponse(exc.to_payload(), status_code=exc.status_code)

    try:
        from geecs_web_theme import static_dir

        app.mount("/theme", StaticFiles(directory=str(static_dir())), name="theme")
    except Exception:  # noqa: BLE001 — the theme is a page concern; the API stands without it
        pass

    app.include_router(create_scanner_router(service))

    @app.get("/")
    def index() -> dict:
        """Where things are, until the page (0.2.0) takes this route."""
        return {
            "service": "geecs-scanner",
            "version": __version__,
            "experiment": service.experiment,
            "api": "/api/status",
            "events": "/api/events",
            "openapi": "/openapi.json",
            "kit": "/theme/kit.html",
        }

    return app
