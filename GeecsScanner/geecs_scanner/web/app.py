"""The app factory and the router factory.

The web glue every GEECS surface shares — the ``X-Forwarded-Prefix``
middleware, the ``/theme`` mount, the templates factory — is imported from
``geecs_web_theme.web`` (the ``web`` extra), never copied here.  This
module adds what is the scanner's own: ``/health`` with the version, the
error taxonomy → HTTP mapping, the named ``/static`` mount the page
addresses through ``url_for(...).path``, ``/openapi.json`` on and the
docs UI off.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from geecs_web_theme.web import ForwardedPrefixMiddleware, mount_theme

from geecs_scanner import __version__
from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.scanner import ScannerService
from geecs_scanner.web import api, events, pages


def create_scanner_router(service: ScannerService) -> APIRouter:
    """The API and the event stream as a router a host could mount under a prefix.

    The page is not in it: it needs the named ``/static`` mount that only
    :func:`create_app` provides.
    """
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

    mount_theme(app)

    # The page's own assets, named so templates can url_for(...).path them.
    app.mount(
        "/static", StaticFiles(directory=str(pages.STATIC_DIR)), name="scanner_static"
    )

    app.include_router(create_scanner_router(service))
    # The page lives on the process, not the router: it addresses its own
    # assets through the named /static mount above, which a host mounting
    # only the router does not have.
    pages.register(app.router, service)

    @app.get("/api", include_in_schema=False)
    def index() -> dict:
        """Where things are."""
        return {
            "service": "geecs-scanner",
            "version": __version__,
            "experiment": service.experiment,
            "api": "/api/status",
            "events": "/api/events",
            "portal": service.portal_url or None,
            "openapi": "/openapi.json",
            "kit": "/theme/kit.html",
        }

    return app
