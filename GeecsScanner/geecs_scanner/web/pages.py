"""The page — one HTML document over the API and the event stream.

Jinja renders the skeleton once; the browser script (``static/scanner.js``)
fills every value from ``/api/*`` and keeps it live over ``/api/events``.
The template reaches the theme through this process's ``/theme`` mount and
its own assets through ``url_for(...).path`` — never an absolute URL, which
behind TLS termination is a mixed-content block.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from geecs_scanner import __version__
from geecs_scanner.service.scanner import ScannerService

PACKAGE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


def _root(request: Request) -> str:
    """The request's URL prefix (``""`` at root) — prepend to every path."""
    return request.scope.get("root_path", "").rstrip("/")


def make_templates() -> Jinja2Templates:
    """The template environment with ``root`` in every context."""
    return Jinja2Templates(
        directory=str(TEMPLATES_DIR),
        context_processors=[lambda request: {"root": _root(request)}],
    )


def register(router: APIRouter, service: ScannerService) -> None:
    """Attach the page route to *router*."""
    templates = make_templates()

    @router.get("/", response_class=HTMLResponse, include_in_schema=False)
    def console(request: Request) -> HTMLResponse:
        """The scanner console."""
        return templates.TemplateResponse(
            request,
            "console.html",
            {
                "experiment": service.experiment,
                "identity": service.identity,
                "version": __version__,
            },
        )
