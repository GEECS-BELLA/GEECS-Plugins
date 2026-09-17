"""The page — one HTML document over the API and the event stream.

Jinja renders the skeleton once; the browser script (``static/scanner.js``)
fills every value from ``/api/*`` and keeps it live over ``/api/events``.
The template reaches the theme through this process's ``/theme`` mount and
its own assets through ``url_for(...).path`` — never an absolute URL, which
behind TLS termination is a mixed-content block.  ``root`` (the proxy
prefix) reaches every context through ``geecs_web_theme.web.make_templates``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from geecs_web_theme.web import make_templates

from geecs_scanner import __version__
from geecs_scanner.service.scanner import ScannerService

PACKAGE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"


def register(router: APIRouter, service: ScannerService) -> None:
    """Attach the page route to *router*."""
    # ``root`` in every context comes from the shared factory.
    templates = make_templates(TEMPLATES_DIR)

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
                "portal": service.portal_url,
            },
        )
