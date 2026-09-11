"""The logbook's HTTP surface, mounted by GEECS-DataPortal at ``/log``.

Following the config-editor precedent (`scan_analysis.config_editor`), this
module exposes a factory returning an :class:`~fastapi.APIRouter` rather
than an app, so the portal owns the process, the port and the unit.

Routes
------
``GET /log/``
    Redirect to today.
``GET /log/day/{day}``
    The day document: every scan folder present for that date.
``GET /log/api/day/{day}``
    The same content as JSON, for OSPREY and for the page's own polling.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from geecs_scan_log.models import DaySummary
from geecs_scan_log.scan_reader import read_day

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_TEMPLATES = _HERE / "templates"
_STATIC = _HERE / "static"

#: How many days either side of the shown date appear in the rail.
_RAIL_SPAN = 14


def _parse_day(raw: str) -> date:
    """Parse a ``YYYY-MM-DD`` path segment, or 400."""
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"day must be YYYY-MM-DD, got {raw!r}"
        ) from exc


def create_log_router(
    experiment: str,
    base_directory: Optional[Union[Path, str]] = None,
) -> APIRouter:
    """Build the logbook router.

    Parameters
    ----------
    experiment : str
        The experiment whose share to read, e.g. ``"Undulator"``. Supplied
        by the host application; this package carries no default, since a
        facility value belongs in the site profile rather than in code.
    base_directory : Path or str, optional
        Override the configured data-share root. Used by tests.

    Returns
    -------
    APIRouter
        Mount it with ``app.include_router(router, prefix="/log")``.
    """
    router = APIRouter()
    templates = Jinja2Templates(directory=str(_TEMPLATES))
    router.mount("/static", StaticFiles(directory=str(_STATIC)), name="log-static")

    def _load(day: date) -> DaySummary:
        """Read one day, turning share trouble into an honest 503."""
        try:
            return read_day(day, experiment, base_directory=base_directory)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — surface, don't 500 blankly
            logger.exception("reading %s failed", day)
            raise HTTPException(
                status_code=503, detail=f"data share unavailable: {exc}"
            ) from exc

    @router.get("/", response_class=RedirectResponse)
    def _today() -> RedirectResponse:
        """Redirect to today's log."""
        return RedirectResponse(url=f"day/{date.today().isoformat()}")

    @router.get("/api/day/{day}")
    def _day_json(day: str) -> DaySummary:
        """Return one day as JSON — the machine peer of the day page."""
        return _load(_parse_day(day))

    @router.get("/day/{day}", response_class=HTMLResponse)
    def _day_page(request: Request, day: str) -> HTMLResponse:
        """Render the day document."""
        when = _parse_day(day)
        summary = _load(when)
        return templates.TemplateResponse(
            request=request,
            name="day.html",
            context={
                "summary": summary,
                "experiment": experiment,
                "rail_days": _rail_days(when),
                "today": date.today(),
            },
        )

    return router


def _rail_days(centre: date) -> list[date]:
    """Return the dates offered in the navigation rail, newest first."""
    from datetime import timedelta

    return [centre - timedelta(days=offset) for offset in range(_RAIL_SPAN)]
