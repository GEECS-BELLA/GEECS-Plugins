"""The scans book: the day document and its JSON peers.

``GET /log/``                         redirect to today
``GET /log/static/{name}``            the page's own assets
``GET /log/day/{day}``                the day document
``GET /log/api/day/{day}``            the derived half (scan folders) as JSON
``GET /log/api/day/{day}/entries``    the commentary half as JSON (both books)
"""

from __future__ import annotations

from datetime import date, timedelta

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from geecs_schemas.log_entry import Book, LogEntry

from geecs_logbook.models import DaySummary
from geecs_logbook.render import render_markdown
from geecs_logbook.routes.attachments import ATTACHMENT_TYPES
from geecs_logbook.routes._common import (
    STATIC_DIR,
    Context,
    RenderedEntry,
    api_base,
    attachment_base,
    parse_day,
    rail_days,
)


def register(router: APIRouter, ctx: Context) -> None:
    """Add the day routes to ``router``."""

    def grouped(request: Request, day: str) -> dict[str, list[RenderedEntry]]:
        """The scans book's entries for a day, rendered and keyed by anchor."""
        if ctx.store is None:
            return {}
        base = attachment_base(request)
        out: dict[str, list[RenderedEntry]] = {}
        for entry in ctx.store.for_day(day, book="scans"):
            html = render_markdown(entry.body_md, attachment_base=base)
            out.setdefault(entry.anchor, []).append(RenderedEntry(entry, html))
        return out

    @router.get("/", response_class=RedirectResponse)
    def _today() -> RedirectResponse:
        """Redirect to today's log."""
        return RedirectResponse(url=f"day/{date.today().isoformat()}")

    @router.get("/static/{name}")
    def _static(name: str) -> FileResponse:
        """Serve the page's own assets.

        A plain route rather than ``router.mount(StaticFiles(...))``:
        ``Mount`` is a ``BaseRoute``, not a ``Route``, and
        ``APIRouter.include_router`` drops it silently on the FastAPI
        versions this package's floor allows — the stylesheet would 404
        and ``url_for`` would raise, with CI green because the lock pins a
        newer release. ``scan_analysis.config_editor`` solved it this way
        first; this follows it.
        """
        target = (STATIC_DIR / name).resolve()
        if target.parent != STATIC_DIR.resolve() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"no such asset: {name}")
        return FileResponse(target)

    @router.get("/day/{day}", response_class=HTMLResponse)
    def _day_page(request: Request, day: str) -> HTMLResponse:
        """Render the day document."""
        when = parse_day(day)
        summary = ctx.load_day(when)
        ctx.maybe_sync()
        entries = grouped(request, day)
        return ctx.templates.TemplateResponse(
            request=request,
            name="day.html",
            context={
                "summary": summary,
                "experiment": ctx.experiment,
                "rail_days": rail_days(when),
                "today": date.today(),
                "prev_day": when - timedelta(days=1),
                "next_day": when + timedelta(days=1),
                "entries": entries,
                "entry_count": sum(len(v) for v in entries.values()),
                "writable": ctx.writable,
                "api_base": api_base(request),
                "accept": ",".join(sorted(ATTACHMENT_TYPES)),
            },
        )

    @router.get("/api/day/{day}")
    def _day_json(day: str) -> DaySummary:
        """Return one day's derived half as JSON."""
        return ctx.load_day(parse_day(day))

    @router.get("/api/day/{day}/entries")
    def _entries_json(day: str, book: Optional[Book] = None) -> list[LogEntry]:
        """Return one day's commentary as JSON; ``?book=`` narrows to one book."""
        parse_day(day)
        if ctx.store is None:
            return []
        return ctx.store.for_day(day, book=book)
