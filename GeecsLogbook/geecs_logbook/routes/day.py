"""The scans book: the day document and its JSON peers.

``GET /``                         redirect to today
``GET /today``                    the same, as a bookmarkable name
``GET /day/{day}``                the day document
``GET /api/day/{day}``            the derived half (scan folders) as JSON
``GET /api/day/{day}/entries``    the commentary half as JSON (both books)

The page's own assets are the app's named ``/static`` mount
(:func:`geecs_logbook.app.create_app`), not a route here.
"""

from __future__ import annotations

from datetime import date, timedelta

from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from geecs_schemas.log_entry import Book, LogEntry
from geecs_web_theme.web import root_of

from geecs_logbook.models import DaySummary
from geecs_logbook.routes.attachments import ATTACHMENT_TYPES
from geecs_logbook.routes._common import (
    Context,
    RenderedEntry,
    api_base,
    month_url,
    parse_day,
    rail_days,
)


def register(router: APIRouter, ctx: Context) -> None:
    """Add the day routes to ``router``."""

    def grouped(request: Request, day: str) -> dict[str, list[RenderedEntry]]:
        """The scans book's entries for a day, rendered and keyed by anchor."""
        if ctx.store is None:
            return {}
        out: dict[str, list[RenderedEntry]] = {}
        for r in ctx.rendered(request, ctx.store.for_day(day, book="scans")):
            out.setdefault(r.entry.anchor, []).append(r)
        return out

    @router.get("/", response_class=RedirectResponse)
    def _today(request: Request) -> RedirectResponse:
        """Redirect to today's log.

        An absolute Location under the request's prefix, not ``day/…``: a
        relative one resolves against the browser's URL, which is right
        for ``/log/`` and wrong for ``/log`` (what a person types) — that
        one landed at the front door's root.
        """
        return RedirectResponse(
            url=f"{root_of(request)}/day/{date.today().isoformat()}"
        )

    @router.get("/today", response_class=RedirectResponse)
    def _today_named(request: Request) -> RedirectResponse:
        """Redirect to today's log — a name a bookmark or a link can use."""
        return RedirectResponse(
            url=f"{root_of(request)}/day/{date.today().isoformat()}"
        )

    @router.get("/day/{day}", response_class=HTMLResponse)
    def _day_page(request: Request, day: str) -> HTMLResponse:
        """Render the day document."""
        when = parse_day(day)
        summary = ctx.load_day(when)
        ctx.maybe_sync()
        entries = grouped(request, day)
        # The other book's count is the cross-link: "3 ops notes today →".
        ops_count = len(ctx.store.for_day(day, book="ops")) if ctx.store else 0
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
                "ops_count": ops_count,
                "month_url": month_url(request, when),
                "ops_url": month_url(request, when, anchor=True),
                "writable": ctx.writable,
                "api_base": api_base(request),
                "accept": ",".join(sorted(ATTACHMENT_TYPES)),
                "seeds": ctx.page_seeds("scans"),
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
