"""The ops book: a month of day-level entries, read from the store alone.

``GET /log/month/{YYYY-MM}``                 the month page (``?tag=`` filters)
``GET /log/api/month/{YYYY-MM}/entries``     the month's entries as JSON

The scans book is a day document over scan folders; the ops book is what
happened around them — laser notes, maintenance, a shift handover — and
it reads by month. Nothing here touches the share: the page is one
database query, grouped by day, newest day first. That is its value on a
bad day: when the share is slow the month page is not.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from geecs_schemas.log_entry import Book, LogEntry

from geecs_logbook.routes.attachments import ATTACHMENT_TYPES
from geecs_logbook.routes._common import (
    Context,
    RenderedEntry,
    api_base,
    month_last_day,
    month_step,
    parse_month,
)

#: The book the month page shows. The scans book has its own page.
BOOK = "ops"


def register(router: APIRouter, ctx: Context) -> None:
    """Add the month routes to ``router``."""

    def month_entries(first: date, last: date) -> list[LogEntry]:
        if ctx.store is None:
            return []
        return ctx.store.query(
            day_from=first.isoformat(), day_to=last.isoformat(), book=BOOK
        )

    @router.get("/month/{month}", response_class=HTMLResponse)
    def _month_page(
        request: Request, month: str, tag: Optional[str] = Query(None, max_length=32)
    ) -> HTMLResponse:
        """Render one month of the ops book."""
        first = parse_month(month)
        last = month_last_day(first)
        today = date.today()
        # No ctx.maybe_sync() here, deliberately: the sync writes to the
        # share on the request thread, and this page's promise is that the
        # share is never on its path. The day page pays the mirror debt.

        # One query for the month; the tag chips count the whole month
        # while the list shows the filtered part, so a chip never reads
        # "0" for a tag that is right there.
        everything = month_entries(first, last)
        tag_counts = Counter(t for e in everything for t in e.tags)
        active = tag.lower() if tag else None
        shown = [e for e in everything if active is None or active in e.tags]

        days: dict[str, list[RenderedEntry]] = {}
        for r in ctx.rendered(request, shown):
            days.setdefault(r.entry.day, []).append(r)
        grouped = [
            (date.fromisoformat(d), rs) for d, rs in sorted(days.items(), reverse=True)
        ]

        if first <= today <= last:
            compose_day = today
        elif today < first:
            compose_day = first
        else:
            compose_day = last

        return ctx.templates.TemplateResponse(
            request=request,
            name="month.html",
            context={
                "experiment": ctx.experiment,
                "first": first,
                "last": last,
                "prev_month": month_step(first, -1),
                "next_month": month_step(first, 1),
                "today": today,
                "days": grouped,
                "entry_count": len(shown),
                "month_count": len(everything),
                "tags": sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])),
                "active_tag": active,
                "compose_day": compose_day,
                "writable": ctx.writable,
                "api_base": api_base(request),
                "accept": ",".join(sorted(ATTACHMENT_TYPES)),
                "seeds": ctx.page_seeds(BOOK),
            },
        )

    @router.get("/api/month/{month}/entries")
    def _month_json(
        month: str,
        book: Book = BOOK,
        tag: Optional[str] = Query(None, max_length=32),
    ) -> list[LogEntry]:
        """Return one month's entries as JSON, oldest first.

        ``?book=scans`` reads the other book (day-level and scan-anchored
        alike); ``?tag=`` narrows to entries carrying that tag.
        """
        first = parse_month(month)
        if ctx.store is None:
            return []
        return ctx.store.query(
            day_from=first.isoformat(),
            day_to=month_last_day(first).isoformat(),
            book=book,
            tag=tag,
        )
