"""The ops book: a month of day-level entries, read from the store alone.

``GET /log/month/today``                     redirect to this month, at today
``GET /log/month/{YYYY-MM}``                 the month page (``?tag=`` filters)
``GET /log/api/month/{YYYY-MM}/entries``     the month's entries as JSON
``GET /log/api/month/{YYYY-MM}/days``        calendar marks: which days have
                                             notes (the store) and a day
                                             folder (one listing of the share)

The scans book is a day document over scan folders; the ops book is what
happened around them — laser notes, maintenance, a shift handover — and
it reads by month. Nothing here touches the share: the page is one
database query, grouped by day, newest day first. That is its value on a
bad day: when the share is slow the month page is not.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from geecs_schemas.log_entry import Book, LogEntry
from pydantic import BaseModel, Field

from geecs_logbook.routes.attachments import ATTACHMENT_TYPES
from geecs_logbook.routes._common import (
    Context,
    RenderedEntry,
    api_base,
    month_last_day,
    month_step,
    month_url,
    parse_month,
)
from geecs_logbook.scan_reader import days_with_folders

logger = logging.getLogger(__name__)

#: The book the month page shows. The scans book has its own page.
BOOK = "ops"


class DayMarks(BaseModel):
    """What a calendar shows for one day."""

    folder: bool = Field(description="A day folder exists on the share.")
    notes: int = Field(0, description="Live entries in the scans book.")
    ops: int = Field(0, description="Live entries in the ops book.")


class MonthMarks(BaseModel):
    """Calendar marks for a month — the lazy fetch behind the rail's calendar."""

    month: str = Field(description="``YYYY-MM``.")
    share: bool = Field(
        description="Whether the share answered; when false, ``folder`` is"
        " unknown for every day and only the store's marks are real."
    )
    days: dict[str, DayMarks] = Field(
        description="Days with anything to mark, keyed ``YYYY-MM-DD``."
    )


def register(router: APIRouter, ctx: Context) -> None:
    """Add the month routes to ``router``."""

    def month_entries(first: date, last: date) -> list[LogEntry]:
        if ctx.store is None:
            return []
        return ctx.store.query(
            day_from=first.isoformat(), day_to=last.isoformat(), book=BOOK
        )

    @router.get("/month/today", response_class=RedirectResponse)
    def _this_month() -> RedirectResponse:
        """Redirect to this month's page, at today's day group."""
        today = date.today()
        return RedirectResponse(
            url=f"{today.strftime('%Y-%m')}#day-{today.isoformat()}"
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
                "today_url": month_url(request, today, anchor=True),
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

    @router.get("/api/month/{month}/days")
    def _month_days(month: str) -> MonthMarks:
        """Return the calendar's marks for one month.

        Two cheap questions: the store's per-day counts (one grouped
        query) and which day folders exist (one listing of the month
        folder — never a walk of the days). The month **page** never calls
        this on its own path; the calendar fetches it when opened, so a
        slow share delays a popup, not a page.
        """
        first = parse_month(month)
        last = month_last_day(first)
        counts = (
            ctx.store.count_by_day(first.isoformat(), last.isoformat())
            if ctx.store is not None
            else {}
        )
        try:
            folders = days_with_folders(first, ctx.experiment, ctx.base_directory)
        except Exception as exc:  # noqa: BLE001 — the same honesty as load_day
            logger.exception("listing the month folder for %s failed", month)
            raise HTTPException(
                status_code=503, detail=f"data share unavailable: {exc}"
            ) from exc
        days: dict[str, DayMarks] = {}
        for day, by_book in counts.items():
            days[day] = DayMarks(
                folder=False,
                notes=by_book.get("scans", 0),
                ops=by_book.get("ops", 0),
            )
        for day in folders or ():
            key = day.isoformat()
            if key in days:
                days[key].folder = True
            else:
                days[key] = DayMarks(folder=True)
        return MonthMarks(
            month=first.strftime("%Y-%m"),
            share=folders is not None,
            days=dict(sorted(days.items())),
        )
