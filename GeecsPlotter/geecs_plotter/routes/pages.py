"""The one page: a day + scan form, the column pickers, the plot.

The selection lives in the URL (``?day=&scan=&x=&y=``), so a link is the
plot. The page renders the pick lists server-side; its script fetches the
figure from ``/api/figure`` and draws it.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from geecs_plotter import scans
from geecs_plotter.routes import Context

#: s-file columns every scan has, never a useful default Y (nor is any
#: device's ``<device> shot #`` counter).
_BOOKKEEPING = ("Shotnumber", "Bin #")


def register(router: APIRouter, ctx: Context) -> None:
    """Add the page route to *router*."""

    @router.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        day: Optional[date] = None,
        scan: Optional[int] = None,
        x: Optional[str] = None,
        y: Optional[str] = None,
    ) -> HTMLResponse:
        """The plotter page; without a day and scan, the empty form."""
        page = {
            "day": day,
            "scan": scan,
            "error": None,
            "columns": [],
            "x": "",
            "y": "",
        }
        if day is not None and scan is not None:
            try:
                folder = ctx.folder(day, scan)
            except HTTPException as missing:
                page["error"] = missing.detail
            else:
                frame = scans.load_frame(folder)
                picks = scans.columns(frame)
                page["columns"] = picks
                page["shots"] = len(frame)
                # x absent = GEECSplotter's "autofill par1"; x="" = shot number.
                chosen_x = scans.default_x(folder, day, scan, picks) if x is None else x
                page["x"] = chosen_x if chosen_x in picks else ""
                # Default Y: the first measurement, not the bookkeeping.
                others = [
                    c
                    for c in picks
                    if c not in {page["x"], *_BOOKKEEPING} and not c.endswith(" shot #")
                ]
                page["y"] = y if y in picks else (others[0] if others else "")
        return ctx.templates.TemplateResponse(
            request, "index.html", {"experiment": ctx.experiment, **page}
        )
