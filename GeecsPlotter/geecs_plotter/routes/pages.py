"""The one page: a day + scan form, the column pickers, the plot.

The selection lives in the URL (``?day=&scan=&x=&y=&xf=&yf=&xalias=
&yalias=&xauto=``), so a link is the plot. ``xauto`` defaults on and then
overrides ``x`` with the open scan's variable; a link to a hand-picked X
carries ``xauto=false``, which the page writes on every hand pick.

The page renders the pick lists server-side; its script fetches the
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
        xalias: bool = True,
        yalias: bool = True,
        xauto: bool = True,
        xf: str = "",
        yf: str = "",
    ) -> HTMLResponse:
        """The plotter page; without a day and scan, the empty form."""
        page = {
            "day": day,
            "scan": scan,
            "error": None,
            "columns": [],
            "x": "",
            "y": "",
            "xalias": xalias,
            "yalias": yalias,
            "xauto": xauto,
            "xf": xf,
            "yf": yf,
        }
        if day is not None and scan is not None:
            try:
                run = ctx.run(day, scan)
            except HTTPException as missing:
                page["error"] = missing.detail
            else:
                frame = scans.load_frame(run)
                picks = scans.columns(frame)
                page["columns"] = picks
                page["xnames"] = {c: scans.display_name(c, xalias) for c in picks}
                page["ynames"] = {c: scans.display_name(c, yalias) for c in picks}
                page["shots"] = len(frame)
                # GEECSplotter's "autofill par1": with xauto (the default) X is
                # the scan variable of whichever scan is open. With it off, x
                # absent is the scan variable too and x="" is the shot number.
                chosen_x = scans.default_x(run, picks) if xauto or x is None else x
                page["x"] = chosen_x if chosen_x in picks else ""
                # Default Y: the first measurement, not the bookkeeping.
                others = [
                    c
                    for c in picks
                    if c not in {page["x"], *_BOOKKEEPING} and not c.endswith(" shot #")
                ]
                # A scan stopped before its first shot has no picks at all.
                page["y"] = y if y in picks else next(iter(others or picks), "")
        return ctx.templates.TemplateResponse(
            request, "index.html", {"experiment": ctx.experiment, **page}
        )
