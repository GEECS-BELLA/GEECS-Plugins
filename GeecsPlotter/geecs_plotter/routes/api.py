"""The JSON routes: a scan's columns, the figure for a pick, the settings sets.

Each is a few calls into :mod:`geecs_plotter.scans`,
:mod:`geecs_plotter.figures` or :mod:`geecs_plotter.settings`; the figure
route also returns ``code``, the notebook snippet that rebuilds the same
frame and figure.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from geecs_plotter import figures, scans
from geecs_plotter.routes import Context
from geecs_plotter.routes.pages import read_form
from geecs_plotter.settings import PlotterSettings, SettingsError, merged, set_name

#: The s-file grows after a scan (analysis appends columns), so no answer
#: is cached.
NO_CACHE = {"Cache-Control": "no-cache"}


class ColumnsResponse(BaseModel):
    """A scan's plot picks."""

    columns: list[str]
    default_x: str
    shots: int


class SetsResponse(BaseModel):
    """The settings folder and the sets saved in it."""

    folder: str
    origin: str
    sets: list[str]


class FigureResponse(BaseModel):
    """One plot: the figure JSON and the snippet that reproduces it."""

    x: str
    y: str
    shots: int
    # The plotly figure JSON: the one free dict in a response (the brief).
    figure: dict[str, Any]
    code: str


def figure_code(
    day: date,
    scan: int,
    y: str,
    x: Optional[str],
    experiment: str = "",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> str:
    """The notebook snippet that rebuilds the frame and the figure.

    Parameters
    ----------
    day : datetime.date
        The scan's day.
    scan : int
        The scan number.
    y : str
        The Y column.
    x : str, optional
        The X column; ``None`` plots against the shot number.
    experiment : str, optional
        Named only when the app was started with one.
    x_label, y_label : str, optional
        The axis titles the alias toggles chose, when not the column names.

    Returns
    -------
    str
        Python source.
    """
    where = f", experiment={experiment!r}" if experiment else ""
    names = [y, x] if x else [y]
    x_arg = f", x={x!r}" if x else ""
    labels = f", x_label={x_label!r}" if x_label else ""
    labels += f", y_label={y_label!r}" if y_label else ""
    return (
        "from datetime import date\n"
        "from geecs_plotter import figures, scans\n\n"
        f"run = scans.find_scan(date({day.year}, {day.month}, {day.day}), {scan}{where})\n"
        "frame = scans.load_frame(run)\n"
        f"series = scans.plot_series(frame, {names!r})\n"
        f"fig = figures.shots_figure(series, {y!r}{x_arg}, shot=scans.shot_axis(frame){labels})\n"
        "fig.show()\n"
    )


def register(router: APIRouter, ctx: Context) -> None:
    """Add the JSON routes to *router*."""

    @router.get("/api/columns", response_model=ColumnsResponse)
    def api_columns(day: date, scan: int) -> JSONResponse:
        """The scan's plottable columns and its default X."""
        run = ctx.run(day, scan)
        frame = scans.load_frame(run)
        picks = scans.columns(frame)
        body = ColumnsResponse(
            columns=picks,
            default_x=scans.default_x(run, picks),
            shots=len(frame),
        )
        return _json(body)

    @router.get("/api/figure", response_model=FigureResponse)
    def api_figure(
        day: date,
        scan: int,
        y: str,
        x: str = Query(default="", description="empty plots against the shot number"),
        xalias: bool = Query(default=True, description="title X with its alias"),
        yalias: bool = Query(default=True, description="title Y with its alias"),
    ) -> JSONResponse:
        """The per-shot figure of ``y`` against ``x``."""
        frame = scans.load_frame(ctx.run(day, scan))
        try:
            series = scans.plot_series(frame, [y, x] if x else [y])
        except KeyError as missing:
            raise HTTPException(
                status_code=404, detail=f"no plottable column {missing.args[0]!r}"
            ) from None
        # A title only where the toggle changes it, so the snippet stays bare.
        x_label = scans.display_name(x, xalias) if x else None
        x_label = x_label if x_label != x else None
        y_label = scans.display_name(y, yalias)
        y_label = y_label if y_label != y else None
        fig = figures.shots_figure(
            series,
            y,
            x=x or None,
            shot=scans.shot_axis(frame),
            x_label=x_label,
            y_label=y_label,
            palette=figures.THEMED_PALETTE,
        )
        body = FigureResponse(
            x=x,
            y=y,
            shots=len(frame),
            figure=figures.page_figure(fig),
            code=figure_code(day, scan, y, x or None, ctx.experiment, x_label, y_label),
        )
        return _json(body)

    @router.get("/api/settings", response_model=SetsResponse)
    def api_sets() -> JSONResponse:
        """Where sets are saved, and their names."""
        store = ctx.store
        body = SetsResponse(
            folder=str(store.folder), origin=store.origin, sets=store.names()
        )
        return _json(body)

    @router.get("/api/settings/{name}", response_model=PlotterSettings)
    def api_set(name: str) -> JSONResponse:
        """One set; a name never saved is the defaults."""
        return _json(_settings(lambda: ctx.store.load(name)))

    @router.post("/api/settings/{name}", response_model=PlotterSettings)
    async def api_save(name: str, request: Request) -> JSONResponse:
        """Merge the page's form fields into the set and save it.

        The page saves a filter as it is typed this way, without a reload.
        """
        form = await read_form(request)

        def save() -> PlotterSettings:
            settings = merged(ctx.store.load(name), form)
            ctx.store.save(set_name(name), settings)
            return settings

        return _json(_settings(save))


def _settings(action: Callable[[], PlotterSettings]) -> PlotterSettings:
    """Run a store call; a bad name or file is a 422 naming it."""
    try:
        return action()
    except SettingsError as bad:
        raise HTTPException(status_code=422, detail=str(bad)) from None


def _json(body: BaseModel) -> JSONResponse:
    return JSONResponse(body.model_dump(), headers=NO_CACHE)
