"""The JSON routes: a scan's columns, the figure for a pick, the settings sets.

Each is a few calls into :mod:`geecs_plotter.scans`,
:mod:`geecs_plotter.figures` or :mod:`geecs_plotter.settings`; the figure
route also returns ``code``, the notebook snippet that rebuilds the same
frame and figure.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Callable, Optional, Sequence, Union

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from geecs_data_utils.data.curve_fit import CurveFit
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from geecs_plotter import figures, scans
from geecs_plotter.routes import Context, read_form
from geecs_plotter.settings import (
    MAX_FIT_ORDER,
    PlotterSettings,
    SettingsError,
    merged,
    set_name,
)

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


class FitTerm(BaseModel):
    """One fitted or derived number, its standard error, and both as text."""

    name: str
    #: ``None`` where undefined (NaN is not JSON).
    value: Optional[float]
    stderr: Optional[float]
    #: ``value ± stderr`` rounded to the error's second significant figure.
    text: str


class FitResult(BaseModel):
    """One Y list's fit, as the page lists it under the plot."""

    #: 1, 2 or 3: the Y list (and its trace colour).
    slot: int
    column: str
    model: str
    #: GEECSplotter's name for the model (``figures.FITS``).
    label: str
    formula: str = ""
    #: Why no fit was made; ``None`` when it was.
    error: Optional[str] = None
    r2: Optional[float] = None
    points: int = 0
    terms: list[FitTerm] = []
    notes: list[str] = []


class FigureResponse(BaseModel):
    """One plot: the figure JSON and the snippet that reproduces it."""

    x: str
    y: str
    #: Y2 and Y3, ``""`` when off.
    y2: str = ""
    y3: str = ""
    shots: int
    # The plotly figure JSON: the one free dict in a response (the brief).
    figure: dict[str, Any]
    code: str
    #: One per Y list with a fit picked, in slot order.
    fits: list[FitResult] = []


#: Each model's curve, in its parameters' names (polynomial: see ``_formula``).
FORMULAS = {
    "gaussian": "amplitude·exp(−(x − center)² / (2·sigma²)) + offset",
    "gaussian_no_offset": "amplitude·exp(−(x − center)² / (2·sigma²))",
    "supergaussian": "amplitude·exp(−(|x − center| / width)^power) + offset",
    "supergaussian_no_offset": "amplitude·exp(−(|x − center| / width)^power)",
}


def _formula(fit: CurveFit) -> str:
    if fit.model != "polynomial":
        return FORMULAS.get(fit.model, "")
    # Far from zero (timestamps) the powers are of x - x_shift (data-utils).
    u = f"(x − {fit.x_shift:.15g})" if fit.x_shift else "x"
    terms = [
        "c0",
        f"c1·{u}",
        *(f"c{k}·{u}^{k}" for k in range(2, (fit.order or 0) + 1)),
    ]
    return " + ".join(terms[: (fit.order or 0) + 1])


def _finite(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def plus_minus(value: float, stderr: float) -> str:
    """``value ± stderr``, both rounded to the error's second significant figure.

    No usable error (NaN, zero) shows the value to four significant
    figures; magnitudes past 1e±5 are written in exponent form.
    """
    if not math.isfinite(value):
        return "undefined"
    if not math.isfinite(stderr) or stderr <= 0:
        return f"{value:.4g}"
    exponent = math.floor(math.log10(stderr))
    if abs(exponent) > 5 or abs(value) >= 1e6:
        return f"{value:.3e} ± {stderr:.1e}"
    decimals = 1 - exponent
    if decimals >= 0:
        return f"{value:.{decimals}f} ± {stderr:.{decimals}f}"
    step = 10.0 ** (-decimals)
    return f"{round(value / step) * step:.0f} ± {round(stderr / step) * step:.0f}"


def fit_result(
    slot: int, column: str, model: str, fit: Optional[CurveFit]
) -> FitResult:
    """The page's view of one fit; *fit* ``None`` is a model this copy lacks."""
    label = figures.FITS.get(model, model)
    if fit is None:
        return FitResult(
            slot=slot,
            column=column,
            model=model,
            label=label,
            error=f"this plotter has no fit called {model!r}",
        )
    terms = [
        FitTerm(
            name=p.name,
            value=_finite(p.value),
            stderr=_finite(p.stderr),
            text=plus_minus(p.value, p.stderr),
        )
        for p in (*fit.parameters, *fit.derived)
    ]
    return FitResult(
        slot=slot,
        column=column,
        model=model,
        label=label,
        formula=_formula(fit),
        error=fit.error,
        r2=_finite(fit.r2),
        points=fit.points,
        terms=terms,
        notes=fit.notes,
    )


def figure_code(
    day: date,
    scan: int,
    y: Union[str, Sequence[Optional[str]]],
    x: Optional[str],
    experiment: str = "",
    x_label: Optional[str] = None,
    y_labels: Sequence[Optional[str]] = (),
    fits: Sequence[Optional[str]] = (),
    orders: Sequence[int] = (),
) -> str:
    """The notebook snippet that rebuilds the frame, the fits and the figure.

    Parameters
    ----------
    day : datetime.date
        The scan's day.
    scan : int
        The scan number.
    y : str or Sequence[str or None]
        The Y column, or Y1–Y3 in their slots (``None`` for one off).
    x : str, optional
        The X column; ``None`` plots against the shot number.
    experiment : str, optional
        Named only when the app was started with one.
    x_label : str, optional
        The X title the alias toggle chose, when not the column name.
    y_labels : Sequence[str or None], optional
        The same for each Y axis.
    fits : Sequence[str or None], optional
        The fit per Y slot (``figures.FITS`` keys); none named, no fit code.
    orders : Sequence[int], optional
        The polynomial order per slot.

    Returns
    -------
    str
        Python source.
    """
    ys = [y] if isinstance(y, str) else list(y)
    where = f", experiment={experiment!r}" if experiment else ""
    drawn = [c for c in ys if c]
    names = [*drawn, x] if x else drawn
    y_arg = repr(ys[0]) if len(ys) == 1 else repr(ys)
    x_arg = f", x={x!r}" if x else ""
    labels = f", x_label={x_label!r}" if x_label else ""
    if any(y_labels):
        labels += (
            f", y_label={y_labels[0]!r}"
            if len(ys) == 1
            else f", y_labels={list(y_labels)!r}"
        )
    fitting = ""
    fit_arg = ""
    if any(f and f != "none" for f in fits):
        models = [f if f and f != "none" else None for f in fits]
        order_arg = f", orders={list(orders)!r}" if "polynomial" in models else ""
        fitting = (
            f"fits = figures.fit_columns(series, {ys!r}, {models!r}{x_arg},"
            f" shot=scans.shot_axis(frame){order_arg})\n"
        )
        fit_arg = ", fits=fits"
    return (
        "from datetime import date\n"
        "from geecs_plotter import figures, scans\n\n"
        f"run = scans.find_scan(date({day.year}, {day.month}, {day.day}), {scan}{where})\n"
        "frame = scans.load_frame(run)\n"
        f"series = scans.plot_series(frame, {names!r})\n"
        f"{fitting}"
        f"fig = figures.shots_figure(series, {y_arg}{x_arg}, shot=scans.shot_axis(frame){labels}{fit_arg})\n"
        "fig.show()\n" + ("for fit in fits:\n    print(fit)\n" if fitting else "")
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
        y2: str = Query(default="", description="Y2 on a second axis; empty is off"),
        y3: str = Query(default="", description="Y3 on a third axis; empty is off"),
        xalias: bool = Query(default=True, description="title X with its alias"),
        yalias: bool = Query(default=True, description="title Y with its alias"),
        y2alias: bool = Query(default=True, description="title Y2 with its alias"),
        y3alias: bool = Query(default=True, description="title Y3 with its alias"),
        yfit: str = Query(default="none", description="the fit on Y (figures.FITS)"),
        y2fit: str = Query(default="none", description="the fit on Y2"),
        y3fit: str = Query(default="none", description="the fit on Y3"),
        yorder: int = Query(default=2, ge=0, le=MAX_FIT_ORDER),
        y2order: int = Query(default=2, ge=0, le=MAX_FIT_ORDER),
        y3order: int = Query(default=2, ge=0, le=MAX_FIT_ORDER),
    ) -> JSONResponse:
        """The per-shot figure of ``y`` (and ``y2``, ``y3``) against ``x``."""
        frame = scans.load_frame(ctx.run(day, scan))
        if not y:
            raise HTTPException(status_code=404, detail="no plottable column ''")
        # Each list keeps its slot (colour and axis): Y3 stays Y3 with Y2
        # off. Trailing off slots are dropped so a one-Y snippet stays bare.
        axes = [(y, yalias), (y2 or None, y2alias), (y3 or None, y3alias)]
        while axes[-1][0] is None:
            axes.pop()
        ys = [c for c, _ in axes]
        drawn = [c for c in ys if c]
        try:
            series = scans.plot_series(frame, [*drawn, x] if x else drawn)
        except KeyError as missing:
            raise HTTPException(
                status_code=404, detail=f"no plottable column {missing.args[0]!r}"
            ) from None
        # A title only where the toggle changes it, so the snippet stays bare.
        x_label = _changed(x, xalias) if x else None
        y_labels = [_changed(c, alias) if c else None for c, alias in axes]
        # A fit only on a list that is drawn; a name this copy lacks (a
        # newer copy saved it) is reported, not fitted.
        asked = [m if c else "none" for c, m in zip(ys, (yfit, y2fit, y3fit))]
        known = [m if m in figures.FITS else "none" for m in asked]
        orders = [yorder, y2order, y3order][: len(ys)]
        shot = scans.shot_axis(frame)
        fits = figures.fit_columns(
            series, ys, known, x=x or None, shot=shot, orders=orders
        )
        fig = figures.shots_figure(
            series,
            ys,
            x=x or None,
            shot=shot,
            x_label=x_label,
            y_labels=y_labels,
            fits=fits,
            palette=figures.THEMED_PALETTE,
        )
        results = [
            fit_result(i + 1, c, m, fit)
            for i, (c, m, fit) in enumerate(zip(ys, asked, fits))
            if c and m != "none"
        ]
        body = FigureResponse(
            x=x,
            y=y,
            y2=y2,
            y3=y3,
            shots=len(frame),
            figure=figures.page_figure(fig),
            code=figure_code(
                day,
                scan,
                ys,
                x or None,
                ctx.experiment,
                x_label,
                y_labels,
                fits=known,
                orders=orders,
            ),
            fits=results,
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

        return _json(await run_in_threadpool(_settings, save))


def _changed(column: str, use_alias: bool) -> Optional[str]:
    """The axis title the alias toggle gives *column*, if not its name."""
    shown = scans.display_name(column, use_alias)
    return shown if shown != column else None


def _settings(action: Callable[[], PlotterSettings]) -> PlotterSettings:
    """Run a store call; a bad name or file is a 422 naming it."""
    try:
        return action()
    except SettingsError as bad:
        raise HTTPException(status_code=422, detail=str(bad)) from None


def _json(body: BaseModel) -> JSONResponse:
    # mode="json": a set's remembered day is a date, which JSONResponse
    # cannot encode; every model goes out as plain JSON values.
    return JSONResponse(body.model_dump(mode="json"), headers=NO_CACHE)
