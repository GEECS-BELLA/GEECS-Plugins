"""The JSON routes: a scan's columns, the figure for a pick, the settings sets.

Each is a few calls into :mod:`geecs_portal.plotter.scans`,
:mod:`geecs_portal.plotter.figures` or :mod:`geecs_portal.plotter.settings`; the figure
route also returns ``code``, the notebook snippet that rebuilds the same
frame and figure.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date
from typing import Any, Callable, Optional, Sequence, Union

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from geecs_data_utils.data.curve_fit import CurveFit
from geecs_data_utils.data.row_filters import RowFilters, apply_filters
from geecs_data_utils.scan_grid import grid_scan
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from geecs_portal import analysis
from geecs_portal.plotter import figures, maps, scans
from geecs_portal.plotter.routes import Context, read_form
from geecs_portal.plotter.settings import (
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

    model_config = ConfigDict(populate_by_name=True)

    x: str
    y: str
    #: Y2 and Y3, ``""`` when off.
    y2: str = ""
    y3: str = ""
    shots: int
    #: ``"shot"`` or ``"bin"``: what the figure draws.
    view: str = "shot"
    #: The shots the filters pass, of all of them.
    passing: int = Field(0, alias="pass")
    total: int = 0
    # The plotly figure JSON: the one free dict in a response (the brief).
    figure: dict[str, Any]
    code: str
    #: One per Y list with a fit picked, in slot order (per-shot view only:
    #: a fit is made to the shots, never to bin centres).
    fits: list[FitResult] = []


def _finite(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def plus_minus(value: float, stderr: float) -> str:
    """``value ± stderr``, both to the error's second significant figure.

    No usable error (NaN, zero) shows the value to four significant
    figures. A value smaller than its error's rounding step keeps three
    significant figures of its own, so an unconstrained term reads
    ``39 ± 3200``, never ``0 ± 3200`` while the curve is drawn at 39.
    Errors past 1e±5 are written in exponent form, the value with as many
    digits as the error leaves meaningful.
    """
    if not math.isfinite(value):
        return "undefined"
    if not math.isfinite(stderr) or stderr <= 0:
        return f"{value:.4g}"
    exponent = math.floor(math.log10(stderr))
    if abs(exponent) > 5:
        magnitude = math.floor(math.log10(abs(value))) if value else exponent
        # A double holds ~15 significant figures; past that is float noise.
        digits = min(14, max(1, magnitude - exponent + 1))
        return f"{value:.{digits}e} ± {stderr:.1e}"
    decimals = 1 - exponent
    if decimals >= 0:
        return f"{value:.{decimals}f} ± {stderr:.{decimals}f}"
    step = 10.0 ** (-decimals)
    err = f"{round(stderr / step) * step:.0f}"
    if abs(value) < 10 * step:
        return f"{value:.3g} ± {err}"
    return f"{round(value / step) * step:.0f} ± {err}"


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
        formula=fit.formula,
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
    filters: Optional[RowFilters] = None,
    cfg: Optional[Any] = None,
    display: Optional[dict] = None,
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
    filters : RowFilters, optional
        The shot filters; their active groups narrow the frame first.
    cfg : BinningConfig, optional
        Given, the snippet bins (the page's binned view) instead of
        fitting and plotting the shots.
    display : dict, optional
        The plot cosmetics; its ``layout`` patch is the browser's.

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
    display_arg = analysis.snippet_display(display)
    head = (
        "from datetime import date\n"
        "from geecs_portal.plotter import figures, scans\n\n"
        f"run = scans.find_scan(date({day.year}, {day.month}, {day.day}), {scan}{where})\n"
        "frame = scans.load_frame(run)\n"
    )
    filtering = filters is not None and bool(filters.active_groups())
    keep = (
        "from geecs_data_utils.data.row_filters import RowFilters, apply_filters\n"
        "kept = apply_filters(frame, RowFilters.model_validate_json("
        f"{filters.model_dump_json()!r}))\n"
        if filtering and filters is not None
        else ""
    )
    if cfg is not None:
        binned = head + keep + ("frame = kept\n" if filtering else "")
        return binned + _binned_code(cfg, ys, drawn, x, x_label, y_labels, display_arg)
    # As the endpoint: columns are read off the whole scan, then cut to the
    # kept shots, so a filter that keeps none still draws (an empty plot).
    series = (
        "series = {n: v.loc[kept.index] for n, v in "
        f"scans.plot_series(frame, {names!r}).items()}}\n"
        "frame = kept\n"
        if filtering
        else f"series = scans.plot_series(frame, {names!r})\n"
    )
    return (
        head + keep + series + f"{fitting}"
        f"fig = figures.shots_figure(series, {y_arg}{x_arg}, shot=scans.shot_axis(frame)"
        f"{labels}{fit_arg}{display_arg})\n"
        "fig.show()\n"
        + ("for fit in filter(None, fits):\n    print(fit)\n" if fitting else "")
    )


def _binned_code(
    cfg: Any,  # BinningConfig: a plain dataclass, its fields named below
    ys: Sequence[Optional[str]],
    drawn: Sequence[str],
    x: Optional[str],
    x_label: Optional[str],
    y_labels: Sequence[Optional[str]],
    display_arg: str,
) -> str:
    """The binned half of :func:`figure_code`: bin, place, draw."""
    non_default = {
        f.name: getattr(cfg, f.name)
        for f in dataclasses.fields(cfg)
        if f.name != "value_cols" and getattr(cfg, f.name) != f.default
    }
    kwargs = ", ".join(f"{k}={v!r}" for k, v in sorted(non_default.items()))
    lines = (
        "from dataclasses import replace\n"
        "from geecs_data_utils.data.binning import BinningConfig, bin_frame\n"
        "from geecs_portal.analysis import jsonable_labels, jsonable_values\n"
        f"cfg = BinningConfig({kwargs})\n"
        f"result = bin_frame(frame, replace(cfg, value_cols={tuple(drawn)!r}))\n"
        "series = {c: {s: jsonable_values(result.frame[(c, s)])\n"
        '              for s in ("center", "err_low", "err_high")}\n'
        f"          for c in {list(drawn)!r} if (c, 'center') in result.frame}}\n"
    )
    place = ""
    if x:
        lines += (
            "# bins group the shots; the X column's per-bin mean places them\n"
            f"x_result = bin_frame(frame, replace(cfg, value_cols=({x!r},), agg='mean'))\n"
            f"x_centers = jsonable_values(x_result.frame[({x!r}, 'center')]"
            ".reindex(result.frame.index))\n"
        )
        place = f", x_values=x_centers, x_label={(x_label or x)!r}"
    y_arg = repr(ys[0]) if len(ys) == 1 else repr(list(ys))
    titled = f", y_labels={list(y_labels)!r}" if any(y_labels) else ""
    return (
        lines
        + f"fig = figures.binned_figure(jsonable_labels(result.frame.index), series, {y_arg}"
        f", bin_col={cfg.bin_col!r}{place}{titled}{display_arg})\n"
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
        view: str = Query(
            default="shot", pattern="^(shot|bin)$", description="per shot or binned"
        ),
        filters: str = Query(default="", description="RowFilters JSON: the shots kept"),
        bincfg: str = Query(default="", description="BinningConfig JSON, for view=bin"),
        display: str = Query(default="", description="the plot cosmetics JSON"),
    ) -> JSONResponse:
        """The figure of ``y`` (and ``y2``, ``y3``) against ``x``, per shot or binned.

        The filters narrow the shots first, in both views. Per shot, each Y
        list's fit is made to the shots drawn; binned, each bin's centre and
        error bars are drawn (the portal's ``/binned``: bins group, the X
        column's per-bin mean places them) and no fit is made.
        """
        return _figure_response(
            ctx,
            day,
            scan,
            axes=[(y, yalias), (y2 or None, y2alias), (y3 or None, y3alias)],
            x=x,
            xalias=xalias,
            fits=(yfit, y2fit, y3fit),
            orders=(yorder, y2order, y3order),
            view=view,
            filters=filters,
            bincfg=bincfg,
            display=display,
        )

    @router.get("/api/filter-count")
    def api_filter_count(day: date, scan: int, filters: str = "") -> JSONResponse:
        """``{pass, total}``: the shots the filters keep, for the editor's live count."""
        frame = scans.load_frame(ctx.run(day, scan))
        try:
            flt = analysis.parse_filters(filters)
        except analysis.BadParam as bad:
            raise HTTPException(status_code=400, detail=str(bad)) from None
        kept = _kept(frame, flt)
        return JSONResponse({"pass": len(kept), "total": len(frame)}, headers=NO_CACHE)

    @router.get("/api/map")
    def api_map(
        day: date,
        scan: int,
        value: str,
        x: str,
        y: str,
        mapcfg: str = Query(default="", description="GridConfig JSON: the statistics"),
        filters: str = Query(default="", description="RowFilters JSON: the shots kept"),
        valuealias: bool = Query(
            default=True, description="title the scalar with its alias"
        ),
        axisalias: bool = Query(
            default=True, description="title the axes with their aliases"
        ),
    ) -> JSONResponse:
        """The 2-D map of ``value`` over ``x`` and ``y``: both figures, the cells, the code.

        The portal scan page's ``/api/run/{uid}/grid`` answer, over this
        page's s-file frame; a column or filter the scan lacks is a 400
        that says which.
        """
        names = {value: valuealias, x: axisalias, y: axisalias}
        return _map_response(ctx, day, scan, (value, x, y), mapcfg, filters, names)

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
        ``copy_from`` names the set the page shows when it is not *name*
        (someone else's): its controls are the base the fields merge into,
        so the viewer's own set takes the whole view, not one field of it.
        """
        form = await read_form(request)
        source = form.pop("copy_from", "") or name

        def save() -> PlotterSettings:
            settings = merged(ctx.store.load(source), form)
            ctx.store.save(set_name(name), settings)
            return settings

        return _json(await run_in_threadpool(_settings, save))


def _figure_response(
    ctx: Context,
    day: date,
    scan: int,
    *,
    axes: Sequence[tuple[Optional[str], bool]],
    x: str,
    xalias: bool,
    fits: Sequence[str],
    orders: Sequence[int],
    view: str,
    filters: str,
    bincfg: str,
    display: str,
) -> JSONResponse:
    """``/api/figure``'s answer: *axes* is ``(column or None, alias)`` per Y slot."""
    y = axes[0][0] or ""
    frame = scans.load_frame(ctx.run(day, scan))
    if not y:
        raise HTTPException(status_code=404, detail="no plottable column ''")
    try:
        flt = analysis.parse_filters(filters)
        disp = analysis.parse_display(display)
        cfg = analysis.parse_bincfg(bincfg) if view == "bin" else None
    except analysis.BadParam as bad:
        raise HTTPException(status_code=400, detail=str(bad)) from None
    kept = _kept(frame, flt)
    # Each list keeps its slot (colour and axis): Y3 stays Y3 with Y2
    # off. Trailing off slots are dropped so a one-Y snippet stays bare.
    axes = list(axes)
    while axes[-1][0] is None:
        axes.pop()
    ys = [c for c, _ in axes]
    drawn = [c for c in ys if c]
    # Columns are judged on the whole scan, then cut to the kept shots: a
    # filter that keeps no shot (or none with a value of one Y) draws an
    # empty plot and a "0 / n shots pass" count, never a missing column.
    try:
        whole = scans.plot_series(frame, [*drawn, x] if x else drawn)
    except KeyError as missing:
        raise HTTPException(
            status_code=404, detail=f"no plottable column {missing.args[0]!r}"
        ) from None
    series = {name: values.loc[kept.index] for name, values in whole.items()}
    # A title only where the toggle changes it, so the snippet stays bare.
    x_label = _changed(x, xalias) if x else None
    y_labels = [_changed(c, alias) if c else None for c, alias in axes]
    asked = [m if c else "none" for c, m in zip(ys, fits)]
    known = [m if m in figures.FITS else "none" for m in asked]
    orders = list(orders)[: len(ys)]
    results: list[FitResult] = []
    if cfg is None:
        # A fit only on a list that is drawn; a name this copy lacks (a
        # newer copy saved it) is reported, not fitted.
        shot = scans.shot_axis(kept)
        made = figures.fit_columns(
            series, ys, known, x=x or None, shot=shot, orders=orders
        )
        fig = figures.shots_figure(
            series,
            ys,
            x=x or None,
            shot=shot,
            x_label=x_label,
            y_labels=y_labels,
            fits=made,
            display=disp,
            palette=figures.THEMED_PALETTE,
        )
        results = [
            fit_result(i + 1, c, m, fit)
            for i, (c, m, fit) in enumerate(zip(ys, asked, made))
            if c and m != "none"
        ]
    else:
        fig = _binned(kept, cfg, ys, drawn, x, x_label, y_labels, disp)
    body = FigureResponse(
        x=x,
        y=y,
        y2=(ys[1] if len(ys) > 1 else None) or "",
        y3=(ys[2] if len(ys) > 2 else None) or "",
        shots=len(frame),
        view=view,
        passing=len(kept),
        total=len(frame),
        figure=figures.page_figure(fig),
        code=figure_code(
            day,
            scan,
            ys,
            x or None,
            ctx.experiment,
            x_label,
            y_labels,
            fits=known if cfg is None else (),
            orders=orders,
            filters=flt,
            cfg=cfg,
            display=disp,
        ),
        fits=results,
    )
    return _json(body)


def _map_response(
    ctx: Context,
    day: date,
    scan: int,
    columns: tuple[str, str, str],
    mapcfg: str,
    filters: str,
    aliases: dict[str, bool],
) -> JSONResponse:
    """The map's answer: ``grid_scan`` over the s-file, as ``grid_document``."""
    run = ctx.run(day, scan)
    frame = scans.load_frame(run)
    try:
        cfg = maps.grid_config(mapcfg, *columns)
        flt = analysis.parse_filters(filters)
        result = grid_scan(frame, run.start_doc, cfg, flt)
    except KeyError as missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"this scan has no column {missing}: pick another, or turn "
                "off the shot filter that names it"
            ),
        ) from None
    except (ValueError, TypeError, IndexError, OverflowError) as bad:
        raise HTTPException(status_code=400, detail=str(bad)) from None
    pretty = {c: scans.display_name(c, aliases[c]) for c in columns if c}
    code = maps.map_code(day, scan, result.config, flt, pretty, ctx.experiment)
    return JSONResponse(analysis.grid_document(result, pretty, code), headers=NO_CACHE)


def _kept(frame: pd.DataFrame, filters: RowFilters) -> pd.DataFrame:
    """The shots *filters* keep; a 400 that says so when one does not fit this scan.

    The filters live in the viewer's set and follow them from scan to scan,
    so a filter on a device that did not save this scan is the usual case.
    """
    try:
        return apply_filters(frame, filters)
    except (KeyError, ValueError) as bad:
        raise HTTPException(
            status_code=400,
            detail=(
                f"a saved shot filter does not fit this scan ({bad}): turn its "
                "group off or remove it under Shot filters"
            ),
        ) from None


def _binned(
    frame: pd.DataFrame,
    cfg: Any,  # BinningConfig
    ys: Sequence[Optional[str]],
    drawn: Sequence[str],
    x: str,
    x_label: Optional[str],
    y_labels: Sequence[Optional[str]],
    display: dict,
) -> Any:  # a plotly Figure
    """The binned figure of the kept shots: the portal's ``/binned`` on this page's axes.

    Raises
    ------
    HTTPException
        404 for a missing bin column, 400 for a column ``bin_frame`` cannot
        aggregate or a degenerate percentile.
    """
    try:
        result, series, x_centers = analysis.binned_series(frame, cfg, drawn, x or None)
    except KeyError as missing:
        raise HTTPException(
            status_code=404,
            detail=f"the saved bin column {missing} is not in this scan: change it under ⚙ bins",
        ) from None
    except (TypeError, ValueError) as bad:
        raise HTTPException(
            status_code=400, detail=f"binned view needs numeric columns: {bad}"
        ) from None
    return figures.binned_figure(
        analysis.jsonable_labels(result.frame.index),
        series,
        ys,
        bin_col=cfg.bin_col,
        x_values=x_centers,
        x_label=(x_label or x) if x else None,
        y_labels=y_labels,
        display=display,
        palette=figures.THEMED_PALETTE,
    )


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
    return JSONResponse(body.model_dump(mode="json", by_alias=True), headers=NO_CACHE)
