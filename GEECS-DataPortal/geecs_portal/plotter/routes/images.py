"""The image strip's JSON and PNG routes.

``/api/images/strip`` lists a device's tiles (one per bin or per shot) with
each tile's PNG URL and the notebook snippet; ``/api/images/tile.png`` is
one tile: a shot's image, or its bin's average, drawn with the colour map
and z scale. Each is a few calls into :mod:`geecs_portal.plotter.images`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional, Sequence
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from geecs_analysis.image_prep import HIT_FILTERS, ImagePrep
from geecs_data_utils.data.scan_merge import SCAN_BIN_COLUMN
from geecs_data_utils.io.display_png import colour_bar_png, safe_cmap, to_display_png
from pydantic import BaseModel

from geecs_portal import analysis
from geecs_portal.plotter import images, multi, scans
from geecs_portal.plotter.routes import Context
from geecs_portal.plotter.routes.api import NO_CACHE
from geecs_portal.plotter.settings import (
    SettingsError,
    channels_value,
    roi_value,
    scan_refs,
)

Strip = Literal["bins", "shots", "pick"]


class DevicesResponse(BaseModel):
    """A scan's device folders."""

    devices: list[str]


class TileModel(BaseModel):
    """One tile of the strip."""

    index: int
    #: The 1-based s-file rows it shows (one for a shot, a bin's for a bin).
    rows: list[int]
    caption: str
    #: The bin it is, or its shot was taken in (``""``: no ``Bin #``).
    bin: str = ""
    #: The tile's PNG, relative to the app (``api/images/tile.png?…``).
    url: str
    #: Several scans, a shot tile: its own scan's run uid, day and row
    #: there (the big view's processing preview opens that scan).
    uid: str = ""
    day: str = ""
    row: int = 0


class LevelsResponse(BaseModel):
    """The strip's shared z scale, and how many tiles gave it."""

    #: The pixel values drawn black and white.
    zmin: float
    zmax: float
    tiles: int


class TraceResponse(BaseModel):
    """A trace tile as the big view draws it."""

    #: The plotly figure (theme sentinels for the page to resolve).
    figure: dict
    points: int
    #: The shots the tile's rows name (a bin's mean is over those that had a trace).
    shots: int
    #: How a bin's mean was made, when the reader should know ("": nothing to say).
    note: str = ""
    #: Each picked channel the tile has no trace of, and why ("Channel 3: …").
    missing: list[str] = []


class StripResponse(BaseModel):
    """A device's strip: its tiles and the snippet that loads their pixels."""

    device: str
    strip: Strip
    tiles: list[TileModel]
    code: str


def parse_roi(text: str) -> Optional[images.Roi]:
    """The ``roi`` parameter as pixels; a bad one is a 422 naming it."""
    try:
        roi = roi_value(text)
    except SettingsError as bad:
        raise HTTPException(status_code=422, detail=str(bad)) from None
    return None if roi is None else (roi[0], roi[1], roi[2], roi[3])


def prep_query(prep: ImagePrep) -> dict[str, object]:
    """The clean-up half of a tile's or the levels' URL (nothing when off)."""
    query: dict[str, object] = {}
    if prep.hits != "none":
        query["hits"] = prep.hits
    if prep.centre is not None:
        query["centre"] = prep.centre
    return query


def tile_url(
    day: date,
    scan: int,
    device: str,
    strip: str,
    index: int,
    prep: ImagePrep = ImagePrep(),
    filters: str = "",
    bg: str = "",
    channels: Optional[Sequence[int]] = None,
    extra: str = "",
    mtol: Optional[float] = None,
    binx: Optional[str] = None,
    window: Optional[images.XWindow] = None,
) -> str:
    """A tile's PNG URL, relative to the app root; the page adds the display.

    A trace row's x *window* rides along (its tile and the big view's figure).

    A trace device's tile names its scope *channels* (``channel=`` once
    each, overlaid; ``None``: a camera).
    Several scans name the others (*extra*, ``YYYY-MM-DD:N,…``) and the
    pooling tolerance (*mtol*, empty: the default), so the route rebuilds
    the same merged rows and bins.

    A bin's one shot (``pick``) is a shot tile: its index is its row. A
    bin carries the shot filters (*filters*, the set's JSON): its index is
    its place among the bins they leave, and it averages only their shots.
    """
    query = {"day": day.isoformat(), "scan": scan, "device": device}
    query.update(strip="shots" if strip == "pick" else strip, index=index)
    if strip == "bins" and filters:
        query["filters"] = filters
    if bg:
        query["bg"] = bg  # the background scan (YYYY-MM-DD:N) taken off each shot
    if channels is not None:
        query["channel"] = list(channels)
    query.update(multi_query(extra, mtol, binx))
    query.update(window_query(window))
    return f"api/images/tile.png?{urlencode({**query, **prep_query(prep)}, doseq=True)}"


def multi_query(
    extra: str, mtol: Optional[float], x: Optional[str] = None
) -> dict[str, object]:
    """The several-scans half of a tile's or the levels' URL (nothing for one scan).

    *x* is the column the pooled bins are placed by (the page's X; ``""``
    the shot number), so every route pools the bins the page shows.
    """
    if not extra:
        return {}
    query: dict[str, object] = {"scans": extra}
    if mtol is not None:
        query["mtol"] = mtol
    if x is not None:
        query["bx"] = x
    return query


def scan_set(
    ctx: Context,
    day: date,
    number: int,
    extra: str,
    mtol: Optional[float],
    binx: Optional[str] = None,
) -> Optional[images.ScanSet]:
    """The open scan with *extra* (``YYYY-MM-DD:N,…``) as one :class:`images.ScanSet`.

    ``multi.scan_set``: the bins ordered by *binx* (the page's X; ``None``
    the open scan's variable, ``""`` scan and bin order) and pooled within
    *mtol* (``None``: half the smallest step). ``None`` with no other scan; a 422 for text that is not scans
    or a negative *mtol*, a 404 naming a scan that is not on the share.
    """
    if not extra:
        return None
    try:
        refs = [(r.day, r.scan) for r in scan_refs(extra)]
    except SettingsError as bad:
        raise HTTPException(status_code=422, detail=f"scans={extra!r}: {bad}") from None
    refs = [(day, number), *(r for r in refs if r != (day, number))]
    if len(refs) == 1:
        return None
    try:
        return multi.scan_set(
            refs, mtol, x=binx, experiment=ctx.experiment, catalog=ctx.catalog
        )
    except LookupError as missing:
        raise HTTPException(status_code=404, detail=str(missing)) from None
    except ValueError as bad:
        raise HTTPException(status_code=422, detail=f"mtol={mtol!r}: {bad}") from None


def parse_rows(text: str) -> list[images.Tile]:
    """The ``rows`` parameter as one shot tile each; a bad one is a 422."""
    try:
        return [images.Tile(r, (r,), "") for r in (int(v) for v in text.split(","))]
    except ValueError:
        raise HTTPException(
            status_code=422, detail=f"rows={text!r}: whole numbers"
        ) from None


def shown_tiles(found: list[images.Tile], shown: str) -> list[images.Tile]:
    """The tiles whose ``index`` the ``shown`` parameter lists (``""``: all).

    The page's one-row layout shows a few tiles spread over the scan; the
    shared scale is taken over those only (the author, 2026-11-03).

    Raises
    ------
    HTTPException
        422 when ``shown`` is not whole numbers.
    """
    if not shown:
        return found
    try:
        wanted = {int(v) for v in shown.split(",")}
    except ValueError:
        raise HTTPException(
            status_code=422, detail=f"shown={shown!r}: whole numbers"
        ) from None
    return [t for t in found if t.index in wanted]


def parse_prep(hits: str, centre: Optional[float]) -> ImagePrep:
    """The ``hits`` and ``centre`` parameters; a bad one is a 422 naming it."""
    if hits not in HIT_FILTERS:
        raise HTTPException(
            status_code=422, detail=f"hits={hits!r}: one of {', '.join(HIT_FILTERS)}"
        )
    if centre is not None and not (math.isfinite(centre) and centre > 0):
        raise HTTPException(status_code=422, detail=f"centre={centre!r}: above 0")
    return ImagePrep(hits=hits, centre=centre)


def strip_axis(run, frame, alias: bool = True) -> tuple[str, str]:
    """The strip's X: the scan variable's column and its name as shown."""
    x = analysis.default_x(run.start_doc, scans.columns(frame))
    return x, scans.display_name(x, alias) if x else ""


def strip_tiles(
    run,
    frame,
    strip: str,
    *,
    day: date,
    scan: int,
    device: str,
    x: Optional[str] = None,
    pick_by: str = "",
    prep: ImagePrep = ImagePrep(),
    filters: str = "",
    bg: str = "",
    channels: Optional[Sequence[int]] = None,
    source: Optional[images.Source] = None,
    extra: str = "",
    mtol: Optional[float] = None,
    window: Optional[images.XWindow] = None,
) -> list[TileModel]:
    """The strip's tiles with their URLs; captions carry the scan variable's value.

    A trace row's x *window* rides on every tile's URL.

    With several scans (*source* an :class:`images.ScanSet`, *frame* its
    merged frame) the tiles are its rows and its one bin number, each
    caption names its scan, and the URLs carry *extra* and *mtol*.

    Only the shots the shot filters keep (*filters*, the set's JSON; a 422
    when they cannot apply here).

    *x* is the scan variable's column when the caller has it already
    (:func:`strip_axis`); ``None`` looks it up. *pick_by* is the column
    whose highest value picks each bin's one shot (``pick``; ``""``: its
    first shot).
    """
    if x is None:
        x = strip_axis(run, frame)[0]
    values = scans.plot_series(frame, [x])[x] if x else None
    by = (
        scans.plot_series(frame, [pick_by])[pick_by]
        if strip == "pick" and pick_by in frame.columns
        else None
    )
    keep = keep_rows(frame, filters)
    many = source if isinstance(source, images.ScanSet) else None
    try:
        found = images.tiles(
            frame,
            strip,
            values,
            scans.shot_axis(frame),
            pick_by=by,
            keep=keep,
            bin_col=many.bin_col if many else SCAN_BIN_COLUMN,
            scan_names=many.names() if many else None,
        )
    except images.NoImage as why:
        raise HTTPException(status_code=404, detail=str(why)) from None
    return [
        TileModel(
            index=t.index,
            rows=list(t.rows),
            caption=t.caption,
            bin=t.bin,
            url=tile_url(
                day,
                scan,
                device,
                strip,
                t.index,
                prep,
                filters,
                bg,
                channels,
                extra if many else "",
                mtol,
                x if many else None,
                window,
            ),
            **(_own_scan(many, t) if many and strip != "bins" else {}),
        )
        for t in found
    ]


def _own_scan(many: images.ScanSet, tile: images.Tile) -> dict[str, object]:
    """A shot tile's own scan: its uid, day and row there (the big view's preview)."""
    part, row = many.locate(tile.rows[0])
    return {"uid": part.uid, "day": part.day.isoformat(), "row": row}


def strip_code(
    day: date,
    scan: int,
    device: str,
    strip: str,
    experiment: str,
    pick_by: str = "",
    prep: ImagePrep = ImagePrep(),
    refs: Sequence[tuple[date, int]] = (),
    mtol: Optional[float] = None,
    binx: Optional[str] = None,
) -> str:
    """The notebook snippet that loads each tile's pixels as the page does.

    With *refs* (the open scan and the others) the strip is their
    ``multi.scan_set``, pooled within *mtol*.
    """
    by = f", pick_by=frame[{pick_by!r}]" if strip == "pick" and pick_by else ""
    return (
        "from datetime import date\n"
        "from geecs_analysis.image_prep import ImagePrep\n"
        + _code_head(day, scan, experiment, refs, mtol, binx)
        + f"prep = ImagePrep(hits={prep.hits!r}, centre={prep.centre!r})\n"
        f"for tile in images.tiles(frame, {strip!r}{by}{_code_bins(refs)}):\n"
        "    try:\n"
        f"        pixels = images.tile_array(folder, frame, {device!r}, tile.rows, prep)\n"
        "    except images.NoImage as why:\n"
        "        print(tile.caption, '—', why)\n"
        "        continue\n"
        "    print(tile.caption, pixels.shape)\n"
    )


def scalars_code(
    day: date, scan: int, columns: list[str], stat: str, filters: str, experiment: str
) -> str:
    """The notebook snippet that rebuilds the scan scalars' numbers."""
    agg, err, _label = scans.SCALAR_STATS[stat]
    where = f", experiment={experiment!r}" if experiment else ""
    keep = (
        f"keep = images.shot_rows(frame, {filters!r})\n"
        "rows = frame if keep is None else frame.iloc[sorted(r - 1 for r in keep)]\n"
        if filters
        else "rows = frame\n"
    )
    return (
        "from datetime import date\n"
        "from geecs_data_utils.data.binning import summarize_columns\n"
        "from geecs_portal.plotter import images, scans\n\n"
        f"run = scans.find_scan(date({day.year}, {day.month}, {day.day}), {scan}{where})\n"
        "frame = scans.load_frame(run)\n"
        f"{keep}"
        f"for column, s in summarize_columns(rows, {columns!r}, agg={agg!r}, err={err!r}).items():\n"
        "    print(column, s.center, s.err_low, s.err_high, s.count)\n"
    )


def keep_rows(frame, filters: str) -> Optional[frozenset[int]]:
    """The rows the shot filters keep (``images.shot_rows``); a 422 when they cannot apply."""
    try:
        return images.shot_rows(frame, filters)
    except images.NoImage as why:
        raise HTTPException(status_code=422, detail=str(why)) from None


def tile_rows(
    frame,
    strip: str,
    index: int,
    keep: Optional[frozenset[int]] = None,
    bin_col: str = SCAN_BIN_COLUMN,
) -> tuple[int, ...]:
    """The rows a tile shows: the shot's row, or the bin's (by position).

    A bin's position is among the bins the shot filters (*keep*) leave.

    Raises
    ------
    HTTPException
        404 when the scan has no bins, or no bin at *index*.
    """
    if strip in ("shots", "pick"):
        return (index,)
    try:
        found = images.kept_bins(frame, keep, bin_col)
    except images.NoImage as why:
        raise HTTPException(status_code=404, detail=str(why)) from None
    if not 0 <= index < len(found):
        raise HTTPException(status_code=404, detail=f"no bin {index} of {len(found)}")
    return found[index][1]


def tile_png(
    folder: Path,
    frame,
    device: str,
    rows: tuple[int, ...],
    roi: Optional[images.Roi],
    cmap: str,
    levels: Optional[tuple[float, float]],
    bar: bool = False,
    prep: ImagePrep = ImagePrep(),
    cbar: bool = False,
    background: Optional[images.Background] = None,
) -> bytes:
    """The tile's pixels, in its ROI, drawn as a PNG; a 404 naming why not.

    With *bar*, a figure with a colour bar beside the picture (the big
    view): the same colour map and levels, so its values read off. With
    *cbar*, the tile itself with a slim labelled bar at its right (a strip
    whose tiles each scale themselves).
    """
    try:
        pixels = images.crop(
            images.kept_tile_array(folder, frame, device, rows, prep, background), roi
        )
        if bar:
            return _with_colour_bar(pixels, cmap, levels)
        # Without levels, the tile's own darkest to brightest pixel.
        return to_display_png(
            pixels, cmap=cmap, plo=0.0, phi=100.0, levels=levels, colour_bar=cbar
        )
    except images.NoImage as why:
        raise HTTPException(status_code=404, detail=str(why)) from None
    except (TypeError, ValueError) as exc:  # an array no PNG can hold
        raise HTTPException(status_code=404, detail=f"cannot draw: {exc}") from None


def _with_colour_bar(pixels, cmap: str, levels: Optional[tuple[float, float]]) -> bytes:
    """The picture and a colour bar, drawn by matplotlib's object API.

    Black and white are *levels* when given, else the picture's own
    darkest and brightest pixel, as the tile draws it. An unknown colour
    map draws greyscale, as the tile does.
    """
    import numpy as np
    from matplotlib.figure import Figure

    from geecs_portal.resources import figure_png

    data = np.asarray(pixels, dtype=float)
    finite = data[np.isfinite(data)]
    lo, hi = (
        levels
        if levels
        else ((float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0))
    )
    height, width = data.shape[:2]
    fig = Figure(figsize=(7, 7 * height / max(width, 1) + 0.4), dpi=110)
    ax = fig.add_subplot()
    shown = ax.imshow(
        data,
        cmap=safe_cmap(cmap) or "gray",
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )
    fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.03)
    return figure_png(fig, tight=True)


def grid_bar_png(
    cmap: str, zmin: float, zmax: float, length: int, vertical: bool = False
) -> bytes:
    """The grid's one colour bar, *zmin* to *zmax*; a 422 unless zmax is above zmin."""
    if not (math.isfinite(zmin) and math.isfinite(zmax)) or zmax <= zmin:
        raise HTTPException(status_code=422, detail="zmax must be above zmin")
    return colour_bar_png(cmap, (zmin, zmax), length=length, vertical=vertical)


def open_background(ctx: Context, bg: str) -> Optional[images.Background]:
    """The ``bg`` parameter's scan (``YYYY-MM-DD:N``), opened; ``None`` for none.

    Raises
    ------
    HTTPException
        422 when *bg* is not one day and scan; 404 when there is no such scan.
    """
    if not bg:
        return None
    try:
        (ref,) = scan_refs(bg)
    except (SettingsError, ValueError) as bad:
        raise HTTPException(status_code=422, detail=f"bg={bg!r}: {bad}") from None
    run = ctx.run(ref.day, ref.scan)  # a 404 naming the scan when it is not there
    return images.Background(
        scans.scan_folder(run), scans.load_frame(run), f"Scan{ref.scan:03d} ({ref.day})"
    )


def trace_code(
    day: date,
    scan: int,
    device: str,
    strip: str,
    experiment: str,
    channels: Sequence[int],
    pick_by: str = "",
    refs: Sequence[tuple[date, int]] = (),
    mtol: Optional[float] = None,
    binx: Optional[str] = None,
    window: Optional[images.XWindow] = None,
) -> str:
    """The notebook snippet that loads each trace tile as the page does.

    *refs* as :func:`strip_code`; the row's x *window* crops each trace;
    each of the row's *channels* the tile has is one trace.
    """
    by = f", pick_by=frame[{pick_by!r}]" if strip == "pick" and pick_by else ""
    crop = (
        f"            trace = images.windowed(trace, {tuple(window)!r})\n"
        if window
        else ""
    )
    return (
        "from datetime import date\n"
        + _code_head(day, scan, experiment, refs, mtol, binx)
        + f"for tile in images.tiles(frame, {strip!r}{by}{_code_bins(refs)}):\n"
        "    try:\n"
        "        found = images.channel_traces(\n"
        f"            folder, frame, {device!r}, tile.rows, {list(channels)!r}\n"
        "        )\n"
        "    except images.NoImage as why:\n"
        "        print(tile.caption, '—', why)\n"
        "        continue\n"
        "    for channel, one in found:  # a bin: each channel's shots' mean\n"
        "        trace = one.result\n"
        + (
            "        try:\n" + crop + "        except images.NoImage as why:\n"
            "            print(tile.caption, channel, '—', why)\n"
            "            continue\n"
            if window
            else ""
        )
        + "        x, y = trace.data[:, 0], trace.data[:, 1]\n"
        "        print(tile.caption, channel, len(x), y.min(), y.max())\n"
    )


def _code_head(
    day: date,
    scan: int,
    experiment: str,
    refs: Sequence[tuple[date, int]],
    mtol: Optional[float],
    binx: Optional[str] = None,
) -> str:
    """A strip snippet's opening: one scan's frame and folder, or the scans' set."""
    where = f", experiment={experiment!r}" if experiment else ""
    if len(refs) > 1:
        listed = ", ".join(
            f"(date({d.year}, {d.month}, {d.day}), {n})" for d, n in refs
        )
        return (
            "from geecs_portal.plotter import images, multi\n\n"
            f"refs = [{listed}]\n"
            f"folder = multi.scan_set(refs, {mtol!r}, x={binx!r}{where})\n"
            "frame = folder.frame\n"
        )
    return (
        "from geecs_portal.plotter import images, scans\n\n"
        f"run = scans.find_scan(date({day.year}, {day.month}, {day.day}), {scan}{where})\n"
        "frame, folder = scans.load_frame(run), scans.scan_folder(run)\n"
    )


def _code_bins(refs: Sequence[tuple[date, int]]) -> str:
    """The strip snippet's ``tiles`` arguments for several scans (their bins, their names)."""
    if len(refs) > 1:
        return ", bin_col=folder.bin_col, scan_names=folder.names()"
    return ""


#: Units a tile's ticks take an SI prefix on (``2 µs``, ``500 mV``); any
#: other (``nm``, ``a.u.``) reads as plain numbers, never ``1 knm``.
_SI_TICK_UNITS = frozenset({"s", "V", "A", "Hz", "W", "J"})


def tick_formatter(units: Optional[str]):
    """A tile axis' tick formatter: engineering (SI-prefixed) or ``None``.

    ``None`` leaves matplotlib's plain numbers, for a named unit that takes
    no SI prefix (``nm`` would read ``1 knm``). No unit at all keeps the
    prefix: the reader cannot say whether it is seconds or nanometres.
    """
    from matplotlib.ticker import EngFormatter

    if units and units not in _SI_TICK_UNITS:
        return None
    return EngFormatter(unit=units or "", sep=" ")


def _axis_title(label: Optional[str], units: Optional[str]) -> str:
    """``"Time (s)"``: the reader's label and units, whichever it gave."""
    if label and units:
        return f"{label} ({units})"
    return label or (f"({units})" if units else "")


#: One line of a trace tile: its channel (which colour), its label, its trace.
TraceLine = tuple[int, str, Any]


def channel_label(names: Sequence[str], channel: int) -> str:
    """A channel's legend label: the scope's own name for it, else ``ch N``."""
    if 0 <= channel < len(names) and names[channel]:
        return names[channel]
    return f"ch {channel}"


def trace_png(
    lines: Sequence[TraceLine],
    levels: Optional[tuple[float, float]] = None,
    window: Optional[images.XWindow] = None,
) -> bytes:
    """A trace tile: its lines, small, with their axes; y from *levels* when given.

    Each line is one scope channel, coloured by its channel number
    (matplotlib's ``C0``, ``C1`` …, so a channel keeps its colour on every
    tile, whichever others it lacks); more than one gets a small legend.
    A line with no sample in the x *window* is left out, and a tile with
    none is :class:`images.NoImage`.

    matplotlib's object API (never pyplot). Without *levels* the y range is
    the trace's own; with them every tile of a row reads on one scale. With
    an x *window* only its samples are drawn, the axis spanning the window
    (a missing bound: the trace's own end). The ticks are few and in
    engineering units (``2 µs``, ``500 mV``), so they read at a tile's size;
    a named unit that takes no SI prefix (``nm``) ticks as plain numbers. An
    axis with no unit at all (a headerless text spectrum) still gets a
    prefix (``1 k``).
    """
    inside = images.windowed_each([trace for _, _, trace in lines], window)
    shown = [
        (channel, label, trace)
        for (channel, label, _), trace in zip(lines, inside)
        if trace is not None
    ]
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    from geecs_portal.resources import figure_png

    fig = Figure(figsize=(3.2, 2.0), dpi=100)
    ax = fig.add_subplot()
    for channel, label, trace in shown:
        ax.plot(
            trace.data[:, 0],
            trace.data[:, 1],
            linewidth=1.0,
            color=f"C{channel % 10}",
            label=label,
        )
    ax.margins(x=0)
    trace = shown[0][2]  # the x axis: the first line's (one scope, one time base)
    # The y ticks carry a unit only when every line has the same one.
    y_units = {t.y_units for _, _, t in shown}
    one_y_unit = y_units.pop() if len(y_units) == 1 else None
    if window is not None:
        x = np.concatenate([t.data[:, 0] for _, _, t in shown])
        low, high = window
        ax.set_xlim(x.min() if low is None else low, x.max() if high is None else high)
    if len(shown) > 1:
        ax.legend(
            fontsize=6,
            loc="upper right",
            frameon=False,
            handlelength=1.0,
            borderaxespad=0.2,
            labelspacing=0.15,
        )
    if levels is not None:
        # A little room either side, as matplotlib's own range leaves,
        # so the highest and lowest samples are not drawn on the frame.
        pad = 0.04 * (levels[1] - levels[0])
        ax.set_ylim(levels[0] - pad, levels[1] + pad)
    for axis, units in ((ax.xaxis, trace.x_units), (ax.yaxis, one_y_unit)):
        axis.set_major_locator(MaxNLocator(4))
        formatter = tick_formatter(units)
        if formatter is not None:
            axis.set_major_formatter(formatter)
    ax.tick_params(labelsize=9, length=2, pad=1)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.subplots_adjust(left=0.2, right=0.96, top=0.95, bottom=0.14)
    return figure_png(fig)


def tile_lines(
    folder: images.Source,
    frame,
    device: str,
    rows: tuple[int, ...],
    channels: Sequence[int],
    names: Sequence[str],
) -> tuple[list[TraceLine], list[str], list[str]]:
    """The tile's lines, one per picked channel it has; its notes; what it lacks.

    The notes are a bin's (how its mean was made); the last list says, for
    each picked channel the tile lacks, why (``Channel 3: the file holds no
    trace``), so a channel left out is said, never silently dropped.

    Raises
    ------
    images.NoImage
        No picked channel has a trace on this tile.
    """
    finds = images.channel_finds(folder, frame, device, rows, channels)
    lines: list[TraceLine] = []
    notes: list[str] = []
    missing: list[str] = []
    for ch, one in finds:
        if isinstance(one, str):
            missing.append(f"{channel_label(names, ch)}: {one}")
            continue
        lines.append((ch, channel_label(names, ch), one.result))
        if one.note:
            notes.append(one.note)
    if not lines:
        raise images.no_channel(finds)
    return lines, list(dict.fromkeys(notes)), missing


def y_title(traces: Sequence) -> str:
    """The y axis' title for overlaid traces: theirs when they agree, else none.

    Two lines of different quantities (an output table's columns) share
    the axis' range, never one line's label; the legend names each.
    """
    titles = {_axis_title(t.y_label, t.y_units) for t in traces}
    return titles.pop() if len(titles) == 1 else ""


def trace_tile_png(
    folder: images.Source,
    frame,
    device: str,
    rows: tuple[int, ...],
    channels: Sequence[int],
    levels: Optional[tuple[float, float]],
    window: Optional[images.XWindow] = None,
    names: Sequence[str] = (),
) -> bytes:
    """A trace tile's PNG (inside the row's x *window*); a 404 naming why there is none.

    Every picked channel the tile has, overlaid (*names*: the scope's
    channel names, for the legend). Levels that do not rise (white not
    above black) draw the traces' own range, rather than a flat line.
    """
    if levels is not None and not levels[1] > levels[0]:
        levels = None
    try:
        lines, _, _ = tile_lines(folder, frame, device, rows, channels, names)
        return trace_png(lines, levels, window)
    except images.NoImage as why:
        raise HTTPException(status_code=404, detail=str(why)) from None


def trace_view(
    lines: Sequence[TraceLine],
    name: str,
    window: Optional[images.XWindow] = None,
    channels_named: bool = False,
) -> dict:
    """The big view's figure for a trace tile (plotly, the page's theme).

    Each line (a channel) in its theme colour, ``--trace-N`` by channel
    number, with a legend when there is more than one; a lone line is
    named *name* (the device), or its channel when *channels_named* (a
    device of several). The whole traces,
    so it zooms out past the row's x *window*; with one, the figure opens
    on it, the y range the window's samples of every line (padded).

    Raises
    ------
    images.NoImage
        No line has a sample in the window.
    """
    from geecs_portal import figures as portal_figures

    palette = portal_figures.THEMED_PALETTE
    first = lines[0][2]
    figure = portal_figures.trace_figure(
        first.data[:, 0],
        first.data[:, 1],
        name=lines[0][1] if channels_named or len(lines) > 1 else name,
        x_title=_axis_title(first.x_label, first.x_units),
        y_title=y_title([trace for _, _, trace in lines]),
        palette=palette,
    )
    figure.data[0].line.color = portal_figures.trace_color(None, lines[0][0], palette)
    for channel, label, trace in lines[1:]:
        figure.add_scatter(
            x=list(trace.data[:, 0]),
            y=list(trace.data[:, 1]),
            mode="lines",
            name=label,
            line={
                "color": portal_figures.trace_color(None, channel, palette),
                "width": 1.5,
            },
        )
    if len(lines) > 1:
        figure.update_layout(showlegend=True)
    if window is not None:
        inside = [
            t.data
            for t in images.windowed_each([trace for _, _, trace in lines], window)
            if t is not None
        ]
        x = np.concatenate([d[:, 0] for d in inside])
        y = np.concatenate([d[:, 1] for d in inside])
        low, high = window
        figure.update_xaxes(
            range=[x.min() if low is None else low, x.max() if high is None else high]
        )
        finite = y[np.isfinite(y)]
        if finite.size:
            pad = 0.04 * ((finite.max() - finite.min()) or 1.0)
            figure.update_yaxes(range=[finite.min() - pad, finite.max() + pad])
    return portal_figures.page_figure(figure)


def parse_channels(channel: Sequence[int]) -> list[int]:
    """The routes' ``channel=`` (repeated) as the row's channels; a 422 if out of range."""
    try:
        return channels_value(",".join(str(c) for c in channel))
    except ValueError as why:
        raise HTTPException(status_code=422, detail=f"channel: {why}") from None


def parse_window(
    xmin: Optional[float], xmax: Optional[float]
) -> Optional[images.XWindow]:
    """A trace row's ``xmin`` / ``xmax``: ``None`` for neither; a 422 unless xmax is above xmin."""
    if xmin is None and xmax is None:
        return None
    for value in (xmin, xmax):
        if value is not None and not math.isfinite(value):
            raise HTTPException(status_code=422, detail="the x window: finite numbers")
    if xmin is not None and xmax is not None and not xmax > xmin:
        raise HTTPException(
            status_code=422, detail="the x window: 'to' must be above 'from'"
        )
    return (xmin, xmax)


def window_query(window: Optional[images.XWindow]) -> dict[str, object]:
    """The x-window half of a trace tile's or the levels' URL (nothing without one)."""
    if window is None:
        return {}
    return {k: v for k, v in zip(("xmin", "xmax"), window) if v is not None}


def opened_scan(ctx: Context, day: date, scan: int):
    """The scan's run, its s-file and its folder (a 404 when it is not there)."""
    run = ctx.run(day, scan)
    return run, scans.load_frame(run), scans.scan_folder(run)


@dataclass(frozen=True)
class Opened:
    """A strip's scans as a route opens them: the open one, and any others."""

    run: Any  # the open scan's RunDetail
    #: Its s-file, or every scan's merged (rows and bins are its).
    frame: pd.DataFrame
    #: The open scan's folder (its device list and kind).
    folder: Path
    #: Where tiles load from: *folder*, or the :class:`images.ScanSet`.
    source: images.Source
    bin_col: str = SCAN_BIN_COLUMN


def opened_scans(
    ctx: Context,
    day: date,
    scan: int,
    extra: str = "",
    mtol: Optional[float] = None,
    binx: Optional[str] = None,
) -> Opened:
    """:func:`opened_scan` with the set's other scans (*extra*) merged in."""
    run, frame, folder = opened_scan(ctx, day, scan)
    found = scan_set(ctx, day, scan, extra, mtol, binx)
    if found is None:
        return Opened(run, frame, folder, folder)
    return Opened(run, found.frame, folder, found, found.bin_col)


def picture_device(folder: Path, device: str) -> bool:
    """Whether *device* records traces; a 404 when it has no pictures either."""
    if images.is_image_device(folder, device):
        return False
    if images.trace_device(folder, device).kind:
        return True
    raise HTTPException(
        status_code=404, detail=f"{device!r} has no pictures in this scan"
    )


def register(router: APIRouter, ctx: Context) -> None:
    """Add the image routes to *router* (the trace and scalar routes too)."""
    register_traces(router, ctx)
    register_scalars(router, ctx)

    @router.get("/api/images/devices", response_model=DevicesResponse)
    def api_devices(day: date, scan: int) -> JSONResponse:
        """The scan's device folders (each is checked for pictures when picked)."""
        folder = scans.scan_folder(ctx.run(day, scan))
        body = DevicesResponse(devices=images.devices(folder))
        return JSONResponse(body.model_dump(), headers=NO_CACHE)

    @router.get("/api/images/strip", response_model=StripResponse)
    def api_strip(
        day: date,
        scan: int,
        device: str,
        strip: Strip = Query(default="bins", description="a tile per bin or per shot"),
        channel: list[int] = Query(
            default=[0], description="a scope's channels, repeated: overlaid"
        ),
        extra: str = Query(
            default="",
            alias="scans",
            description="more scans: YYYY-MM-DD:N, comma-separated",
        ),
        mtol: Optional[float] = Query(
            default=None, ge=0, description="pool two scans' bins this close in X"
        ),
        bx: Optional[str] = Query(
            default=None,
            description="the column pooled bins are placed by; empty: none",
        ),
    ) -> JSONResponse:
        """The device's tiles, in order, each with its PNG URL."""
        got = opened_scans(ctx, day, scan, extra, mtol, bx)
        traces = picture_device(got.folder, device)
        channels = parse_channels(channel)
        tiles = strip_tiles(
            got.run,
            got.frame,
            strip,
            day=day,
            scan=scan,
            device=device,
            x=bx,  # the tiles' URLs rebuild the set this listing was pooled by
            channels=channels if traces else None,
            source=got.source,
            extra=extra,
            mtol=mtol,
        )
        code = (
            trace_code(day, scan, device, strip, ctx.experiment, channels)
            if traces
            else strip_code(day, scan, device, strip, ctx.experiment)
        )
        body = StripResponse(device=device, strip=strip, tiles=tiles, code=code)
        return JSONResponse(body.model_dump(), headers=NO_CACHE)

    @router.get("/api/images/levels", response_model=LevelsResponse)
    def api_levels(
        day: date,
        scan: int,
        device: str,
        strip: Strip = "bins",
        roi: str = Query(
            default="", description="x0,y0,x1,y1 in pixels; empty is whole"
        ),
        rows: str = Query(
            default="", description="the strip's shot rows, comma-separated (pick)"
        ),
        hits: str = Query(default="none", description="the X-ray-hit filter"),
        centre: Optional[float] = Query(
            default=None, description="centre on the beam: the threshold"
        ),
        filters: str = Query(default="", description="the shot filters (JSON)"),
        bg: str = Query(default="", description="a background scan, YYYY-MM-DD:N"),
        channel: list[int] = Query(
            default=[0], description="a scope's channels, repeated: overlaid"
        ),
        xmin: Optional[float] = Query(
            default=None, description="a trace row's x window: from"
        ),
        xmax: Optional[float] = Query(default=None, description="… to"),
        shown: str = Query(
            default="",
            description="the tiles on screen, their indices comma-separated; empty is all",
        ),
        extra: str = Query(
            default="",
            alias="scans",
            description="more scans: YYYY-MM-DD:N, comma-separated",
        ),
        mtol: Optional[float] = Query(
            default=None, ge=0, description="pool two scans' bins this close in X"
        ),
        bx: Optional[str] = Query(
            default=None,
            description="the column pooled bins are placed by; empty: none",
        ),
    ) -> JSONResponse:
        """One black and white over the tiles shown (the "shared" z scale).

        Reads each of them: its darkest and its brightest pixel (a trace
        device: its lowest and highest sample, the y range). A strip of one
        shot per bin names its rows (``rows``); otherwise the strip is the
        shots the shot filters (``filters``) keep. ``shown`` narrows it to
        the tiles on screen (one row spread over the scan shows a few), so a
        hidden tile is never read for the scale.
        """
        got = opened_scans(ctx, day, scan, extra, mtol, bx)
        traces = picture_device(got.folder, device)
        prep = parse_prep(hits, centre)
        frame = got.frame
        keep = None if rows else keep_rows(frame, filters)
        try:
            found = (
                parse_rows(rows)
                if rows
                else images.tiles(frame, strip, keep=keep, bin_col=got.bin_col)
            )
            found = shown_tiles(found, shown)
            if traces:
                # A trace row's shared y range: its lowest to highest sample.
                lo, hi, count = images.trace_levels(
                    got.source,
                    frame,
                    device,
                    found,
                    parse_channels(channel),
                    parse_window(xmin, xmax),
                )
                body = LevelsResponse(zmin=lo, zmax=hi, tiles=count)
                return JSONResponse(body.model_dump(), headers=NO_CACHE)
            zmin, zmax, count = images.shared_levels(
                got.source,
                frame,
                device,
                found,
                parse_roi(roi),
                prep,
                open_background(ctx, bg),
            )
        except images.NoImage as why:
            raise HTTPException(status_code=404, detail=str(why)) from None
        body = LevelsResponse(zmin=zmin, zmax=zmax, tiles=count)
        return JSONResponse(body.model_dump(), headers=NO_CACHE)

    @router.get("/api/images/tile.png")
    def api_tile(
        day: date,
        scan: int,
        device: str,
        strip: Strip = "bins",
        index: int = Query(default=0, description="the bin's position, or the row"),
        cmap: str = Query(default="gray", description="a matplotlib colour map"),
        roi: str = Query(
            default="", description="x0,y0,x1,y1 in pixels; empty is whole"
        ),
        zmin: Optional[float] = Query(default=None, description="black at this value"),
        zmax: Optional[float] = Query(default=None, description="white at this value"),
        bar: bool = Query(
            default=False, description="with a colour bar (the big view)"
        ),
        cbar: bool = Query(
            default=False, description="a slim labelled colour bar at the tile's right"
        ),
        hits: str = Query(default="none", description="the X-ray-hit filter"),
        centre: Optional[float] = Query(
            default=None, description="centre on the beam: the threshold"
        ),
        filters: str = Query(
            default="", description="the shot filters (JSON): a bin's shots"
        ),
        bg: str = Query(
            default="",
            description="a background scan, YYYY-MM-DD:N: its average off each shot",
        ),
        channel: list[int] = Query(
            default=[0], description="a scope's channels, repeated: overlaid"
        ),
        xmin: Optional[float] = Query(
            default=None, description="a trace row's x window: from"
        ),
        xmax: Optional[float] = Query(default=None, description="… to"),
        extra: str = Query(
            default="",
            alias="scans",
            description="more scans: YYYY-MM-DD:N, comma-separated",
        ),
        mtol: Optional[float] = Query(
            default=None, ge=0, description="pool two scans' bins this close in X"
        ),
        bx: Optional[str] = Query(
            default=None,
            description="the column pooled bins are placed by; empty: none",
        ),
    ) -> Response:
        """One tile's PNG: the shot's image, or the bin's average, in its ROI.

        A trace device's tile is its trace drawn small (a bin: its shots'
        mean on their shared x axis), ``zmin``/``zmax`` its y range.

        With ``zmin`` and ``zmax`` both given they are black and white
        (a fixed or shared z scale); otherwise the tile's own darkest and
        brightest pixel. A tile with no image (the device missed every shot of it,
        or its files are not there) is a 404 naming why.
        """
        got = opened_scans(ctx, day, scan, extra, mtol, bx)
        traces = picture_device(got.folder, device)
        frame = got.frame
        keep = keep_rows(frame, filters) if strip == "bins" else None
        rows = tile_rows(frame, strip, index, keep, got.bin_col)
        levels = (zmin, zmax) if zmin is not None and zmax is not None else None
        if traces:
            png = trace_tile_png(
                got.source,
                frame,
                device,
                rows,
                parse_channels(channel),
                levels,
                parse_window(xmin, xmax),
                images.trace_device(got.folder, device).channels,
            )
            return Response(png, media_type="image/png", headers=NO_CACHE)
        prep = parse_prep(hits, centre)
        png = tile_png(
            got.source,
            frame,
            device,
            rows,
            parse_roi(roi),
            cmap,
            levels,
            bar,
            prep,
            cbar,
            open_background(ctx, bg),
        )
        return Response(png, media_type="image/png", headers=NO_CACHE)

    @router.get("/api/images/colourbar.png")
    def api_colour_bar(
        zmin: float,
        zmax: float,
        cmap: str = Query(default="gray", description="a colour map"),
        length: int = Query(
            default=480, ge=64, le=4000, description="the ramp, pixels"
        ),
        vertical: bool = Query(
            default=False, description="upright, values at its right"
        ),
    ) -> Response:
        """The strip's one colour bar on a shared or fixed z scale: *zmin* to *zmax*."""
        png = grid_bar_png(cmap, zmin, zmax, length, vertical)
        return Response(png, media_type="image/png")


def register_scalars(router: APIRouter, ctx: Context) -> None:
    """Add the scan scalars' route (the line under the conditions) to *router*."""

    @router.get("/api/images/scalars")
    def api_scalars(
        day: date,
        scan: int,
        cols: str = Query(description="the s-file columns, as a JSON list"),
        stat: str = Query(
            default="mean_std", description="mean_std | median_iqr | mean_stderr"
        ),
        filters: str = Query(default="", description="the set's shot filters (JSON)"),
        extra: str = Query(
            default="",
            alias="scans",
            description="more scans: YYYY-MM-DD:N, comma-separated",
        ),
        mtol: Optional[float] = Query(
            default=None, ge=0, description="pool two scans' bins this close in X"
        ),
        bx: Optional[str] = Query(
            default=None,
            description="the column pooled bins are placed by; empty: none",
        ),
    ) -> JSONResponse:
        """The Images page's scan scalars: each column over the shots the strip shows.

        With ``scans``, over every scan's shots together.
        """
        try:
            columns = json.loads(cols)
            if not (
                isinstance(columns, list) and all(isinstance(c, str) for c in columns)
            ):
                raise ValueError
        except ValueError:
            raise HTTPException(
                status_code=400, detail="cols: a JSON list of column names"
            ) from None
        if stat not in scans.SCALAR_STATS:
            raise HTTPException(
                status_code=400, detail=f"stat: one of {', '.join(scans.SCALAR_STATS)}"
            )
        frame = opened_scans(ctx, day, scan, extra, mtol, bx).frame
        keep = keep_rows(frame, filters)
        body = scans.scan_scalars(frame, keep, columns, stat)
        body["code"] = scalars_code(day, scan, columns, stat, filters, ctx.experiment)
        return JSONResponse(body, headers=NO_CACHE)


def register_traces(router: APIRouter, ctx: Context) -> None:
    """Add the trace tile's figure route (the big view's) to *router*."""

    @router.get("/api/images/trace", response_model=TraceResponse)
    def api_trace(
        day: date,
        scan: int,
        device: str,
        strip: Strip = "bins",
        index: int = Query(default=0, description="the bin's position, or the row"),
        channel: list[int] = Query(
            default=[0], description="a scope's channels, repeated: overlaid"
        ),
        xmin: Optional[float] = Query(
            default=None, description="a trace row's x window: from"
        ),
        xmax: Optional[float] = Query(default=None, description="… to"),
        filters: str = Query(
            default="", description="the shot filters (JSON): a bin's shots"
        ),
        extra: str = Query(
            default="",
            alias="scans",
            description="more scans: YYYY-MM-DD:N, comma-separated",
        ),
        mtol: Optional[float] = Query(
            default=None, ge=0, description="pool two scans' bins this close in X"
        ),
        bx: Optional[str] = Query(
            default=None,
            description="the column pooled bins are placed by; empty: none",
        ),
    ) -> JSONResponse:
        """A trace tile as a figure (the big view): the shot's trace, or the bin's mean.

        The same rows and refusals as its ``tile.png``; a figure, not a
        PNG, so the big view zooms and reads values off.
        """
        got = opened_scans(ctx, day, scan, extra, mtol, bx)
        if not picture_device(got.folder, device):
            raise HTTPException(status_code=404, detail=f"{device!r} records no traces")
        frame = got.frame
        keep = keep_rows(frame, filters) if strip == "bins" else None
        rows = tile_rows(frame, strip, index, keep, got.bin_col)
        names = images.trace_device(got.folder, device).channels
        try:
            lines, notes, missing = tile_lines(
                got.source, frame, device, rows, parse_channels(channel), names
            )
            figure = trace_view(
                lines, device, parse_window(xmin, xmax), channels_named=len(names) > 1
            )
        except images.NoImage as why:
            raise HTTPException(status_code=404, detail=str(why)) from None
        body = TraceResponse(
            figure=figure,
            points=max(int(line[2].data.shape[0]) for line in lines),
            shots=len(rows),
            note="; ".join(notes),
            missing=missing,
        )
        return JSONResponse(body.model_dump(), headers=NO_CACHE)
