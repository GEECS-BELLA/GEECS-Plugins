"""The image strip's JSON and PNG routes.

``/api/images/strip`` lists a device's tiles (one per bin or per shot) with
each tile's PNG URL and the notebook snippet; ``/api/images/tile.png`` is
one tile: a shot's image, or its bin's average, drawn with the colour map
and z scale. Each is a few calls into :mod:`geecs_portal.plotter.images`.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from geecs_data_utils.data.image_prep import HIT_FILTERS, ImagePrep
from geecs_data_utils.io.display_png import colour_bar_png, safe_cmap, to_display_png
from pydantic import BaseModel

from geecs_portal.plotter import images, scans
from geecs_portal.plotter.routes import Context
from geecs_portal.plotter.routes.api import NO_CACHE
from geecs_portal.plotter.settings import SettingsError, roi_value

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


class LevelsResponse(BaseModel):
    """The strip's shared z scale, and how many tiles gave it."""

    #: The pixel values drawn black and white.
    zmin: float
    zmax: float
    tiles: int


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
) -> str:
    """A tile's PNG URL, relative to the app root; the page adds the display.

    A bin's one shot (``pick``) is a shot tile: its index is its row. A
    bin carries the shot filters (*filters*, the set's JSON): its index is
    its place among the bins they leave, and it averages only their shots.
    """
    query = {"day": day.isoformat(), "scan": scan, "device": device}
    query.update(strip="shots" if strip == "pick" else strip, index=index)
    if strip == "bins" and filters:
        query["filters"] = filters
    return f"api/images/tile.png?{urlencode({**query, **prep_query(prep)})}"


def parse_rows(text: str) -> list[images.Tile]:
    """The ``rows`` parameter as one shot tile each; a bad one is a 422."""
    try:
        return [images.Tile(r, (r,), "") for r in (int(v) for v in text.split(","))]
    except ValueError:
        raise HTTPException(
            status_code=422, detail=f"rows={text!r}: whole numbers"
        ) from None


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
    x = scans.default_x(run, scans.columns(frame))
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
) -> list[TileModel]:
    """The strip's tiles with their URLs; captions carry the scan variable's value.

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
    try:
        found = images.tiles(
            frame, strip, values, scans.shot_axis(frame), pick_by=by, keep=keep
        )
    except images.NoImage as why:
        raise HTTPException(status_code=404, detail=str(why)) from None
    return [
        TileModel(
            index=t.index,
            rows=list(t.rows),
            caption=t.caption,
            bin=t.bin,
            url=tile_url(day, scan, device, strip, t.index, prep, filters),
        )
        for t in found
    ]


def strip_code(
    day: date,
    scan: int,
    device: str,
    strip: str,
    experiment: str,
    pick_by: str = "",
    prep: ImagePrep = ImagePrep(),
) -> str:
    """The notebook snippet that loads each tile's pixels as the page does."""
    where = f", experiment={experiment!r}" if experiment else ""
    by = f", pick_by=frame[{pick_by!r}]" if strip == "pick" and pick_by else ""
    return (
        "from datetime import date\n"
        "from geecs_data_utils.data.image_prep import ImagePrep\n"
        "from geecs_portal.plotter import images, scans\n\n"
        f"run = scans.find_scan(date({day.year}, {day.month}, {day.day}), {scan}{where})\n"
        "frame, folder = scans.load_frame(run), scans.scan_folder(run)\n"
        f"prep = ImagePrep(hits={prep.hits!r}, centre={prep.centre!r})\n"
        f"for tile in images.tiles(frame, {strip!r}{by}):\n"
        "    try:\n"
        f"        pixels = images.tile_array(folder, frame, {device!r}, tile.rows, prep)\n"
        "    except images.NoImage as why:\n"
        "        print(tile.caption, '—', why)\n"
        "        continue\n"
        "    print(tile.caption, pixels.shape)\n"
    )


def keep_rows(frame, filters: str) -> Optional[frozenset[int]]:
    """The rows the shot filters keep (``images.shot_rows``); a 422 when they cannot apply."""
    try:
        return images.shot_rows(frame, filters)
    except images.NoImage as why:
        raise HTTPException(status_code=422, detail=str(why)) from None


def tile_rows(
    frame, strip: str, index: int, keep: Optional[frozenset[int]] = None
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
        found = images.kept_bins(frame, keep)
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
) -> bytes:
    """The tile's pixels, in its ROI, drawn as a PNG; a 404 naming why not.

    With *bar*, a figure with a colour bar beside the picture (the big
    view): the same colour map and levels, so its values read off. With
    *cbar*, the tile itself with a slim labelled bar at its right (a strip
    whose tiles each scale themselves).
    """
    try:
        pixels = images.crop(
            images.kept_tile_array(folder, frame, device, rows, prep), roi
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


def register(router: APIRouter, ctx: Context) -> None:
    """Add the image routes to *router*."""

    def opened(day: date, scan: int):
        run = ctx.run(day, scan)
        return run, scans.load_frame(run), scans.scan_folder(run)

    def picture_device(folder: Path, device: str) -> None:
        if not images.is_image_device(folder, device):
            raise HTTPException(
                status_code=404, detail=f"{device!r} has no pictures in this scan"
            )

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
    ) -> JSONResponse:
        """The device's tiles, in order, each with its PNG URL."""
        run, frame, folder = opened(day, scan)
        picture_device(folder, device)
        body = StripResponse(
            device=device,
            strip=strip,
            tiles=strip_tiles(run, frame, strip, day=day, scan=scan, device=device),
            code=strip_code(day, scan, device, strip, ctx.experiment),
        )
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
    ) -> JSONResponse:
        """One black and white for the whole strip (the "shared" z scale).

        Reads every tile of the strip: its darkest and its brightest pixel.
        A strip of one shot per bin names its rows (``rows``); otherwise
        the strip is the shots the shot filters (``filters``) keep.
        """
        run, frame, folder = opened(day, scan)
        picture_device(folder, device)
        prep = parse_prep(hits, centre)
        keep = None if rows else keep_rows(frame, filters)
        try:
            found = parse_rows(rows) if rows else images.tiles(frame, strip, keep=keep)
            zmin, zmax, count = images.shared_levels(
                folder, frame, device, found, parse_roi(roi), prep
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
    ) -> Response:
        """One tile's PNG: the shot's image, or the bin's average, in its ROI.

        With ``zmin`` and ``zmax`` both given they are black and white
        (a fixed or shared z scale); otherwise the tile's own darkest and
        brightest pixel. A tile with no image (the device missed every shot of it,
        or its files are not there) is a 404 naming why.
        """
        _, frame, folder = opened(day, scan)
        picture_device(folder, device)
        keep = keep_rows(frame, filters) if strip == "bins" else None
        rows = tile_rows(frame, strip, index, keep)
        levels = (zmin, zmax) if zmin is not None and zmax is not None else None
        prep = parse_prep(hits, centre)
        png = tile_png(
            folder, frame, device, rows, parse_roi(roi), cmap, levels, bar, prep, cbar
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
