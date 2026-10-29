"""The image strip's JSON and PNG routes.

``/api/images/strip`` lists a device's tiles (one per bin or per shot) with
each tile's PNG URL and the notebook snippet; ``/api/images/tile.png`` is
one tile: a shot's image, or its bin's average, drawn with the colour map
and z scale. Each is a few calls into :mod:`geecs_portal.plotter.images`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from geecs_data_utils.io.display_png import to_display_png
from pydantic import BaseModel

from geecs_portal.plotter import images, scans
from geecs_portal.plotter.routes import Context
from geecs_portal.plotter.routes.api import NO_CACHE
from geecs_portal.plotter.settings import SettingsError, roi_value

Strip = Literal["bins", "shots"]


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


def tile_url(day: date, scan: int, device: str, strip: str, index: int) -> str:
    """A tile's PNG URL, relative to the app root; the page adds the display."""
    query = {"day": day.isoformat(), "scan": scan, "device": device}
    return f"api/images/tile.png?{urlencode({**query, 'strip': strip, 'index': index})}"


def strip_tiles(
    run, frame, strip: str, *, day: date, scan: int, device: str
) -> list[TileModel]:
    """The strip's tiles with their URLs; captions carry the scan variable."""
    x = scans.default_x(run, scans.columns(frame))
    values = scans.plot_series(frame, [x])[x] if x else None
    try:
        found = images.tiles(frame, strip, values, scans.shot_axis(frame))
    except images.NoImage as why:
        raise HTTPException(status_code=404, detail=str(why)) from None
    return [
        TileModel(
            index=t.index,
            rows=list(t.rows),
            caption=t.caption,
            bin=t.bin,
            url=tile_url(day, scan, device, strip, t.index),
        )
        for t in found
    ]


def strip_code(day: date, scan: int, device: str, strip: str, experiment: str) -> str:
    """The notebook snippet that loads each tile's pixels as the page does."""
    where = f", experiment={experiment!r}" if experiment else ""
    return (
        "from datetime import date\n"
        "from geecs_portal.plotter import images, scans\n\n"
        f"run = scans.find_scan(date({day.year}, {day.month}, {day.day}), {scan}{where})\n"
        "frame, folder = scans.load_frame(run), scans.scan_folder(run)\n"
        f"for tile in images.tiles(frame, {strip!r}):\n"
        "    try:\n"
        f"        pixels = images.tile_array(folder, frame, {device!r}, tile.rows)\n"
        "    except images.NoImage as why:\n"
        "        print(tile.caption, '—', why)\n"
        "        continue\n"
        "    print(tile.caption, pixels.shape)\n"
    )


def tile_rows(frame, strip: str, index: int) -> tuple[int, ...]:
    """The rows a tile shows: the shot's row, or the bin's (by position).

    Raises
    ------
    HTTPException
        404 when the scan has no bins, or no bin at *index*.
    """
    if strip == "shots":
        return (index,)
    try:
        found = images.bins(frame)
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
) -> bytes:
    """The tile's pixels, in its ROI, drawn as a PNG; a 404 naming why not.

    With *bar*, a figure with a colour bar beside the picture (the big
    view): the same colour map and levels, so its values read off.
    """
    try:
        pixels = images.crop(images.kept_tile_array(folder, frame, device, rows), roi)
        if bar:
            return _with_colour_bar(pixels, cmap, levels)
        # Without levels, the tile's own darkest to brightest pixel.
        return to_display_png(pixels, cmap=cmap, plo=0.0, phi=100.0, levels=levels)
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
    from matplotlib import colormaps
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
        cmap=cmap if cmap in colormaps else "gray",
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )
    fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.03)
    return figure_png(fig, tight=True)


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
    ) -> JSONResponse:
        """One black and white for the whole strip (the "shared" z scale).

        Reads every tile of the strip: its darkest and its brightest pixel.
        """
        run, frame, folder = opened(day, scan)
        picture_device(folder, device)
        try:
            found = images.tiles(frame, strip)
            zmin, zmax, count = images.shared_levels(
                folder, frame, device, found, parse_roi(roi)
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
    ) -> Response:
        """One tile's PNG: the shot's image, or the bin's average, in its ROI.

        With ``zmin`` and ``zmax`` both given they are black and white
        (a fixed or shared z scale); otherwise the tile's own darkest and
        brightest pixel. A tile with no image (the device missed every shot of it,
        or its files are not there) is a 404 naming why.
        """
        _, frame, folder = opened(day, scan)
        picture_device(folder, device)
        rows = tile_rows(frame, strip, index)
        levels = (zmin, zmax) if zmin is not None and zmax is not None else None
        png = tile_png(folder, frame, device, rows, parse_roi(roi), cmap, levels, bar)
        return Response(png, media_type="image/png", headers=NO_CACHE)
