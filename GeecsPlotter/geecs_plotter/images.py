"""A scan's camera images: which devices, which shots, one PNG per tile.

GEECSplotter's image strip, as calls into GEECS-Data-Utils: the shot ladder
and the trace check are ``geecs_data_utils.io.shot_images``, the bin key is
``geecs_data_utils.data.binning``, the per-bin average
``geecs_data_utils.io.images.average_frames`` and the display PNG
``geecs_data_utils.io.display_png``. Nothing here is numerics.

A shot is a row of the scan's s-file, 1-based. A device that missed a shot
(its row says so) gives no image for it, and is left out of its bin's
average — never a neighbouring shot's image. Read-only, and lazy: each
tile loads its own shots when the browser asks for it.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from geecs_data_utils.data.binning import bin_rows
from geecs_data_utils.io import shot_images
from geecs_data_utils.io.display_png import window_levels

#: The colour maps offered (matplotlib names), "gray" first, the default:
#: the perceptual ones, then the classic false-colour tables.
CMAPS = (
    "gray",
    "viridis",
    "magma",
    "inferno",
    "plasma",
    "cividis",
    "jet",
    "turbo",
    "hot",
    "rainbow",
    "nipy_spectral",
    "gist_ncar",
    "cubehelix",
    "coolwarm",
    "gray_r",
)

#: A region of interest, ``(x0, y0, x1, y1)`` in pixels, the end excluded.
Roi = tuple[int, int, int, int]

#: Pixels kept in memory: a strip's tiles are read once for a shared z
#: scale and again to draw. A strip larger than this is read twice.
_KEPT_BYTES = 512 * 2**20

#: Seconds a tile's "no image" is remembered: long enough to answer the
#: page's ask for the reason, short enough that a passing failure (a share
#: blip, a file still being written) is looked at again on the next try.
_REFUSAL_SECONDS = 30.0


class NoImage(LookupError):
    """No image for this tile; the message says why."""


@dataclass(frozen=True)
class Tile:
    """One tile of the strip: which shots it shows, and its caption."""

    #: The bin's position in the strip (bins), or the shot's row (shots).
    index: int
    #: The 1-based s-file rows whose images it shows.
    rows: tuple[int, ...]
    caption: str


def is_image_device(
    folder: Path, device: str, devices: Optional[list[str]] = None
) -> bool:
    """Whether *device* holds pictures: per-shot images or an image stack.

    Vendor formats (``.himg``), non-image files and trace stacks (scope
    traces, spectra) do not: none of them draws as a picture. An unknown
    name is not one either (the folder listing is the guard).
    """
    probe = shot_images.device_kind(folder, device, devices=devices)
    return probe.kind == "native" or (
        probe.kind == "stack" and shot_images.stack_content(probe) == "image"
    )


def devices(folder: Path) -> list[str]:
    """The scan's device folders: one listing, no probe of each.

    Probing every folder for pictures costs a listing of each over the
    share (tens of seconds cold, for a scan with 70 devices), so the list
    is every folder and the picked one is checked
    (:func:`is_image_device`).
    """
    return shot_images.image_devices(folder)


def bins(frame: pd.DataFrame) -> list[tuple[object, tuple[int, ...]]]:
    """Each bin's label and rows, in bin order (GEECSplotter's ``Bin #``).

    ``geecs_data_utils.data.binning.bin_rows``: the binned plot's bins,
    the rows with an empty ``Bin #`` a bin of their own (last).

    Raises
    ------
    NoImage
        The scan has no ``Bin #`` column.
    """
    try:
        return bin_rows(frame)
    except KeyError:
        raise NoImage("this scan has no Bin # column") from None


def _number(value: object, missing: str) -> str:
    """A label or shot number as shown: 1.0 as 1, NaN as *missing*."""
    if isinstance(value, float):
        if value != value:
            return missing
        if value.is_integer():
            return str(int(value))
    return str(value)


def _value(values: Optional[pd.Series], rows: tuple[int, ...]) -> str:
    """The scan variable over *rows* (its median), for the caption."""
    if values is None:
        return ""
    picked = values.iloc[[r - 1 for r in rows]].dropna()
    return f" · {picked.median():.4g}" if len(picked) else ""


def tiles(
    frame: pd.DataFrame,
    strip: str,
    scan_values: Optional[pd.Series] = None,
    shot_numbers: Optional[pd.Series] = None,
) -> list[Tile]:
    """The strip's tiles, in order.

    Parameters
    ----------
    frame : pandas.DataFrame
        The scan's s-file, one row per shot.
    strip : {"bins", "shots"}
        One tile per bin, or one per shot.
    scan_values : pandas.Series, optional
        The scan variable per row, shown in each caption.
    shot_numbers : pandas.Series, optional
        The shot number per row (the s-file's), for the shot captions.

    Returns
    -------
    list of Tile
    """
    if strip == "shots":
        out = []
        for row in range(1, len(frame) + 1):
            shot = row if shot_numbers is None else shot_numbers.iloc[row - 1]
            label = f"shot {_number(shot, f'? (row {row})')}"
            out.append(Tile(row, (row,), f"{label}{_value(scan_values, (row,))}"))
        return out
    return [
        Tile(
            index,
            rows,
            f"bin {_number(label, 'none')} · {len(rows)} shot{'s' if len(rows) != 1 else ''}"
            f"{_value(scan_values, rows)}",
        )
        for index, (label, rows) in enumerate(bins(frame))
    ]


def tile_array(
    folder: Path, frame: pd.DataFrame, device: str, rows: tuple[int, ...]
) -> np.ndarray:
    """The tile's pixels: one shot's image, or its shots' average.

    ``geecs_data_utils.io.shot_images.average_rows``: shots the device
    missed, or whose file is missing or unreadable, are left out of the
    average, never replaced by a neighbour's image.

    Raises
    ------
    NoImage
        No shot of the tile has an image; the reason why.
    """
    found = shot_images.average_rows(folder, frame, device, rows)
    if found.array is None:
        raise NoImage(found.reason or found.kind)
    return found.array


_kept: OrderedDict[tuple, np.ndarray | tuple[str, float]] = OrderedDict()
_kept_bytes = 0
_kept_lock = threading.Lock()


def kept_tile_array(
    folder: Path, frame: pd.DataFrame, device: str, rows: tuple[int, ...]
) -> np.ndarray:
    """:func:`tile_array`, remembered until the camera's folder changes.

    The key carries the device folder's modification time and the table's
    length, so a file or a row added while a scan runs is read again. A
    tile with no image is remembered for ``_REFUSAL_SECONDS`` only (its
    reason), so the page's ask for the reason does not search the share
    again while a passing failure is never made permanent. The store
    holds at most ``_KEPT_BYTES`` of pixels, oldest out first.

    Raises
    ------
    NoImage
        As :func:`tile_array`.
    """
    try:
        stamp = (folder / device).stat().st_mtime_ns
    except OSError:
        return tile_array(folder, frame, device, rows)
    key = (str(folder), device, rows, len(frame), stamp)
    now = time.monotonic()
    with _kept_lock:
        kept = _kept.get(key)
        if isinstance(kept, tuple) and now - kept[1] > _REFUSAL_SECONDS:
            del _kept[key]  # look again: the failure may have passed
            kept = None
        if kept is not None:
            _kept.move_to_end(key)
    if isinstance(kept, tuple):
        raise NoImage(kept[0])
    if kept is not None:
        return kept
    try:
        pixels = tile_array(folder, frame, device, rows)
    except NoImage as why:
        _keep(key, (str(why), now))
        raise
    pixels.flags.writeable = False  # shared between requests: never edited
    _keep(key, pixels)
    return pixels


def _keep(key: tuple, value: np.ndarray | tuple[str, float]) -> None:
    """Store *value*, dropping the oldest until the store fits its bytes."""
    global _kept_bytes
    size = value.nbytes if isinstance(value, np.ndarray) else 0
    if size > _KEPT_BYTES:
        return  # one tile bigger than the whole store: never kept
    with _kept_lock:
        if key in _kept:
            return
        _kept[key] = value
        _kept_bytes += size
        while _kept_bytes > _KEPT_BYTES:
            _, old = _kept.popitem(last=False)
            _kept_bytes -= old.nbytes if isinstance(old, np.ndarray) else 0


def crop(pixels: np.ndarray, roi: Optional[Roi]) -> np.ndarray:
    """The ROI of *pixels* (a view), or all of them when *roi* is ``None``.

    A region past the image's edge is cut at the edge.

    Raises
    ------
    NoImage
        The region lies wholly outside the image.
    """
    if roi is None:
        return pixels
    x0, y0, x1, y1 = roi
    part = pixels[y0:y1, x0:x1]
    if part.size == 0:
        height, width = pixels.shape[:2]
        raise NoImage(f"the ROI is outside this {width}×{height} image")
    return part


def shared_levels(
    folder: Path,
    frame: pd.DataFrame,
    device: str,
    tiles: list[Tile],
    roi: Optional[Roi] = None,
) -> tuple[float, float, int]:
    """One black and one white for every tile: the darkest and brightest pixel.

    Over each tile's ROI; the brightest tile sets white, so no tile
    saturates. This reads EVERY tile of the strip, the whole scan for a
    strip of shots (the author's ruling: the scale must be consistent
    across all of them), a few at a time; the page says so while it runs.

    Returns
    -------
    tuple
        ``(black, white, tiles)``: the levels and how many tiles had an
        image.

    Raises
    ------
    NoImage
        No tile has an image.
    """

    def levels(tile: Tile) -> Optional[tuple[float, float]]:
        try:
            pixels = crop(kept_tile_array(folder, frame, device, tile.rows), roi)
        except NoImage:
            return None
        return window_levels(pixels, 0.0, 100.0)  # its darkest and brightest

    with ThreadPoolExecutor(max_workers=8) as pool:
        found = [w for w in pool.map(levels, tiles) if w is not None]
    if not found:
        raise NoImage("no tile has an image")
    return min(lo for lo, _ in found), max(hi for _, hi in found), len(found)
