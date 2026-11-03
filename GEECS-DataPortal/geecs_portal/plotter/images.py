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
from geecs_data_utils.data.image_prep import HIT_FILTERS, ImagePrep
from geecs_data_utils.io import shot_images
from geecs_data_utils.io.display_png import window_levels

#: The X-ray-hit filters as the rail names them, in its order; one per
#: ``image_prep.HIT_FILTERS`` (the import-time check keeps the two in step).
HIT_LABELS = {
    "none": "keep",
    "hot": "remove hot pixels",
    "median3": "median 3×3",
    "median5": "median 5×5",
}
if set(HIT_LABELS) != set(HIT_FILTERS):  # pragma: no cover - a developer's slip
    raise RuntimeError("every image_prep.HIT_FILTERS entry needs a HIT_LABELS label")

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
    #: The bin it is (bins) or its shot was taken in (shots), as shown
    #: (``""``: no ``Bin #``); a 2-D map cell finds its tile by it.
    bin: str = ""


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


def trace_kind(folder: Path, device: str, devices: Optional[list[str]] = None) -> str:
    """``"lineout"`` or ``"waveform"`` for a device whose stack holds traces, else ``""``.

    A trace has no strip (one shot carries its own axis, so shots do not
    average); the Images page draws it one shot at a time.
    """
    probe = shot_images.device_kind(folder, device, devices=devices)
    if probe.kind != "stack":
        return ""
    content = shot_images.stack_content(probe)
    return "" if content == "image" else content


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

    ``geecs_data_utils.data.binning.bin_rows`` over the whole s-file (its
    positions are the shot loaders' rows): the rows with an empty
    ``Bin #`` a bin of their own (last). A bin whose measurements are all
    missing is still a tile, though the binned plot drops it.

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
    """The scan variable over *rows* (its median), as shown; ``""`` without one.

    The value alone: the strip's title names the variable once.
    """
    if values is None:
        return ""
    picked = values.iloc[[r - 1 for r in rows]].dropna()
    if not len(picked):
        return ""
    return f"{picked.median():.4g}"


def pick_row(rows: tuple[int, ...], by: Optional[pd.Series] = None) -> int:
    """The one shot shown for a bin: its first, or its highest *by*.

    *by* is a column per s-file row (the plot's Y1); its missing values
    are skipped, and a bin with none of them shows its first shot.
    """
    if by is None:
        return rows[0]
    values = by.iloc[[r - 1 for r in rows]].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).any():
        return rows[0]
    return rows[int(np.nanargmax(values))]


def tiles(
    frame: pd.DataFrame,
    strip: str,
    scan_values: Optional[pd.Series] = None,
    shot_numbers: Optional[pd.Series] = None,
    pick_by: Optional[pd.Series] = None,
) -> list[Tile]:
    """The strip's tiles, in order.

    A bin's caption is its scan variable's value, ``"17.5 · 5 shots"``
    (``"bin 2 · 5 shots"`` without one, ``"17.5 · bin 2 · 5 shots"`` when
    another bin has the same value); a shot's is ``"shot 3 · 17.5"``;
    a bin's one shot (``"pick"``) is ``"17.5 · shot 12"``. The variable's
    name is the strip's title, said once.

    Parameters
    ----------
    frame : pandas.DataFrame
        The scan's s-file, one row per shot.
    strip : {"bins", "shots", "pick"}
        One tile per bin (its average), one per shot, or one shot per bin
        (``pick``: its first, or its highest *pick_by*); a picked tile's
        ``index`` is its shot's row, as a shot tile's is.
    scan_values : pandas.Series, optional
        The scan variable per row, shown in each caption.
    shot_numbers : pandas.Series, optional
        The shot number per row (the s-file's), for the shot captions.
    pick_by : pandas.Series, optional
        For ``pick``: the column whose highest value picks each bin's
        shot (the plot's Y1); ``None`` picks the first shot.

    Returns
    -------
    list of Tile
    """

    def shot_label(row: int) -> str:
        shot = row if shot_numbers is None else shot_numbers.iloc[row - 1]
        return f"shot {_number(shot, f'? (row {row})')}"

    if strip == "shots":
        out = []
        bin_numbers = frame.get("Bin #")
        for row in range(1, len(frame) + 1):
            label = shot_label(row)
            in_bin = (
                "" if bin_numbers is None else _number(bin_numbers.iloc[row - 1], "")
            )
            value = _value(scan_values, (row,))
            caption = f"{label} · {value}" if value else label
            out.append(Tile(row, (row,), caption, in_bin))
        return out
    found = list(bins(frame))
    wheres = [_value(scan_values, rows) for _, rows in found]
    # A value two bins share (a scan back over its start, or medians alike
    # to 4 figures) names the bin too, so no two tiles read the same.
    shared = {w for w in wheres if w and wheres.count(w) > 1}
    out = []
    for index, ((label, rows), value) in enumerate(zip(found, wheres)):
        number = f"bin {_number(label, 'none')}"
        where = (f"{value} · {number}" if value in shared else value) or number
        if strip == "pick":
            row = pick_row(rows, pick_by)
            caption = f"{where} · {shot_label(row)}"
            out.append(Tile(row, (row,), caption, _number(label, "")))
            continue
        count = f"{len(rows)} shot{'s' if len(rows) != 1 else ''}"
        out.append(Tile(index, rows, f"{where} · {count}", _number(label, "")))
    return out


def tile_array(
    folder: Path,
    frame: pd.DataFrame,
    device: str,
    rows: tuple[int, ...],
    prep: ImagePrep = ImagePrep(),
) -> np.ndarray:
    """The tile's pixels: one shot's image, or its shots' average.

    *prep* cleans each shot before the average
    (``geecs_data_utils.data.image_prep``: X-ray hits out, then centred
    on its beam), so a bin's average is of centred shots.

    ``geecs_data_utils.io.shot_images.average_rows``: shots the device
    missed, or whose file is missing or unreadable, are left out of the
    average, never replaced by a neighbour's image.

    Raises
    ------
    NoImage
        No shot of the tile has an image; the reason why.
    """
    found = shot_images.average_rows(
        folder, frame, device, rows, prepare=prep.apply if prep.active else None
    )
    if found.array is None:
        raise NoImage(found.reason or found.kind)
    return found.array


_kept: OrderedDict[tuple, np.ndarray | tuple[str, float]] = OrderedDict()
_kept_bytes = 0
_kept_lock = threading.Lock()


def kept_tile_array(
    folder: Path,
    frame: pd.DataFrame,
    device: str,
    rows: tuple[int, ...],
    prep: ImagePrep = ImagePrep(),
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
        return tile_array(folder, frame, device, rows, prep)
    key = (str(folder), device, rows, len(frame), stamp, prep)
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
        pixels = tile_array(folder, frame, device, rows, prep)
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
    prep: ImagePrep = ImagePrep(),
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
            pixels = crop(kept_tile_array(folder, frame, device, tile.rows, prep), roi)
        except NoImage:
            return None
        return window_levels(pixels, 0.0, 100.0)  # its darkest and brightest

    with ThreadPoolExecutor(max_workers=8) as pool:
        found = [w for w in pool.map(levels, tiles) if w is not None]
    if not found:
        raise NoImage("no tile has an image")
    return min(lo for lo, _ in found), max(hi for _, hi in found), len(found)
