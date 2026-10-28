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
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from geecs_data_utils.data.binning import BinningConfig, compute_bin_key
from geecs_data_utils.io import shot_images
from geecs_data_utils.io.display_png import window_levels
from geecs_data_utils.io.images import average_frames

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

#: Tiles kept in memory: a strip's tiles are read once for a shared z
#: scale and again to draw, and the camera's files do not change.
_KEPT_TILES = 64


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

    Raises
    ------
    NoImage
        The scan has no ``Bin #`` column.
    """
    try:
        labels, _ = compute_bin_key(frame.reset_index(drop=True), BinningConfig())
    except KeyError:
        raise NoImage("this scan has no Bin # column") from None
    rows = pd.Series(range(1, len(frame) + 1), index=labels.index)
    return [
        (label, tuple(int(r) for r in group))
        for label, group in rows.groupby(labels, sort=True, observed=True)
    ]


def _number(label: object) -> object:
    """A whole-number bin label as an integer (``Bin #`` may read as 1.0)."""
    return int(label) if isinstance(label, float) and label.is_integer() else label


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
            out.append(Tile(row, (row,), f"shot {shot}{_value(scan_values, (row,))}"))
        return out
    return [
        Tile(
            index,
            rows,
            f"bin {_number(label)} · {len(rows)} shot{'s' if len(rows) != 1 else ''}"
            f"{_value(scan_values, rows)}",
        )
        for index, (label, rows) in enumerate(bins(frame))
    ]


def row_array(
    folder: Path, frame: pd.DataFrame, device: str, row: int
) -> shot_images.ShotArray:
    """One row's image from *device*, or the refusal and its reason.

    The row's own ``acq_timestamp`` (when the s-file has one) picks the
    file; a row that says the device missed the shot is refused here.
    """
    if not 1 <= row <= len(frame):
        return shot_images.ShotArray(kind="missing", reason=f"no shot {row}")
    stamp, present = shot_images.row_acq_timestamp(frame, device, row)
    if present and stamp is None:
        return shot_images.ShotArray(
            kind="missing", reason="the device missed this shot"
        )
    return shot_images.load_shot_array(
        folder,
        device,
        row,
        acq_timestamp=stamp,
        file_shot=shot_images.row_file_shot(frame, row),
    )


def tile_array(
    folder: Path, frame: pd.DataFrame, device: str, rows: tuple[int, ...]
) -> np.ndarray:
    """The tile's pixels: one shot's image, or its shots' average.

    Shots the device missed, or whose file is missing or unreadable, are
    left out of the average.

    Raises
    ------
    NoImage
        No shot of the tile has an image; the reason of the last one.
    """
    arrays = []
    reason = "no shots"
    for row in rows:
        found = row_array(folder, frame, device, row)
        if found.array is None:
            if found.kind in ("vendor", "unrenderable"):
                raise NoImage(found.reason or found.kind)  # the same for every shot
            reason = found.reason or found.kind
            continue
        arrays.append(found.array)
    if len(arrays) == 1:
        return arrays[0]
    averaged = average_frames(arrays, label=f"{device} rows {rows[0]}-{rows[-1]}")
    if averaged is None:
        raise NoImage(reason if not arrays else "the shots' images differ in size")
    return averaged


_kept: OrderedDict[tuple, np.ndarray] = OrderedDict()
_kept_lock = threading.Lock()


def kept_tile_array(
    folder: Path, frame: pd.DataFrame, device: str, rows: tuple[int, ...]
) -> np.ndarray:
    """:func:`tile_array`, remembered until the camera's folder changes.

    The key carries the device folder's modification time, so a file
    added while a scan runs is read; a refusal is never remembered.
    """
    try:
        stamp = (folder / device).stat().st_mtime_ns
    except OSError:
        return tile_array(folder, frame, device, rows)
    key = (str(folder), device, rows, len(frame), stamp)
    with _kept_lock:
        if key in _kept:
            _kept.move_to_end(key)
            return _kept[key]
    pixels = tile_array(folder, frame, device, rows)
    pixels.flags.writeable = False  # shared between requests: never edited
    with _kept_lock:
        _kept[key] = pixels
        while len(_kept) > _KEPT_TILES:
            _kept.popitem(last=False)
    return pixels


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
    saturates. Tiles are read a few at a time.

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
