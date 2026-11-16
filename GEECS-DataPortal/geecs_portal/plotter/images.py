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

Several scans (the set's extra scans) are a :class:`ScanSet`: their
s-files merged (``scan_merge.merge_scan_frames``), a row of the merged
frame a shot of one of them, and the bins the merged scans' one bin
number (``scan_merge.with_merged_bins``: each scan's bins by X, the same
setting pooled). Every loader here takes either one scan's folder with
its s-file, or a ScanSet with its merged frame.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from geecs_data_utils.data.binning import BinningConfig, bin_rows
from geecs_data_utils.data.row_filters import RowFilters, filter_mask
from geecs_data_utils.data.image_prep import HIT_FILTERS, ImagePrep, subtract_background
from geecs_data_utils.io import shot_images, shot_traces
from geecs_data_utils.io.array1d import Data1DResult
from geecs_data_utils.data.scan_merge import (
    MERGED_BIN_COLUMN,
    SCAN_BIN_COLUMN,
    merge_scan_frames,
    with_merged_bins,
)
from geecs_data_utils.io.display_png import window_levels
from geecs_data_utils.io.shot_images import ScanRows

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
#: GEECSplotter's LabVIEW tables (``io.colour_tables``), the perceptual
#: ones, then the classic false-colour tables.
CMAPS = (
    "gray",
    "enhance_low",
    "white_blue_yellow",
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


@dataclass(frozen=True)
class ScanPart:
    """One scan of a :class:`ScanSet`: its folder, its s-file and its names."""

    folder: Path
    day: date
    scan: int
    frame: pd.DataFrame = field(compare=False, hash=False, repr=False)
    #: ``"11-03 S005"``: the scan as a caption names it.
    label: str = field(default="", compare=False)
    #: The portal run's uid (the big view's processing preview opens it).
    uid: str = field(default="", compare=False)


@dataclass(frozen=True)
class ScanSet:
    """Several scans on one strip: their merged frame and its one bin number.

    A row of :attr:`frame` (1-based) is a shot of one part; :meth:`split`
    sends a tile's rows to their own scans. Identified by its parts (the
    cache keys on it).
    """

    parts: tuple[ScanPart, ...]
    #: ``merge_scan_frames`` over the parts, with :data:`MERGED_BIN_COLUMN`.
    frame: pd.DataFrame = field(compare=False, hash=False, repr=False)
    #: The column that bins the strip (the merged scans' one bin number).
    bin_col: str = MERGED_BIN_COLUMN
    #: Two scans' bins this close in X were pooled.
    tolerance: float = field(default=0.0, compare=False)

    @classmethod
    def merged(
        cls,
        parts: tuple[ScanPart, ...],
        x: Optional[str],
        tolerance: Optional[float] = None,
    ) -> "ScanSet":
        """The parts merged, their bins numbered by *x* and pooled within *tolerance*.

        ``None`` *tolerance* is half the smallest step of any one scan
        (``scan_merge.default_bin_tolerance``).
        """
        merged = merge_scan_frames({(p.day, p.scan): p.frame for p in parts})
        frame, tol = with_merged_bins(merged, x, tolerance)
        return cls(parts, frame, MERGED_BIN_COLUMN, tol)

    def _starts(self) -> list[int]:
        """Each part's first row in :attr:`frame`, less one."""
        starts, at = [], 0
        for part in self.parts:
            starts.append(at)
            at += len(part.frame)
        return starts

    def locate(self, row: int) -> tuple[ScanPart, int]:
        """The scan a merged *row* is a shot of, and its row there (both 1-based).

        Raises
        ------
        NoImage
            A row past the merged frame.
        """
        for part, start in zip(self.parts, self._starts()):
            if start < row <= start + len(part.frame):
                return part, row - start
        raise NoImage(f"no shot {row} in these scans")

    def split(self, rows: tuple[int, ...]) -> list[ScanRows]:
        """*rows* of the merged frame as each scan's own rows, in scan order."""
        found: dict[int, list[int]] = {}
        for row in rows:
            part, local = self.locate(row)
            found.setdefault(self.parts.index(part), []).append(local)
        return [
            ScanRows(self.parts[i].folder, self.parts[i].frame, tuple(found[i]))
            for i in sorted(found)
        ]

    def names(self) -> pd.Series:
        """Each merged row's scan, as captions name it (``"11-03 S005"``)."""
        labels = [p.label for p in self.parts for _ in range(len(p.frame))]
        return pd.Series(labels, index=self.frame.index)


#: Where a strip's shots come from: one scan's folder (with its s-file),
#: or several scans (:class:`ScanSet`, with its merged frame).
Source = Union[Path, ScanSet]


def _parts(
    source: Source, frame: pd.DataFrame, rows: tuple[int, ...]
) -> list[ScanRows]:
    """*rows* of *source* as each scan's folder, table and own rows."""
    if isinstance(source, ScanSet):
        return source.split(rows)
    return [ScanRows(source, frame, rows)]


def _source_key(source: Source) -> object:
    """What names *source* in the cache: its folder, or its scans' folders."""
    if isinstance(source, ScanSet):
        return tuple(str(p.folder) for p in source.parts)
    return str(source)


def _source_stamp(source: Source, device: str) -> object:
    """The device folders' modification times (``None``: the main one cannot be read).

    Another scan without the device has ``None`` in its place: its rows
    give no image, and a folder appearing there later changes the key.
    """
    if not isinstance(source, ScanSet):
        return _folder_stamp(source, device)
    stamps = tuple(_folder_stamp(p.folder, device) for p in source.parts)
    return stamps if stamps[0] is not None else None


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


def trace_device(
    folder: Path, device: str, devices: Optional[list[str]] = None
) -> shot_traces.TraceDevice:
    """How *device* records traces (``kind == ""``: it does not), remembered.

    ``geecs_data_utils.io.shot_traces.trace_device``: an array capture
    stack, per-shot scope files (one opened to name its channels), or
    per-shot x/y text columns (a spectrometer's wavelength and counts). The
    answer is kept until the device folder changes, so a page of tiles
    probes the share once.
    """
    stamp = _folder_stamp(folder, device)
    key = ("trace-device", str(folder), device, stamp)
    with _kept_lock:
        kept = _kept.get(key)
    if isinstance(kept, shot_traces.TraceDevice):
        return kept
    found = shot_traces.trace_device(folder, device, devices=devices)
    if stamp is not None:
        _keep(key, found)
    return found


def devices(folder: Path) -> list[str]:
    """The scan's device folders: one listing, no probe of each.

    Probing every folder for pictures costs a listing of each over the
    share (tens of seconds cold, for a scan with 70 devices), so the list
    is every folder and the picked one is checked
    (:func:`is_image_device`).
    """
    return shot_images.image_devices(folder)


def bins(
    frame: pd.DataFrame, bin_col: str = SCAN_BIN_COLUMN
) -> list[tuple[object, tuple[int, ...]]]:
    """Each bin's label and rows, in bin order (GEECSplotter's ``Bin #``).

    ``geecs_data_utils.data.binning.bin_rows`` over the whole s-file (its
    positions are the shot loaders' rows): the rows with an empty
    ``Bin #`` a bin of their own (last). A bin whose measurements are all
    missing is still a tile, though the binned plot drops it. Several
    scans bin by their merged frame's one number (*bin_col*,
    :attr:`ScanSet.bin_col`).

    Raises
    ------
    NoImage
        The scan has no ``Bin #`` column.
    """
    try:
        if bin_col == SCAN_BIN_COLUMN:
            return bin_rows(frame)
        return bin_rows(frame, BinningConfig(bin_col=bin_col))
    except KeyError:
        raise NoImage(f"this scan has no {bin_col} column") from None


def _parsed_filters(filters: str) -> RowFilters:
    try:
        return RowFilters.model_validate_json(filters)
    except ValueError as why:  # pydantic's ValidationError is one
        raise NoImage(f"the shot filters cannot be read: {why}") from None


def missing_filter_columns(frame: pd.DataFrame, filters: str) -> list[str]:
    """The columns the active shot filters name that *frame* lacks, in order.

    The Plot page filters several scans merged (``scan``, ``day``, columns
    only some have); the Images page shows one scan, and leaves those
    conditions out (:func:`shot_rows`) rather than show nothing.
    """
    if not filters:
        return []
    missing: list[str] = []
    for group in _parsed_filters(filters).active_groups():
        for c in group.conditions:
            if c.column not in frame.columns and c.column not in missing:
                missing.append(c.column)
    return missing


def shot_rows(frame: pd.DataFrame, filters: str) -> Optional[frozenset[int]]:
    """The rows (from 1) the page's shot filters keep; ``None`` when none is on.

    *filters* is the set's ``shot_filters`` JSON (``RowFilters``), the
    same the Plot page draws with, so both pages keep the same shots. A
    condition on a column this scan lacks (:func:`missing_filter_columns`)
    is left out; a group left with none is off.

    Raises
    ------
    NoImage
        Filters that cannot be read, or applied (a date column).
    """
    if not filters:
        return None
    parsed = _parsed_filters(filters)
    groups = [
        g.model_copy(
            update={
                "conditions": [c for c in g.conditions if c.column in frame.columns]
            }
        )
        for g in parsed.groups
    ]
    parsed = parsed.model_copy(update={"groups": groups})
    try:
        if not parsed.active_groups():
            return None
        mask = filter_mask(frame, parsed).to_numpy(dtype=bool)
    except ValueError as why:  # pydantic's ValidationError is one too
        raise NoImage(f"the shot filters cannot be applied here: {why}") from None
    return frozenset(int(i) + 1 for i in np.flatnonzero(mask))


def kept_bins(
    frame: pd.DataFrame,
    keep: Optional[frozenset[int]] = None,
    bin_col: str = SCAN_BIN_COLUMN,
) -> list[tuple[object, tuple[int, ...]]]:
    """:func:`bins` with only the rows *keep* holds; a bin left empty goes."""
    found = bins(frame, bin_col)
    if keep is None:
        return found
    out = []
    for label, rows in found:
        kept = tuple(r for r in rows if r in keep)
        if kept:
            out.append((label, kept))
    return out


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
    keep: Optional[frozenset[int]] = None,
    bin_col: str = SCAN_BIN_COLUMN,
    scan_names: Optional[pd.Series] = None,
) -> list[Tile]:
    """The strip's tiles, in order (only the shots *keep* holds, when given).

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
    keep : frozenset of int, optional
        The rows the shot filters keep (:func:`shot_rows`): a bin averages
        (or picks from) only those, a bin with none goes, and a shots
        strip shows only those. A bin tile's ``index`` is its place among
        the bins left.
    bin_col : str, optional
        The bin column: ``Bin #``, or several scans' one bin number
        (:attr:`ScanSet.bin_col`).
    scan_names : pandas.Series, optional
        Several scans: each row's scan (:meth:`ScanSet.names`); a shot's
        caption names it (``"11-03 S005 shot 3"``), a bin's how many
        scans it pools (``"17.5 · 2 scans · 10 shots"``).

    Returns
    -------
    list of Tile
    """

    def shot_label(row: int) -> str:
        shot = row if shot_numbers is None else shot_numbers.iloc[row - 1]
        named = f"shot {_number(shot, f'? (row {row})')}"
        return named if scan_names is None else f"{scan_names.iloc[row - 1]} {named}"

    def pooled(rows: tuple[int, ...]) -> str:
        if scan_names is None:
            return ""
        count = len({scan_names.iloc[r - 1] for r in rows})
        return (
            f"{count} scans · " if count > 1 else f"{scan_names.iloc[rows[0] - 1]} · "
        )

    if strip == "shots":
        out = []
        bin_numbers = frame.get(bin_col)
        for row in range(1, len(frame) + 1):
            if keep is not None and row not in keep:
                continue
            label = shot_label(row)
            in_bin = (
                "" if bin_numbers is None else _number(bin_numbers.iloc[row - 1], "")
            )
            value = _value(scan_values, (row,))
            caption = f"{label} · {value}" if value else label
            out.append(Tile(row, (row,), caption, in_bin))
        return out
    found = kept_bins(frame, keep, bin_col)
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
        count = f"{pooled(rows)}{len(rows)} shot{'s' if len(rows) != 1 else ''}"
        out.append(Tile(index, rows, f"{where} · {count}", _number(label, "")))
    return out


@dataclass(frozen=True)
class Background:
    """A background scan: each camera's average over all its shots there.

    Identified by its folder (the cache keys on it); the s-file rides
    along to name the shots.
    """

    folder: Path
    frame: pd.DataFrame = field(compare=False, hash=False, repr=False)
    label: str = field(default="", compare=False)


def _background_stamp(background: Optional[Background], device: str) -> object:
    """What makes a cached tile stale with its background (folder time, rows).

    A background scan still being written changes both.
    """
    if background is None:
        return None
    try:
        stamp = (background.folder / device).stat().st_mtime_ns
    except OSError:
        stamp = None
    return (stamp, len(background.frame))


def background_array(background: Background, device: str) -> np.ndarray:
    """*device*'s average over every shot of the background scan (remembered).

    Raises
    ------
    NoImage
        The background scan has no image of *device*.
    """
    rows = tuple(range(1, len(background.frame) + 1))
    try:
        return kept_tile_array(background.folder, background.frame, device, rows)
    except NoImage as why:
        raise NoImage(
            f"no background: {background.label or 'the background scan'} {why}"
        ) from None


def tile_array(
    folder: Source,
    frame: pd.DataFrame,
    device: str,
    rows: tuple[int, ...],
    prep: ImagePrep = ImagePrep(),
    background: Optional[Background] = None,
) -> np.ndarray:
    """The tile's pixels: one shot's image, or its shots' average.

    *background* comes off each shot first (the camera's average over the
    background scan, :func:`background_array`), then *prep* cleans it
    (``geecs_data_utils.data.image_prep``: X-ray hits out, then centred
    on its beam), so a bin's average is of subtracted, centred shots.

    ``geecs_data_utils.io.shot_images.average_scan_rows``: shots the
    device missed, or whose file is missing or unreadable, are left out of
    the average, never replaced by a neighbour's image. *folder* is one
    scan's (*frame* its s-file) or a :class:`ScanSet` (*frame* its merged
    frame), whose rows each come from their own scan.

    Raises
    ------
    NoImage
        No shot of the tile has an image; the reason why.
    """
    back = background_array(background, device) if background is not None else None

    def prepare(image: np.ndarray) -> np.ndarray:
        if back is not None:
            try:
                image = subtract_background(image, back)
            except ValueError as why:
                raise NoImage(str(why)) from None
        return prep.apply(image)

    found = shot_images.average_scan_rows(
        _parts(folder, frame, rows),
        device,
        prepare=prepare if (prep.active or back is not None) else None,
    )
    if found.array is None:
        raise NoImage(found.reason or found.kind)
    return found.array


_kept: OrderedDict[tuple, object] = OrderedDict()
_kept_bytes = 0
_kept_lock = threading.Lock()


def _size(value: object) -> int:
    """The bytes a kept value holds: pixels, a trace's samples, or a reason (0)."""
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Data1DResult):
        return value.data.nbytes
    return 0


def _remembered(key: tuple, compute: Callable[[], object]) -> object:
    """*compute*'s value, kept under *key* (a :class:`NoImage` kept as its reason).

    A refusal is kept for ``_REFUSAL_SECONDS`` only, so a passing failure
    (a share blip, a file being written) is looked at again; a value until
    the store, at most ``_KEPT_BYTES``, needs its room (oldest out first).

    Raises
    ------
    NoImage
        From *compute*, or remembered.
    """
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
        value = compute()
    except NoImage as why:
        _keep(key, (str(why), now))
        raise
    _keep(key, value)
    return value


def _folder_stamp(folder: Path, device: str) -> Optional[int]:
    """The device folder's modification time (``None``: it cannot be read)."""
    try:
        return (folder / device).stat().st_mtime_ns
    except OSError:
        return None


def kept_tile_array(
    folder: Source,
    frame: pd.DataFrame,
    device: str,
    rows: tuple[int, ...],
    prep: ImagePrep = ImagePrep(),
    background: Optional[Background] = None,
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
    stamp = _source_stamp(folder, device)
    if stamp is None:
        return tile_array(folder, frame, device, rows, prep, background)
    key = (
        _source_key(folder),
        device,
        rows,
        len(frame),
        stamp,
        prep,
        background,
        _background_stamp(background, device),
    )

    def compute() -> np.ndarray:
        pixels = tile_array(folder, frame, device, rows, prep, background)
        pixels.flags.writeable = False  # shared between requests: never edited
        return pixels

    found = _remembered(key, compute)
    assert isinstance(found, np.ndarray)
    return found


def _keep(key: tuple, value: object) -> None:
    """Store *value*, dropping the oldest until the store fits its bytes."""
    global _kept_bytes
    size = _size(value)
    if size > _KEPT_BYTES:
        return  # one tile bigger than the whole store: never kept
    with _kept_lock:
        if key in _kept:
            return
        _kept[key] = value
        _kept_bytes += size
        while _kept_bytes > _KEPT_BYTES:
            _, old = _kept.popitem(last=False)
            _kept_bytes -= _size(old)


def trace_tile(
    folder: Source,
    frame: pd.DataFrame,
    device: str,
    rows: tuple[int, ...],
    channel: int = 0,
) -> Data1DResult:
    """The tile's trace: one shot's, or its shots' sample-by-sample mean.

    ``geecs_data_utils.io.shot_traces.average_trace_scan_rows``: shots the
    device missed, or whose file is missing or unreadable, are left out;
    a bin whose shots' x axes differ has no mean (nothing is resampled).
    Remembered as :func:`kept_tile_array` is, until the folder changes.
    *folder* is one scan's, or a :class:`ScanSet`.

    Raises
    ------
    NoImage
        No shot of the tile has a trace, or the axes differ; the reason.
    """
    parts = _parts(folder, frame, rows)
    probes = [trace_device(p.scan_folder, device) for p in parts]

    def compute() -> Data1DResult:
        found = shot_traces.average_trace_scan_rows(parts, device, channel, probes)
        if found.result is None:
            raise NoImage(found.reason or found.kind)
        found.result.data.flags.writeable = False  # shared between requests
        return found.result

    stamp = _source_stamp(folder, device)
    if stamp is None:
        return compute()
    key = ("trace", _source_key(folder), device, rows, len(frame), stamp, channel)
    found = _remembered(key, compute)
    assert isinstance(found, Data1DResult)
    return found


def trace_levels(
    folder: Source,
    frame: pd.DataFrame,
    device: str,
    tiles: list[Tile],
    channel: int = 0,
) -> tuple[float, float, int]:
    """One y range for every trace tile: the lowest and highest sample of them all.

    The trace row's "shared" scale, as :func:`shared_levels` is a camera's:
    every tile given (the page's shown ones) is read first, a few at a time.

    Returns
    -------
    tuple
        ``(low, high, tiles)``: the range and how many tiles had a trace.

    Raises
    ------
    NoImage
        No tile has a trace.
    """

    def levels(tile: Tile) -> Optional[tuple[float, float]]:
        try:
            y = trace_tile(folder, frame, device, tile.rows, channel).data[:, 1]
        except NoImage:
            return None
        finite = y[np.isfinite(y)]
        return (float(finite.min()), float(finite.max())) if finite.size else None

    with ThreadPoolExecutor(max_workers=8) as pool:
        found = [w for w in pool.map(levels, tiles) if w is not None]
    if not found:
        raise NoImage("no tile has a trace")
    return min(lo for lo, _ in found), max(hi for _, hi in found), len(found)


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
    folder: Source,
    frame: pd.DataFrame,
    device: str,
    tiles: list[Tile],
    roi: Optional[Roi] = None,
    prep: ImagePrep = ImagePrep(),
    background: Optional[Background] = None,
) -> tuple[float, float, int]:
    """One black and one white for *tiles*: the darkest and brightest pixel.

    Over each tile's ROI; the brightest tile sets white, so none of them
    saturates. Reads every tile given, a few at a time: the page gives the
    tiles it shows (the author, 2026-11-03: a row spread over the scan
    reads its few, not the whole scan; a hidden tile stepped to in the big
    view may clip), and every tile of a grid; it says so while it runs.

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
            pixels = crop(
                kept_tile_array(folder, frame, device, tile.rows, prep, background), roi
            )
        except NoImage:
            return None
        return window_levels(pixels, 0.0, 100.0)  # its darkest and brightest

    with ThreadPoolExecutor(max_workers=8) as pool:
        found = [w for w in pool.map(levels, tiles) if w is not None]
    if not found:
        raise NoImage("no tile has an image")
    return min(lo for lo, _ in found), max(hi for _, hi in found), len(found)
