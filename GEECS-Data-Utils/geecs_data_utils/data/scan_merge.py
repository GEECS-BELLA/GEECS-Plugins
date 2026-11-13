"""Several scans' per-shot tables as one: GEECSplotter's "merge s files".

:func:`merge_scan_frames` stacks one frame per scan, in the order given,
and tags each row with the scan it came from (:data:`SCAN_COLUMN`,
:data:`DAY_COLUMN`). The columns are the union: a column only some scans
have is NaN on the others' rows. Nothing is renumbered: each shot keeps
its own ``Shotnumber`` and ``Bin #``, so a shot is named by its scan and
its number together, and ``Bin #`` means nothing across scans.
:func:`merged_bin_numbers` gives the merged table one bin number: each
scan's bins placed at their median X and pooled where two scans visited
the same setting (:func:`with_merged_bins` adds it as
:data:`MERGED_BIN_COLUMN`).

Pure: frames in, a frame out; no file is read or written.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: A scan as the merge names it: its day and its number.
ScanRef = tuple[date, int]

#: The scan number of each merged row.
SCAN_COLUMN = "scan"
#: The day of each merged row, as ISO text (``"2026-10-26"``).
DAY_COLUMN = "day"


def merge_scan_frames(frames: Mapping[ScanRef, "pd.DataFrame"]) -> "pd.DataFrame":
    """Stack the scans' frames into one, each row tagged with its scan.

    Parameters
    ----------
    frames : Mapping[tuple[datetime.date, int], pandas.DataFrame]
        ``(day, scan number) -> that scan's per-shot table``, in the order
        the rows are wanted.

    Returns
    -------
    pandas.DataFrame
        Every row of every frame, in order, on a fresh ``0..n-1`` index
        (labels are unique across scans). :data:`SCAN_COLUMN` (int) and
        :data:`DAY_COLUMN` (ISO text) come first, then the union of the
        frames' columns in the order first seen; a column a scan lacks is
        NaN on its rows.

        A frame's own ``scan`` or ``day`` column (the Bluesky s-files
        carry ``scan``, the scan number on every row) is replaced by the
        tag when every value it has names the same scan.

    Raises
    ------
    ValueError
        No frames, a frame with multi-level columns, or one whose own
        ``scan`` or ``day`` column names another scan (the tag would
        overwrite it).

    Notes
    -----
    A named index (``Shotnumber``, as ``ScanData``'s TDMS frame has it)
    becomes a column first, so the shot numbers survive the fresh index.
    """
    import pandas as pd

    if not frames:
        raise ValueError("merge at least one scan")
    parts = []
    for (day, scan), frame in frames.items():
        if isinstance(frame.columns, pd.MultiIndex):
            raise ValueError(
                f"Scan{scan:03d} on {day.isoformat()} has multi-level columns; "
                "flatten them first (data.columns.flatten_columns)"
            )
        if any(name is not None for name in frame.index.names):
            frame = frame.reset_index()
        tag = {SCAN_COLUMN: int(scan), DAY_COLUMN: day.isoformat()}
        for column, value in tag.items():
            if column in frame.columns and not _names(frame[column], value):
                raise ValueError(
                    f"Scan{scan:03d} on {day.isoformat()} has a column {column!r} "
                    f"that is not {value!r} on every row; the merge cannot tag its rows"
                )
        tags = pd.DataFrame(tag, index=frame.index)
        own = frame.drop(columns=[c for c in tag if c in frame.columns])
        parts.append(pd.concat([tags, own], axis=1))
    return pd.concat(parts, ignore_index=True, sort=False)


def _names(values: "pd.Series", tag: object) -> bool:
    """Whether every value *values* has is *tag* (a number compares as one)."""
    import pandas as pd

    present = values.dropna()
    if isinstance(tag, int):
        numbers = pd.to_numeric(present, errors="coerce")
        return bool(numbers.notna().all() and (numbers == tag).all())
    return bool((present.astype(str) == tag).all())


#: The in-memory column :func:`with_merged_bins` adds: one bin number
#: across the merged scans. ``Bin #`` itself is left as each scan wrote it.
MERGED_BIN_COLUMN = "Bin # (merged)"

#: The per-scan bin column the merged numbers are built from.
SCAN_BIN_COLUMN = "Bin #"


@dataclass(frozen=True)
class MergedBins:
    """The merged scans' bin numbers and the tolerance that pooled them."""

    #: One number per row of the merged frame, from 1 in X order; a group
    #: with no X value is a bin of its own, after the placed ones, in scan
    #: and bin order (never pooled: nothing says two of them are one setting).
    numbers: "pd.Series"
    #: Groups whose median X differed by at most this were pooled.
    tolerance: float


def default_bin_tolerance(frame: "pd.DataFrame", x: str) -> float:
    """Half the smallest step between bins within any one scan; 0 with none.

    Each scan's bins (``Bin #``) have a median X; the smallest positive
    gap between neighbouring medians of one scan is its step. Half of the
    smallest step over every scan pools the same setting visited by two
    scans and never two settings of one scan.
    """
    keys, medians = _group_medians(frame, x)
    by_scan: dict[tuple, list[float]] = {}
    for key, median in zip(keys, medians):
        if np.isfinite(median):
            by_scan.setdefault(key[:2], []).append(median)
    steps = []
    for values in by_scan.values():
        gaps = np.diff(np.sort(values))
        gaps = gaps[gaps > 0]
        if gaps.size:
            steps.append(float(gaps.min()))
    return min(steps) / 2 if steps else 0.0


def merged_bin_numbers(
    frame: "pd.DataFrame", x: Optional[str], tolerance: Optional[float] = None
) -> MergedBins:
    """One bin number across the merged scans, ordered by X.

    Each scan's bins stay together (its ``Bin #``: the scanner's own step,
    robust to a readback that wobbles from shot to shot); each (scan, bin)
    group is placed at its median *x*. The groups are sorted by that
    median, and a group within *tolerance* of the first group of the bin
    being built joins it, so the same setting from two scans is one bin.
    A scan with no ``Bin #`` is one group.

    Parameters
    ----------
    frame : pandas.DataFrame
        :func:`merge_scan_frames`' result (its ``scan`` and ``day`` tags).
    x : str or None
        The column that orders the bins (the scan variable). ``None`` (or
        a column the frame lacks) numbers the groups in scan and bin
        order, pooling none.
    tolerance : float, optional
        How far apart two groups' medians may be and still pool;
        ``None`` is :func:`default_bin_tolerance`.

    Returns
    -------
    MergedBins

    Raises
    ------
    ValueError
        A negative or non-finite *tolerance*.
    """
    import pandas as pd

    if tolerance is not None and not (np.isfinite(tolerance) and tolerance >= 0):
        raise ValueError(f"tolerance={tolerance!r}: a number, 0 or more")
    rows = _row_keys(frame)
    if not x or x not in frame.columns:
        order = {key: i + 1 for i, key in enumerate(dict.fromkeys(rows))}
        numbers = pd.Series([float(order[k]) for k in rows], index=frame.index)
        return MergedBins(numbers, 0.0)
    keys, medians = _group_medians(frame, x)
    tol = default_bin_tolerance(frame, x) if tolerance is None else float(tolerance)
    number: dict[tuple, float] = {}
    current, anchor = 0, None
    placed = sorted(
        (m, i) for i, m in enumerate(medians) if np.isfinite(m)
    )  # by median, ties in first-seen order
    for median, i in placed:
        if anchor is None or median - anchor > tol:
            current, anchor = current + 1, median
        number[keys[i]] = float(current)
    for i, median in enumerate(medians):
        if not np.isfinite(median):  # no X: its own bin, after the placed ones
            current += 1
            number[keys[i]] = float(current)
    values = [number[k] for k in rows]
    return MergedBins(pd.Series(values, index=frame.index), tol)


def with_merged_bins(
    frame: "pd.DataFrame", x: Optional[str], tolerance: Optional[float] = None
) -> tuple["pd.DataFrame", float]:
    """*frame* with :data:`MERGED_BIN_COLUMN` added, and the tolerance used.

    A copy; the scans' own ``Bin #`` is left as it was.
    """
    found = merged_bin_numbers(frame, x, tolerance)
    out = frame.copy()
    out[MERGED_BIN_COLUMN] = found.numbers
    return out, found.tolerance


def _column(frame: "pd.DataFrame", name: str, missing: object) -> list:
    """*frame*'s column *name* as a list, or *missing* on every row."""
    if name in frame.columns:
        return list(frame[name])
    return [missing] * len(frame)


def _row_keys(frame: "pd.DataFrame") -> list[tuple[str, int, float]]:
    """Each row's (day, scan, bin) group; a missing bin is the scan's one group (-inf)."""
    import pandas as pd

    bins = (
        pd.to_numeric(frame[SCAN_BIN_COLUMN], errors="coerce").tolist()
        if SCAN_BIN_COLUMN in frame.columns
        else [np.nan] * len(frame)
    )
    return [
        (str(d), int(n), float(b) if b == b else -np.inf)
        for d, n, b in zip(
            _column(frame, DAY_COLUMN, ""), _column(frame, SCAN_COLUMN, 0), bins
        )
    ]


def _group_medians(frame: "pd.DataFrame", x: str) -> tuple[list[tuple], list[float]]:
    """The (day, scan, bin) groups in first-seen order and each one's median *x*."""
    import pandas as pd

    rows = _row_keys(frame)
    values = (
        pd.to_numeric(frame[x], errors="coerce").to_numpy(dtype=float)
        if x in frame.columns
        else np.full(len(frame), np.nan)
    )
    members: dict[tuple, list[float]] = {}
    for key, value in zip(rows, values):
        members.setdefault(key, []).append(value)
    keys = list(members)
    medians = []
    for key in keys:
        finite = [v for v in members[key] if np.isfinite(v)]
        medians.append(float(np.median(finite)) if finite else np.nan)
    return keys, medians
