"""Join a run's per-frame stream columns onto its shot rows.

A strict run records every essential device in the ``primary`` event rows,
so its s-file is those rows renamed (:mod:`geecs_data_utils.tiled_export`).
Two shapes of the native-Bluesky scanner put per-shot values *outside* the
rows instead (``Planning/native_bluesky/08_gated_batch.md`` §4.5):

- a **gated** run's frames and their per-frame scalars live in each
  plugin-backed camera's stack, referenced by a datum-only stream with no
  events at all; its rows are the per-shot sampler's ``shots`` events;
- a **non-essential** camera streams for a whole run into its own
  ``<name>_stream``, and its frame for shot *k* may arrive during *k+1*.

Both are joined to the rows by the one shot identity GEECS has, the
device's ``acq_timestamp``.  Cross-device stamps of one shot differ by a
per-device constant — the camera's drain offset
(``03_clean_room_rebuild.md`` §11.3/§11.4) — so the join corrects each
side by its own offset and matches each row to the **nearest** frame
inside that row's own window.

The window is **per row**, not per run (:func:`row_windows`): half the
shot period, narrowed to half the distance to that row's closest
neighbour, and half-open so a frame exactly on the boundary two rows share
belongs to one of them.  So no frame can ever be claimed by two rows, a
faster rep rate tightens the window by itself, and one anomalous pair of
row stamps tightens only those two rows instead of the whole run.

A frame no row claims is an **orphan**: the extra edge at a step's end, a
frame taken during an interrupted step, a non-essential camera's frame for
a shot nobody recorded.  It stays in the stack and in Tiled and is left out
of the s-file (Sam, 2026-09-12, ``08`` §6 Q4): one s-file row per
essential shot, always.

Pure arithmetic over arrays — no I/O, no Bluesky, no pandas — so both the
worker's live callback (reading the stacks off the share at the stop
document) and the offline Tiled re-export share one join.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: The laser rep rate the windows assume when a run does not say (HTU:
#: 1 Hz).  Only the *window* depends on it, and a run's own row spacing
#: narrows each window further — see :func:`row_windows`.
DEFAULT_SHOT_PERIOD_S = 1.0

#: The suffix of a device's stamp column in an event row.
ACQ_TIMESTAMP_SUFFIX = "-acq_timestamp"

#: The stream a gated run's per-shot rows live in — the per-shot sampler's
#: events.  Named here because it is a *document* contract: the worker's
#: plan writes it, the worker's s-file callback reads it, and the offline
#: re-export reads it back out of Tiled.
SHOTS_STREAM = "shots"

#: The stack attribute suffix that carries a frame's GEECS acquisition
#: stamp (``geecs_data_utils.io.scan_stack.TIMESTAMP_SUFFIX``); as an
#: s-file column it is spelled like the strict row's, ``<device> acq_timestamp``.
FRAME_STAMP_SUFFIX = "frame_acq_timestamp"

#: Row stamps closer together than this are one row's publish race, not two
#: shots, and are ignored when sizing a window.
_SAME_SHOT_S = 1e-3


@dataclass(frozen=True)
class FrameColumns:
    """The per-frame columns of one datum-only stream source, aligned by frame.

    Attributes
    ----------
    object_name :
        The ophyd device name the stream's data keys are built from
        (``uc_amp4_ir_input``) — the event-key prefix of every column, and
        the key its drain offset is looked up under.
    stamps :
        ``(N,)`` per-frame ``acq_timestamp``, **LabVIEW epoch** (the rows'
        and the s-file's convention); ``NaN`` where a frame has none.
    columns :
        Event key → ``(N,)`` values, the spelling a strict row would use
        for the same quantity (``uc_cam-meancounts``), so one header map
        renames both.
    raw_names :
        Event key → the raw GEECS variable behind it, where the source
        knows it (the stack's own ``scalar_variables`` manifest).  Used
        only to name a column in a message: normalization is one-way, so
        without this a column no header claims cannot be reported in the
        vocabulary an operator recognises.
    """

    object_name: str
    stamps: np.ndarray
    columns: Mapping[str, np.ndarray] = field(default_factory=dict)
    raw_names: Mapping[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        """The number of frames."""
        return int(np.size(self.stamps))

    def truncated(self, frames: int) -> "FrameColumns":
        """Return this source with only its first *frames* frames.

        The worker knows how many frames a stream's datums referenced;
        anything past that is in the file but not in the documents (a
        non-essential camera keeps writing between its ``collect`` and its
        ``unstage``), and joining it would put a value in the s-file for a
        frame the catalog does not have.
        """
        if frames < 0 or frames >= len(self):
            return self
        return FrameColumns(
            object_name=self.object_name,
            stamps=self.stamps[:frames],
            columns={key: values[:frames] for key, values in self.columns.items()},
            raw_names=dict(self.raw_names),
        )


@dataclass(frozen=True)
class ShotJoin:
    """Which frame belongs to which shot row, and which frames belong to none.

    Attributes
    ----------
    frame_for_shot :
        One entry per shot row: the frame index, or ``None`` for a shot
        this device has no frame for.
    orphans :
        Frame indices no row claimed — kept in the stack, left out of the
        s-file.
    contested :
        Frame indices that fell inside a row's window but lost it to a
        nearer frame — a warning, not an error.
    """

    frame_for_shot: tuple[int | None, ...]
    orphans: tuple[int, ...] = ()
    contested: tuple[int, ...] = ()

    @property
    def matched(self) -> int:
        """How many shot rows got a frame."""
        return sum(1 for index in self.frame_for_shot if index is not None)


def shot_clock_column(
    start_doc: Mapping[str, Any], columns: Sequence[str]
) -> str | None:
    """The stamp column of *columns* that identifies the shot of each row.

    The gated sampler writes the clock device's stamp into every ``shots``
    row as the shot id itself (every other stamp in the row is only "the
    latest value at the tick"), and the run's start document names it.  In
    order:

    1. ``shot_clock_column``, the column itself — what the plan records
       from GeecsBluesky 0.85;
    2. ``shot_clock``, the GEECS *device* name, matched to a column by
       device name ignoring case and separators (the runs recorded before
       the plan knew to write the column — the mangling that produced the
       column lives in ``geecs_core.pv_naming`` and is never reimplemented
       here, so this is a match, not a derivation);
    3. the first ``detectors`` entry that has a stamp column — a strict
       run's rows carry one per essential device, and any of them is a
       valid clock because the join corrects both sides by their own
       offsets;
    4. the first stamp column at all.

    Parameters
    ----------
    start_doc :
        The run's start document (``shot_clock_column``, ``shot_clock``,
        ``detectors``).
    columns :
        The row columns available.

    Returns
    -------
    str or None
        The column name, or ``None`` when no row column is a stamp.
    """
    stamps = [c for c in columns if c.endswith(ACQ_TIMESTAMP_SUFFIX)]
    if not stamps:
        return None
    named = start_doc.get("shot_clock_column")
    if named:
        if str(named) in stamps:
            return str(named)
        logger.warning(
            "shot_clock_column %r is not a row column; falling back", str(named)
        )
    clock = start_doc.get("shot_clock")
    if clock:
        wanted = _squash(str(clock))
        for column in stamps:
            if _squash(column[: -len(ACQ_TIMESTAMP_SUFFIX)]) == wanted:
                return column
        logger.warning(
            "shot_clock %r has no stamp column among %s; falling back", clock, stamps
        )
    for detector in start_doc.get("detectors") or ():
        candidate = f"{detector}{ACQ_TIMESTAMP_SUFFIX}"
        if candidate in stamps:
            return candidate
    return stamps[0]


def clock_device(column: str) -> str:
    """The object name behind a stamp *column* (``uc_a-acq_timestamp`` → ``uc_a``)."""
    if column.endswith(ACQ_TIMESTAMP_SUFFIX):
        return column[: -len(ACQ_TIMESTAMP_SUFFIX)]
    return column


def _squash(name: str) -> str:
    """Lowercase *name* with every non-alphanumeric character removed.

    A comparison key only: it matches a GEECS device name against the
    event-column component derived from it without reproducing the
    derivation (``normalize_component``, which this package cannot
    import).  Two device names differing only in separators squash alike,
    so the first matching column wins; a run recorded by GeecsBluesky
    >= 0.85 carries ``shot_clock_column`` and never reaches here.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def row_windows(
    shot_stamps: np.ndarray, period: float = DEFAULT_SHOT_PERIOD_S
) -> np.ndarray:
    """Half-width of the window each row may claim a frame inside, seconds.

    Half the shot *period*, narrowed for a row to half the distance to its
    closest neighbouring row — so two rows can never both reach one frame
    whatever the rep rate, a run faster than *period* tightens its windows
    by itself, and **one** anomalous pair of row stamps tightens only those
    two rows rather than the whole run.  Neighbours closer than a
    millisecond are one row's publish race, not a second shot, and are
    skipped when measuring.

    Parameters
    ----------
    shot_stamps :
        The rows' clock stamps, seconds; a non-finite stamp gets the
        period's half-window (it will match nothing anyway).
    period :
        The trigger period; pass a run's own ``shot_period`` when it has one.

    Returns
    -------
    numpy.ndarray
        ``(M,)`` half-windows, one per row, in row order.
    """
    stamps = np.asarray(shot_stamps, dtype=float)
    half = float(period) / 2.0
    windows = np.full(stamps.shape, half, dtype=float)
    finite = [i for i in np.argsort(stamps, kind="stable") if np.isfinite(stamps[i])]
    for position, row in enumerate(finite):
        gap = np.inf
        for step in (-1, 1):
            probe = position + step
            while 0 <= probe < len(finite):
                distance = abs(float(stamps[finite[probe]]) - float(stamps[row]))
                if distance >= _SAME_SHOT_S:
                    gap = min(gap, distance)
                    break
                probe += step
        if np.isfinite(gap):
            windows[row] = min(half, gap / 2.0)
    return windows


def join_frames_to_shots(
    shot_stamps: np.ndarray,
    frame_stamps: np.ndarray,
    *,
    windows: np.ndarray,
    shot_offset: float = 0.0,
    frame_offset: float = 0.0,
) -> ShotJoin:
    """Give each row the nearest frame inside its own window.

    Both sides are corrected by their device's drain offset first (``03``
    §11.4: the stamp is the trigger's arrival plus a per-device constant),
    so after the correction one shot's stamps land within NTP jitter of
    each other.  Row-centric and nearest-wins: when two frames fall in one
    row's window the **closer** one takes the row and the other is
    reported as contested, never the one that happens to come first in the
    file.

    Parameters
    ----------
    shot_stamps :
        ``(M,)`` clock stamps of the rows, in row order.
    frame_stamps :
        ``(N,)`` per-frame stamps, in frame order, same epoch.
    windows :
        ``(M,)`` per-row half-windows (:func:`row_windows`).
    shot_offset, frame_offset :
        The clock device's and the frame device's drain offsets, seconds.

    Returns
    -------
    ShotJoin
        The per-row frame indices, the orphan frames and the contested ones.
    """
    shots = np.asarray(shot_stamps, dtype=float) - float(shot_offset)
    frames = np.asarray(frame_stamps, dtype=float) - float(frame_offset)
    half = np.asarray(windows, dtype=float)
    if half.size != shots.size:
        raise ValueError(
            f"windows has {half.size} entry/entries for {shots.size} row(s)"
        )
    finite = np.array(
        [i for i in np.argsort(frames, kind="stable") if np.isfinite(frames[i])],
        dtype=int,
    )
    sorted_frames = frames[finite] if finite.size else np.empty(0)
    owner: dict[int, int] = {}
    losers: set[int] = set()
    for row in range(shots.size):
        if not np.isfinite(shots[row]) or not sorted_frames.size:
            continue
        low = int(np.searchsorted(sorted_frames, shots[row] - half[row], "left"))
        # "left" on the upper bound too: the window is half-open, so a frame
        # exactly on the boundary two rows share goes to the later row only.
        high = int(np.searchsorted(sorted_frames, shots[row] + half[row], "left"))
        best: int | None = None
        best_delta = float("inf")
        for position in range(low, high):
            frame = int(finite[position])
            delta = abs(float(sorted_frames[position]) - float(shots[row]))
            if delta < best_delta:
                if best is not None:
                    losers.add(best)
                best, best_delta = frame, delta
            else:
                losers.add(frame)
        if best is not None:
            owner[row] = best
            losers.discard(best)
    claimed = set(owner.values())
    return ShotJoin(
        frame_for_shot=tuple(owner.get(row) for row in range(shots.size)),
        orphans=tuple(
            frame
            for frame in range(frames.size)
            if frame not in claimed and frame not in losers
        ),
        contested=tuple(sorted(losers)),
    )


def frame_columns_from_attributes(
    object_name: str,
    attributes: Mapping[str, np.ndarray],
    *,
    variables: Mapping[str, str] | None = None,
    labview_epoch_offset: float = 0.0,
) -> FrameColumns | None:
    """Build a :class:`FrameColumns` from a stack's per-frame attribute datasets.

    The plugin names every attribute ``<device>-hdf-<variable>-<suffix>``
    (``io.scan_stack.parse_attribute_name``).  The frame stamp
    (``frame_acq_timestamp``) becomes the stamp array *and* the
    ``<device>-acq_timestamp`` column — the spelling a strict row uses for
    the same quantity, so the one header map renames it.  Every other
    suffix becomes ``<device>-<suffix>``, which is how that subscribed
    scalar is spelled in a strict row; a suffix no header names (the
    plugin's ``frame_recv_timestamp``) is simply never emitted downstream.

    A column that is not the stamps' own length is dropped with a warning
    rather than padded: an attribute dataset shorter than the frames is a
    structural defect (a truncated rewind, a half-flushed file), and
    silently filling the tail with ``NaN`` would hide it.

    Parameters
    ----------
    object_name :
        The stream's ophyd device name, used when an attribute name does
        not carry one (a pre-0.8 stack's bare ``acq_timestamp``).
    attributes :
        Dataset name → ``(N,)`` values
        (``io.scan_stack.read_stack_attributes``).
    variables :
        Dataset name → the raw GEECS variable behind it
        (``io.scan_stack.stack_scalar_variables``), when the caller has the
        file's manifest; carried through for messages only.
    labview_epoch_offset :
        Added to the stamps — the stacks store Unix seconds and the rows
        carry LabVIEW seconds, so pass
        ``io.scan_stack.LABVIEW_EPOCH_OFFSET`` for a stack read off disk.

    Returns
    -------
    FrameColumns or None
        ``None`` when no attribute carries a frame stamp (nothing to join on).
    """
    from geecs_data_utils.io.scan_stack import parse_attribute_name

    manifest = dict(variables or {})
    stamps: np.ndarray | None = None
    columns: dict[str, np.ndarray] = {}
    raw_names: dict[str, str] = {}
    for name, values in attributes.items():
        parsed = parse_attribute_name(name)
        device = parsed[0] if parsed else object_name
        suffix = parsed[2] if parsed else name
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            logger.warning(
                "%s: attribute %r is not numeric; not joined", object_name, name
            )
            continue
        if suffix in (FRAME_STAMP_SUFFIX, "acq_timestamp"):
            stamps = array + float(labview_epoch_offset)
            columns[f"{device}{ACQ_TIMESTAMP_SUFFIX}"] = stamps
        else:
            key = f"{device}-{suffix}"
            columns[key] = array
            if name in manifest:
                raw_names[key] = str(manifest[name])
    if stamps is None:
        logger.warning(
            "%s: no frame stamp among the stack's attributes (%s); not joined",
            object_name,
            ", ".join(sorted(attributes)) or "none",
        )
        return None
    ragged = [
        key for key, values in columns.items() if np.size(values) != np.size(stamps)
    ]
    for key in ragged:
        logger.warning(
            "%s: attribute column %r has %d value(s) for %d frame(s); dropped",
            object_name,
            key,
            np.size(columns[key]),
            np.size(stamps),
        )
        del columns[key]
    return FrameColumns(
        object_name=object_name,
        stamps=stamps,
        columns=columns,
        raw_names=raw_names,
    )


__all__ = [
    "ACQ_TIMESTAMP_SUFFIX",
    "DEFAULT_SHOT_PERIOD_S",
    "FRAME_STAMP_SUFFIX",
    "SHOTS_STREAM",
    "FrameColumns",
    "ShotJoin",
    "clock_device",
    "frame_columns_from_attributes",
    "join_frames_to_shots",
    "row_windows",
    "shot_clock_column",
]
