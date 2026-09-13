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
side by its own offset and matches a frame to the **nearest** shot within
a window that can never reach a neighbouring shot (half the shot period,
and never more than half the closest gap between two rows).

A frame with no shot inside its window is an **orphan**: the extra edge at
a step's end, a frame taken during an interrupted step, a non-essential
camera's frame for a shot nobody recorded.  It stays in the stack and in
Tiled and is left out of the s-file (Sam, 2026-09-12, ``08`` §6 Q4): one
s-file row per essential shot, always.

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

#: The laser rep rate the join's window assumes when a run does not say
#: (HTU: 1 Hz).  Only the *window* depends on it, and a run's own row
#: spacing narrows that window further — see :func:`join_window`.
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


@dataclass(frozen=True)
class FrameColumns:
    """The per-frame columns of one datum-only stream source, aligned by frame.

    Attributes
    ----------
    object_name :
        The ophyd device name the stream's data keys are built from
        (``uc_amp4_ir_input``) — the event-key prefix of every column.
    stamps :
        ``(N,)`` per-frame ``acq_timestamp``, **LabVIEW epoch** (the rows'
        and the s-file's convention); ``NaN`` where a frame has none.
    columns :
        Event key → ``(N,)`` values, the spelling a strict row would use
        for the same quantity (``uc_cam-meancounts``), so one header map
        renames both.
    drain_offset :
        Seconds between the trigger's arrival and this device's stamp
        (``03`` §11.4) — subtracted before the join.
    """

    object_name: str
    stamps: np.ndarray
    columns: Mapping[str, np.ndarray] = field(default_factory=dict)
    drain_offset: float = 0.0

    def __len__(self) -> int:
        """The number of frames."""
        return int(np.size(self.stamps))


@dataclass(frozen=True)
class ShotJoin:
    """Which frame belongs to which shot row, and which frames belong to none.

    Attributes
    ----------
    frame_for_shot :
        One entry per shot row: the frame index, or ``None`` for a shot
        this device has no frame for.
    orphans :
        Frame indices with no shot row inside the window — kept in the
        stack, left out of the s-file.
    contested :
        Frame indices that matched a row an earlier frame already owns
        (keep-first, the repo's duplicate rule) — a warning, not an error.
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
    latest value at the tick"), and the run's start document names that
    device in ``shot_clock``; the column is matched to it by device name,
    ignoring case and separators — the mangling that produced the column
    lives in ``geecs_core.pv_naming`` and is never reimplemented here.
    Falling back, in order: the first ``detectors`` entry that has a stamp
    column (a strict run's rows carry one per essential device, and any of
    them is a valid clock because the join corrects both sides by their
    own offsets), then the first stamp column at all.

    Parameters
    ----------
    start_doc :
        The run's start document (``shot_clock``, ``detectors``).
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


def _squash(name: str) -> str:
    """Lowercase *name* with every non-alphanumeric character removed.

    A comparison key only: it matches a GEECS device name against the
    event-column component derived from it without reproducing the
    derivation (``normalize_component``, which this package cannot import).
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def join_window(
    shot_stamps: np.ndarray, period: float = DEFAULT_SHOT_PERIOD_S
) -> float:
    """Half-width of the window a frame may match a shot row inside, seconds.

    Half the shot *period*, and never more than half the closest gap
    between two rows — so two rows can never contend for one frame
    whatever the rep rate, and a run faster than *period* narrows the
    window by itself.  Gaps below a millisecond are ignored (a stamp
    repeated inside one row's publish race, not a second shot).

    Parameters
    ----------
    shot_stamps :
        The rows' clock stamps, seconds.
    period :
        The trigger period; pass a run's own ``shot_period`` when it has one.

    Returns
    -------
    float
        The half-window, seconds.
    """
    stamps = np.asarray(shot_stamps, dtype=float)
    finite = np.sort(stamps[np.isfinite(stamps)])
    span = float(period)
    if finite.size > 1:
        gaps = np.diff(finite)
        gaps = gaps[gaps >= 1e-3]
        if gaps.size:
            span = min(span, float(gaps.min()))
    return span / 2.0


def join_frames_to_shots(
    shot_stamps: np.ndarray,
    frame_stamps: np.ndarray,
    *,
    window: float,
    shot_offset: float = 0.0,
    frame_offset: float = 0.0,
) -> ShotJoin:
    """Match each frame to the nearest shot row within *window*, keep-first.

    Both sides are corrected by their device's drain offset first (``03``
    §11.4: the stamp is the trigger's arrival plus a per-device constant),
    so after the correction one shot's stamps land within NTP jitter of
    each other.

    Parameters
    ----------
    shot_stamps :
        ``(M,)`` clock stamps of the rows, in row order.
    frame_stamps :
        ``(N,)`` per-frame stamps, in frame order, same epoch.
    window :
        Half-width of the match window, seconds (:func:`join_window`).
    shot_offset, frame_offset :
        The clock device's and the frame device's drain offsets, seconds.

    Returns
    -------
    ShotJoin
        The per-row frame indices, the orphan frames and the contested ones.
    """
    shots = np.asarray(shot_stamps, dtype=float) - float(shot_offset)
    frames = np.asarray(frame_stamps, dtype=float) - float(frame_offset)
    order = [i for i in np.argsort(shots, kind="stable") if np.isfinite(shots[i])]
    sorted_shots = shots[order] if order else np.empty(0)
    owner: dict[int, int] = {}  # shot row index → frame index
    orphans: list[int] = []
    contested: list[int] = []
    for frame, stamp in enumerate(frames):
        if not np.isfinite(stamp) or not len(sorted_shots):
            orphans.append(frame)
            continue
        position = int(np.searchsorted(sorted_shots, stamp))
        best: int | None = None
        best_delta = float("inf")
        for candidate in (position - 1, position):
            if not 0 <= candidate < len(sorted_shots):
                continue
            delta = abs(float(sorted_shots[candidate]) - float(stamp))
            if delta < best_delta:
                best, best_delta = candidate, delta
        if best is None or best_delta > window:
            orphans.append(frame)
            continue
        row = int(order[best])
        if row in owner:
            contested.append(frame)
            continue
        owner[row] = frame
    return ShotJoin(
        frame_for_shot=tuple(owner.get(row) for row in range(len(shots))),
        orphans=tuple(orphans),
        contested=tuple(contested),
    )


def frame_columns_from_attributes(
    object_name: str,
    attributes: Mapping[str, np.ndarray],
    *,
    drain_offset: float = 0.0,
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

    Parameters
    ----------
    object_name :
        The stream's ophyd device name, used when an attribute name does
        not carry one (a pre-0.8 stack's bare ``acq_timestamp``).
    attributes :
        Dataset name → ``(N,)`` values
        (``io.scan_stack.read_stack_attributes``).
    drain_offset :
        The device's drain offset, seconds.
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

    stamps: np.ndarray | None = None
    columns: dict[str, np.ndarray] = {}
    for name, values in attributes.items():
        parsed = parse_attribute_name(name)
        device = parsed[0] if parsed else object_name
        suffix = parsed[2] if parsed else name
        if suffix in (FRAME_STAMP_SUFFIX, "acq_timestamp"):
            stamps = np.asarray(values, dtype=float) + float(labview_epoch_offset)
            columns[f"{device}{ACQ_TIMESTAMP_SUFFIX}"] = stamps
        else:
            columns[f"{device}-{suffix}"] = np.asarray(values, dtype=float)
    if stamps is None:
        logger.warning(
            "%s: no frame stamp among the stack's attributes (%s); not joined",
            object_name,
            ", ".join(sorted(attributes)) or "none",
        )
        return None
    return FrameColumns(
        object_name=object_name,
        stamps=stamps,
        columns=columns,
        drain_offset=float(drain_offset),
    )


__all__ = [
    "ACQ_TIMESTAMP_SUFFIX",
    "SHOTS_STREAM",
    "DEFAULT_SHOT_PERIOD_S",
    "FRAME_STAMP_SUFFIX",
    "FrameColumns",
    "ShotJoin",
    "frame_columns_from_attributes",
    "join_frames_to_shots",
    "join_window",
    "shot_clock_column",
]
