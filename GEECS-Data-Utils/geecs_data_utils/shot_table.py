"""The shot table: which record of each per-shot device belongs to which shot.

One reader for both run layouts (``Planning/shot_synchronization.md``
§6.6, "The shared reader helper"), over plain inputs: the start document,
the row stream as a DataFrame and one :class:`DeviceRecords` per device.
Tiled and document-buffer adapters build those; nothing here does I/O.

- **Old layout** (no ``layout`` key).  A device whose records live in the
  row stream itself (a strict run's ``primary`` stacks) is positional:
  row *k* is frame *k*.  Every other source (a gated run's ``primary``
  stacks under ``shots`` rows, a non-essential ``<name>_stream``) is
  joined exactly as the s-file joins it (:mod:`geecs_data_utils.shot_join`).
- **Shot-table layout** (``layout: "shot-table/1"``).  ``primary`` is the
  shot table and every device is in its own stream.  Time assigns,
  counters audit (§4.3):

  ============================  ==============================================
  rows / device                 assignment
  ============================  ==============================================
  witnessed, synchronous        nearest offset-corrected stamp, globally
                                nearest pair first (the s-file's join)
  unwitnessed / witness late    the interval (fire *k*, fire *k*+1]
  stamp-shared (count +k)       the group's records in counter order
  asynchronous (free-running)   last record at or before the row's clock,
                                its age kept, *stale* past a limit
  ============================  ==============================================

  A soft-tick (fly-mode) run's ticks fill the witness columns; its pressed
  devices are synchronous, its free-running ones asynchronous.

Row flags are strings, a device's spelled ``<flag>:<device>``:
``missing``, ``stale``, ``audit-mismatch``, ``counter-reset``,
``witness-stale``, ``unwitnessed``, ``witness-stamp-shared``,
``complete-mismatch``.  A stream with a W5 attribute that is NaN for the
whole run gets the run-level ``no-attribute-audit``.  The audit never moves
a record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from geecs_data_utils.shot_join import (
    DEFAULT_SHOT_PERIOD_S,
    FrameColumns,
    clock_device,
    join_frames_to_shots,
    row_windows,
    shot_clock_column,
    to_labview_epoch,
)

logger = logging.getLogger(__name__)

#: The start-document key and value of the shot-table layout.
LAYOUT_KEY, SHOT_TABLE_LAYOUT = "layout", "shot-table/1"
#: The start-document ``clock`` of a fly-mode run.
SOFT_TICK = "soft-tick"
SYNCHRONOUS, ASYNCHRONOUS = "synchronous", "asynchronous"


@dataclass(frozen=True)
class DeviceRecords:
    """One per-shot device's records, in its stream's order.

    Attributes
    ----------
    stream :
        The stream holding them (``uc_cam_stream``; ``primary`` for an old
        strict run's stacks), what :func:`frame_for` returns.
    records :
        Stamps (LabVIEW epoch, ``NaN`` where none) and scalar columns; an
        old strict stack whose stamps were never read may carry ``NaN``
        stamps of the right length.
    counter :
        The per-record device counter (``ArrayCounter``, ``UniqueId``), or
        ``None``: audits only.
    witness_count :
        The witness count recorded on each record (the W5 attribute), or
        ``None``: audits only.
    sync_class :
        :data:`SYNCHRONOUS` or :data:`ASYNCHRONOUS`.
    stale_after :
        An asynchronous record older than this, seconds, is *stale*;
        ``None`` means two shot periods.
    """

    stream: str
    records: FrameColumns
    counter: np.ndarray | None = None
    witness_count: np.ndarray | None = None
    sync_class: str = SYNCHRONOUS
    stale_after: float | None = None

    @property
    def device(self) -> str:
        """The device's object name."""
        return self.records.object_name


@dataclass(frozen=True)
class ShotTable:
    """A run's shots with each device's record assigned.

    Attributes
    ----------
    rows :
        The row stream with every device's scalars joined (``NaN`` on a
        gap; a column the row already carries is kept as the authority).
    clock :
        The row column the join timed against, or ``None``.
    record_for_shot :
        Device → one record index (or ``None``, a gap) per row.
    streams :
        Device → its stream.
    flags :
        One tuple of flags per row.
    extras :
        Device → record indices no row claimed (kept in the stream).
    ages :
        Asynchronous device → per-row age of its record, seconds.
    device_flags :
        Device → run-level flags (``no-attribute-audit``).
    """

    rows: pd.DataFrame
    clock: str | None
    record_for_shot: Mapping[str, tuple[int | None, ...]]
    streams: Mapping[str, str]
    flags: tuple[tuple[str, ...], ...]
    extras: Mapping[str, tuple[int, ...]]
    ages: Mapping[str, np.ndarray]
    device_flags: Mapping[str, tuple[str, ...]]


def frame_for(table: ShotTable, device: str, shot: int) -> tuple[str, int] | None:
    """``(stream, record index)`` of *device* for *shot*, or ``None``.

    *shot* is the s-file's ``Shotnumber``: the 1-based row position.
    """
    indices = table.record_for_shot.get(device)
    if indices is None or not 1 <= shot <= len(indices):
        return None
    index = indices[shot - 1]
    return None if index is None else (table.streams[device], index)


def shot_table(
    start: Mapping[str, Any],
    rows: pd.DataFrame,
    devices: Sequence[DeviceRecords],
    *,
    row_stream: str = "primary",
    drain_offsets: Mapping[str, float] | None = None,
) -> ShotTable:
    """Assign every device's records to the run's rows, and flag them.

    Parameters
    ----------
    start :
        The run's start document.
    rows :
        The row stream (``primary``; an old gated run's ``shots``).
    devices :
        One entry per per-shot device.
    row_stream :
        The stream *rows* came from (old layout: positional sources).
    drain_offsets :
        Object name → drain offset, seconds, for the clock and the devices.
    """
    rows = rows.reset_index(drop=True)
    new = start.get(LAYOUT_KEY) == SHOT_TABLE_LAYOUT
    offsets = dict(drain_offsets or {})
    clock = _RowClock.of(start, rows, offsets, new)
    record_for: dict[str, tuple[int | None, ...]] = {}
    extras: dict[str, tuple[int, ...]] = {}
    ages: dict[str, np.ndarray] = {}
    device_flags: dict[str, tuple[str, ...]] = {}
    joined: dict[str, np.ndarray] = {}  # allocated once: ~300 columns fragment
    for source in devices:
        name = source.device
        records = np.asarray(source.records.stamps, dtype=float) - float(
            offsets.get(name, 0.0)
        )
        if source.sync_class == ASYNCHRONOUS and (new or source.stream != row_stream):
            assigned, ages[name] = _last_before(clock.stamps, records)
            limit = source.stale_after or 2.0 * clock.period
            for k in np.flatnonzero(ages[name] > limit):
                clock.flags[k].append(f"stale:{name}")
        else:
            assigned = clock.assign(
                source, records, positional=not new and source.stream == row_stream
            )
            claimed = set(assigned)
            extras[name] = tuple(i for i in range(len(records)) if i not in claimed)
            run_flag = clock.audit(
                source, assigned, attribute=start.get("clock") != SOFT_TICK
            )
            if run_flag:
                device_flags[name] = (run_flag,)
        record_for[name] = tuple(assigned)
        for key, values in source.records.columns.items():
            if key not in rows.columns and key not in joined:
                joined[key] = _gathered(values, assigned)
    if new and "complete" in rows:
        _flag_complete_mismatch(start, rows, record_for, clock.flags)
    return ShotTable(
        rows=pd.concat([rows, pd.DataFrame(joined, index=rows.index)], axis=1),
        clock=clock.column,
        record_for_shot=record_for,
        streams={source.device: source.stream for source in devices},
        flags=tuple(tuple(f) for f in clock.flags),
        extras=extras,
        ages=ages,
        device_flags=device_flags,
    )


@dataclass
class _RowClock:
    """The rows' timing: offset-corrected LabVIEW stamps and which rows trust them."""

    column: str | None
    stamps: np.ndarray
    loose: np.ndarray  # rows timed by fire interval (unwitnessed, witness late)
    shared: np.ndarray  # rows whose witness stamp another row shares
    fires: np.ndarray
    counts: np.ndarray | None  # None: no audit (old layout, no witness column)
    epochs: np.ndarray | None
    period: float
    flags: list[list[str]]

    @classmethod
    def of(cls, start, rows, offsets, new: bool) -> "_RowClock":
        n = len(rows)
        column = shot_clock_column(start, list(rows.columns))
        stamps = np.full(n, np.nan)
        if column:
            stamps = to_labview_epoch(column, rows[column]) - float(
                offsets.get(clock_device(column), 0.0)
            )
        loose, shared = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        flags: list[list[str]] = [[] for _ in range(n)]
        if new:
            loose = ~np.isfinite(stamps) | (column == "fire_stamp")
            for name in ("unwitnessed", "witness_late"):
                if name in rows:
                    loose |= rows[name].fillna(False).astype(bool).to_numpy()
            witnessed = pd.Series(np.where(loose, np.nan, stamps))
            shared = (witnessed.duplicated(keep=False) & witnessed.notna()).to_numpy()
            for k in range(n):
                flags[k] += ["unwitnessed"] * bool(loose[k])
                flags[k] += ["witness-stamp-shared"] * bool(shared[k])
        has = new and "fire_stamp" in rows
        return cls(
            column=column,
            stamps=stamps,
            loose=loose,
            shared=shared,
            fires=to_labview_epoch("fire_stamp", rows["fire_stamp"])
            if has
            else np.full(n, np.nan),
            counts=_column(rows, "witness_count") if new else None,
            epochs=rows["witness_epoch"].to_numpy()
            if new and "witness_epoch" in rows
            else None,
            period=float(start.get("shot_period") or DEFAULT_SHOT_PERIOD_S),
            flags=flags,
        )

    def assign(
        self, source: DeviceRecords, records: np.ndarray, *, positional: bool
    ) -> list[int | None]:
        """A synchronous device's record per row; flags its gaps *missing*."""
        n = self.stamps.size
        if positional:
            assigned: list[int | None] = [
                k if k < records.size else None for k in range(n)
            ]
        else:
            join = join_frames_to_shots(
                np.where(self.loose, np.nan, self.stamps),
                records,
                windows=row_windows(self.stamps, self.period),
            )
            assigned = list(join.frame_for_shot)
            _assign_intervals(assigned, self.fires, self.loose, records, self.period)
            _order_shared(assigned, self.stamps, self.shared, source.counter)
        for k in range(n):
            self.flags[k] += [f"missing:{source.device}"] * (assigned[k] is None)
        return assigned

    def audit(self, source: DeviceRecords, assigned, *, attribute: bool) -> str | None:
        """The counter audit, then (not on a soft tick) the attribute audit."""
        if self.counts is None:
            return None
        name = source.device
        _audit(name, source, assigned, self.counts, self.epochs, self.loose, self.flags)
        if not attribute:
            return None
        return _attribute_audit(
            name, source, assigned, self.counts, self.loose, self.flags
        )


def _flag_complete_mismatch(start, rows, record_for, flags: list[list[str]]) -> None:
    """Flag rows whose ``complete`` disagrees with the join for the required devices."""
    required = [str(d) for d in start.get("required") or ()]
    for k, complete in enumerate(rows["complete"].fillna(False).astype(bool)):
        filled = all(
            record_for.get(d, (None,) * len(rows))[k] is not None for d in required
        )
        flags[k] += ["complete-mismatch"] * (filled != complete)


def _column(rows: pd.DataFrame, name: str) -> np.ndarray | None:
    """*name* as floats, or ``None`` when the rows lack it."""
    return (
        pd.to_numeric(rows[name], errors="coerce").to_numpy(float)
        if name in rows
        else None
    )


def _gathered(values: Any, indices: Sequence[int | None]) -> np.ndarray:
    """*values* at *indices*, ``NaN`` on a gap (``tiled_export``'s rule)."""
    array = np.asarray(values, dtype=float)
    out = np.full(len(indices), np.nan)
    for row, index in enumerate(indices):
        if index is not None and 0 <= index < array.size:
            out[row] = array[index]
    return out


def _last_before(
    clocks: np.ndarray, records: np.ndarray
) -> tuple[list[int | None], np.ndarray]:
    """Per row, the last record at or before its clock, and that record's age."""
    order = np.array(
        [i for i in np.argsort(records, kind="stable") if np.isfinite(records[i])],
        dtype=int,
    )
    ordered = records[order]
    assigned: list[int | None] = []
    ages = np.full(clocks.size, np.nan)
    for k, clock in enumerate(clocks):
        position = int(np.searchsorted(ordered, clock, "right")) - 1
        if not np.isfinite(clock) or position < 0:
            assigned.append(None)
            continue
        assigned.append(int(order[position]))
        ages[k] = clock - ordered[position]
    return assigned, ages


def _assign_intervals(
    assigned: list[int | None],
    fires: np.ndarray,
    loose: np.ndarray,
    records: np.ndarray,
    period: float,
) -> None:
    """Give each unwitnessed row the first unclaimed record in (fire k, fire k+1].

    SINGLESHOT fires on the next external edge, so a fire leads its edge by
    up to a period and the nearest fire is the wrong rule.  The last row's
    interval closes a period after its fire.
    """
    claimed = {i for i in assigned if i is not None}
    for k in np.flatnonzero(loose):
        low = fires[k]
        high = (
            fires[k + 1]
            if k + 1 < fires.size and np.isfinite(fires[k + 1])
            else low + period
        )
        inside = np.flatnonzero((records > low) & (records <= high))
        free = [
            int(i)
            for i in inside[np.argsort(records[inside], kind="stable")]
            if i not in claimed
        ]
        if free:
            assigned[k] = free[0]
            claimed.add(free[0])


def _order_shared(
    assigned: list[int | None],
    stamps: np.ndarray,
    shared: np.ndarray,
    counter: np.ndarray | None,
) -> None:
    """Within each group of rows sharing one witness stamp, file records in counter order.

    Time cannot order shots that share a stamp; the device's counter (or
    its stream order) can.  The rows the join filled stay filled.
    """
    for stamp in np.unique(stamps[shared]):
        group = [
            k
            for k in np.flatnonzero(shared & (stamps == stamp))
            if assigned[k] is not None
        ]
        key = (lambda i: (counter[i], i)) if counter is not None else (lambda i: i)
        for k, index in zip(group, sorted((assigned[k] for k in group), key=key)):
            assigned[k] = index


def _audit(
    name: str,
    source: DeviceRecords,
    assigned: Sequence[int | None],
    counts: np.ndarray,
    epochs: np.ndarray | None,
    loose: np.ndarray,
    flags: list[list[str]],
) -> None:
    """Counter deltas against shot deltas between consecutive assigned records.

    Each pair is compared on its own, so a mismatch re-anchors the audit at
    the later record.  With *g* rows between them that the device has no
    record for, the counter may advance anything from the shot delta less
    *g* (the device missed those edges) to the shot delta (it took them,
    the stream lost them); outside that range both ends are flagged, and a
    counter that went backwards is a reset.
    """
    if source.counter is None:
        return
    counter = np.asarray(source.counter, dtype=float)
    pairs = [(k, i) for k, i in enumerate(assigned) if i is not None and not loose[k]]
    for (ka, ia), (kb, ib) in zip(pairs, pairs[1:]):
        same_epoch = epochs is None or epochs[ka] == epochs[kb]
        shots, advanced = counts[kb] - counts[ka], counter[ib] - counter[ia]
        if not (same_epoch and np.isfinite(shots) and np.isfinite(advanced)):
            continue
        if advanced < 0:
            flags[kb].append(f"counter-reset:{name}")
        elif not max(shots - (kb - ka - 1), 1) <= advanced <= shots:
            for k in (ka, kb):
                if f"audit-mismatch:{name}" not in flags[k]:
                    flags[k].append(f"audit-mismatch:{name}")


def _attribute_audit(
    name: str,
    source: DeviceRecords,
    assigned: Sequence[int | None],
    counts: np.ndarray,
    loose: np.ndarray,
    flags: list[list[str]],
) -> str | None:
    """The W5 attribute against the row's count: equal, or one behind.

    An attribute that stopped advancing while the count did is
    *witness-stale*; any other disagreement is *audit-mismatch*.  An
    attribute NaN on every record returns ``no-attribute-audit``.
    """
    if source.witness_count is None:
        return None
    attribute = np.asarray(source.witness_count, dtype=float)
    if attribute.size and not np.isfinite(attribute).any():
        logger.warning(
            "%s: the witness attribute is NaN all run; no attribute audit", name
        )
        return "no-attribute-audit"
    previous: tuple[float, float] | None = None
    for k, i in enumerate(assigned):
        if (
            i is None
            or loose[k]
            or not (np.isfinite(attribute[i]) and np.isfinite(counts[k]))
        ):
            continue
        if counts[k] - attribute[i] not in (0.0, 1.0):
            frozen = (
                previous is not None
                and attribute[i] == previous[0]
                and counts[k] > previous[1]
            )
            flag = f"witness-stale:{name}" if frozen else f"audit-mismatch:{name}"
            if flag not in flags[k]:
                flags[k].append(flag)
        previous = (attribute[i], counts[k])
    return None


__all__ = [
    "ASYNCHRONOUS",
    "LAYOUT_KEY",
    "SHOT_TABLE_LAYOUT",
    "SOFT_TICK",
    "SYNCHRONOUS",
    "DeviceRecords",
    "ShotTable",
    "frame_for",
    "shot_table",
]
