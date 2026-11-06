"""The shot table: which record of each per-shot device belongs to which shot.

One reader for both run layouts, over plain inputs: the start document,
the row stream as a DataFrame and one :class:`DeviceRecords` per device.
Tiled and document-buffer adapters build those; nothing here does I/O.

- **Old layout** (no ``layout`` key).  A device whose records live in the
  row stream itself (a strict run's ``primary`` stacks) is positional:
  row *k* is frame *k*.  Every other source (a gated run's ``primary``
  stacks under ``shots`` rows, a streamed ``<name>_stream``) is
  joined exactly as the s-file joins it (:mod:`geecs_data_utils.shot_join`).
- **Shot-table layout** (``layout: "shot-table/1"``).  ``primary`` is the
  shot table and every device is in its own stream.  Time assigns,
  counters audit — a counter never moves a record:

  ============================  ==============================================
  rows / device                 assignment
  ============================  ==============================================
  witnessed, synchronous        nearest offset-corrected stamp, globally
                                nearest pair first (the s-file's join)
  unwitnessed / witness late    the interval (fire *k* − m, fire *k*+1],
  / count not advanced          *m* the stamp budget; the overlap by the
                                run's phase, else *interval-ambiguous*;
                                flagged *unwitnessed* (a repeated count's
                                stamp is the previous row's edge)
  stamp-shared (count +k)       the group's records in counter order when
                                there is one per row and the counters are
                                consecutive, else unresolved (never guessed)
  asynchronous (free-running)   last record at or before the row's clock
                                (fire + *m* on a loose row), its age kept,
                                *stale* past a limit
  ============================  ==============================================

  A soft-tick (fly-mode) run's ticks fill the witness columns; its pressed
  devices are synchronous, filed by tick interval [t_k − m, t_k+1 − m)
  (they acquire after their tick, however late in the period), its
  free-running ones asynchronous.

Row flags are strings, a device's spelled ``<flag>:<device>``:
``missing``, ``stale``, ``audit-mismatch``, ``counter-reset``,
``witness-stale``, ``clock-step``, ``unwitnessed``, ``witness-stamp-shared``,
``witness-stamp-shared-unresolved``, ``interval-ambiguous``,
``complete-mismatch``.  Run-level
device flags (:attr:`ShotTable.device_flags`): ``no-attribute-audit`` (the
witness attribute NaN all run), ``attribute-lag-systematic``,
``required-unresolved``.  The audit never moves a record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from geecs_data_utils.shot_join import (
    DEFAULT_SHOT_PERIOD_S,
    DEVICE_STREAM_SUFFIX,
    FrameColumns,
    _squash,
    clock_device,
    clock_stamps,
    join_frames_to_shots,
    device_stream,
    row_windows,
    shot_clock_column,
)

logger = logging.getLogger(__name__)

#: The start-document key and value of the shot-table layout.
LAYOUT_KEY, SHOT_TABLE_LAYOUT = "layout", "shot-table/1"
#: The start-document ``clock`` of a fly-mode run.
SOFT_TICK = "soft-tick"
SYNCHRONOUS, ASYNCHRONOUS = "synchronous", "asynchronous"

#: The start-document key a shot-table run writes per device stream:
#: ``{<stream>: {"sync_class": ..., "stale_after": <seconds or null>}}``;
#: absent means synchronous, default staleness.
STREAM_META_KEY = "stream_meta"
#: A device's own per-frame counter, by its dataset name, first found wins:
#: areaDetector's ``NDArrayUniqueId`` in a stock stack.
COUNTER_KEYS = ("NDArrayUniqueId", "UniqueId", "ArrayCounter")
#: The witness attribute: the witness count recorded on each frame, the camera
#: family's NDAttribute (``/entry/instrument/NDAttributes/WitnessCount``,
#: int32; NaN on a camera booted without the witness).  No alias: nothing
#: in this repo writes a per-frame ``witness_count`` (that name is the
#: ``primary`` row's column, ``devices/witness.py``).
WITNESS_ATTRIBUTE_KEY = "WitnessCount"

#: How far before its fire an unwitnessed record may be stamped, seconds:
#: a LabVIEW host's clock runs up to 16 ms off the box and a stored
#: drain offset dithers by up to ~10 ms across hosts, so 30 ms.
#: It opens each fire interval early; a record within it of the *next*
#: fire is in the overlap of two intervals (:func:`_assign_intervals`).
STAMP_BUDGET_S = 0.03

#: A run whose audited records sit one shot behind the witness attribute
#: this often is flagged ``attribute-lag-systematic``: the measured poll
#: lag puts 1.7 % there, a period-wrong offset nearly all of them.
SYSTEMATIC_LAG_SHARE = 0.2
#: ...over at least this many audited records.
SYSTEMATIC_LAG_MIN_FRAMES = 10
#: A synchronous device's residual (record stamp − row clock, both
#: offset-corrected) that moves this many shot periods off its local level,
#: and stays there, is a clock step (:func:`_clock_step_audit`).  The
#: residual dithers by a few stamp budgets at most (host clocks, drain
#: offsets), so the step threshold is never under two of them.
CLOCK_STEP_SHARE = 0.1
#: The local level: the median of this many preceding residuals.
CLOCK_STEP_LEVEL_FRAMES = 5


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
        The witness count recorded on each record (the witness attribute), or
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
        Device (or unresolved ``required`` name) → run-level flags.
    held :
        Device → the extras inside a ``witness-stamp-shared-unresolved``
        group's window: records nothing was guessed for, not strays.
    columns :
        Device → the columns it joined onto :attr:`rows`.
    """

    rows: pd.DataFrame
    clock: str | None
    record_for_shot: Mapping[str, tuple[int | None, ...]]
    streams: Mapping[str, str]
    flags: tuple[tuple[str, ...], ...]
    extras: Mapping[str, tuple[int, ...]]
    ages: Mapping[str, np.ndarray]
    device_flags: Mapping[str, tuple[str, ...]]
    held: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    columns: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def fresh_rows(table: ShotTable) -> pd.DataFrame:
    """*table*'s rows with every *stale* sample blanked: the scalar tables' view.

    A stale asynchronous sample stays in :attr:`ShotTable.rows`, with its
    age and its ``stale:<device>`` flag beside it.  A table that carries
    neither (the s-file, a scalar plot) must not pass it off as the shot's
    measurement, so there that device's joined columns are ``NaN`` on each
    row flagged stale.
    """
    blank = {
        name: [k for k, flags in enumerate(table.flags) if f"stale:{name}" in flags]
        for name in table.ages
    }
    if not any(blank.values()):
        return table.rows
    rows = table.rows.copy()
    for name, stale in blank.items():
        for column in table.columns.get(name, ()):
            if stale:
                rows.iloc[stale, rows.columns.get_loc(column)] = np.nan
    return rows


#: (stream, value) pairs already warned about, so a bad value warns once.
_WARNED_SYNC_CLASSES: set[tuple[str, str]] = set()


def device_records(
    stream: str,
    records: FrameColumns,
    start: Mapping[str, Any],
    attributes: Mapping[str, Any] | None = None,
) -> DeviceRecords:
    """The one adapter core: *records* of *stream* as :class:`DeviceRecords`.

    The class and staleness come from the start document's
    :data:`STREAM_META_KEY`; the counter (:data:`COUNTER_KEYS`) and the witness
    attribute (:data:`WITNESS_ATTRIBUTE_KEY`) from *attributes* (a stack's
    datasets, bare names) or from the columns (``<device>-<name>``).
    """
    meta = dict((start.get(STREAM_META_KEY) or {}).get(stream) or {})
    # Case- and separator-blind: event columns are lowercase (``-uniqueid``).
    found = {_squash(str(k)): v for k, v in (attributes or {}).items()}
    columns = {_squash(k): v for k, v in records.columns.items()}

    def per_frame(key: str) -> np.ndarray | None:
        values = found.get(_squash(key))
        if values is None:
            values = columns.get(_squash(f"{records.object_name}-{key}"))
        if values is None or np.size(values) < len(records):
            return None
        return np.asarray(values, dtype=float)[: len(records)]

    counter = next((c for c in map(per_frame, COUNTER_KEYS) if c is not None), None)
    stale = meta.get("stale_after")
    sync_class = str(meta.get("sync_class") or SYNCHRONOUS)
    if sync_class not in (SYNCHRONOUS, ASYNCHRONOUS):
        if (stream, sync_class) not in _WARNED_SYNC_CLASSES:
            _WARNED_SYNC_CLASSES.add((stream, sync_class))
            logger.warning(
                "%s: unknown sync_class %r; read as synchronous", stream, sync_class
            )
        sync_class = SYNCHRONOUS
    return DeviceRecords(
        stream=stream,
        records=records,
        counter=counter,
        witness_count=per_frame(WITNESS_ATTRIBUTE_KEY),
        sync_class=sync_class,
        stale_after=None if stale is None else float(stale),
    )


def per_shot_event_streams(start: Mapping[str, Any]) -> dict[str, str]:
    """Stream → object name of every per-shot device stream *start* names.

    A shot-table run names its streams (``per_shot_streams``,
    ``<device>_stream``); a gated run and an earlier strict one name their
    streamed devices (``non_essential``).  Both are read, so a reader
    gating event streams on either never drops a device's stream.
    """
    streams = {
        device_stream(str(name)): str(name) for name in start.get("non_essential") or ()
    }
    suffix = DEVICE_STREAM_SUFFIX
    for stream in map(str, start.get("per_shot_streams") or ()):
        streams.setdefault(stream, stream.removesuffix(suffix))
    return streams


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
    stamp_budget: float = STAMP_BUDGET_S,
) -> ShotTable:
    """Assign every device's records to the run's rows, and flag them.

    Parameters
    ----------
    start :
        The run's start document.  A shot-table run's ``required`` names
        devices by object name, matched ignoring case and separators; a
        name no :class:`DeviceRecords` resolves gets the device-level flag
        ``required-unresolved`` once and is left out of completeness.
    rows :
        The row stream (``primary``; an old gated run's ``shots``).
    devices :
        One entry per per-shot device.
    row_stream :
        The stream *rows* came from (old layout: positional sources).
    drain_offsets :
        Object name → drain offset, seconds, for the clock and the devices.
    stamp_budget :
        How far before its fire an unwitnessed record may be stamped,
        seconds (:data:`STAMP_BUDGET_S`).

    Notes
    -----
    In the old layout the rows come back exactly as the s-file's
    ``tiled_export.join_frame_columns`` returns them: positional sources
    add no columns, and nothing is joined onto rows without a clock.
    """
    new = start.get(LAYOUT_KEY) == SHOT_TABLE_LAYOUT
    offsets = dict(drain_offsets or {})
    clock = _RowClock.of(start, rows, offsets, new, float(stamp_budget))
    record_for: dict[str, tuple[int | None, ...]] = {}
    fresh: dict[str, tuple[bool, ...]] = {}
    extras: dict[str, tuple[int, ...]] = {}
    ages: dict[str, np.ndarray] = {}
    device_flags: dict[str, tuple[str, ...]] = {}
    joined: dict[str, np.ndarray] = {}  # allocated once: ~300 columns fragment
    columns: dict[str, tuple[str, ...]] = {}
    for source in devices:
        name = source.device
        records = np.asarray(source.records.stamps, dtype=float) - float(
            offsets.get(name, 0.0)
        )
        positional = not new and source.stream == row_stream
        if source.sync_class == ASYNCHRONOUS and not positional:
            assigned, ages[name] = _last_before(clock.sample_at, records)
            stale = ages[name] > (source.stale_after or 2.0 * clock.period)
            for k in np.flatnonzero(stale):
                clock.flags[k].append(f"stale:{name}")
            fresh[name] = tuple(
                a is not None and not s for a, s in zip(assigned, stale)
            )
        else:
            assigned = clock.assign(source, records, positional=positional)
            claimed = set(assigned)
            extras[name] = tuple(i for i in range(len(records)) if i not in claimed)
            run_flags = clock.audit(
                source,
                assigned,
                records,
                attribute=start.get("clock") != SOFT_TICK,
            )
            if run_flags:
                device_flags[name] = run_flags
            fresh[name] = tuple(a is not None for a in assigned)
        record_for[name] = tuple(assigned)
        if new or (not positional and clock.column is not None and len(rows)):
            mine = [
                key
                for key in source.records.columns
                if key not in rows.columns and key not in joined
            ]
            for key in mine:
                joined[key] = _gathered(source.records.columns[key], assigned)
            columns[name] = tuple(mine)
    if new:
        required = _resolve_required(start, list(fresh), device_flags)
        if "complete" in rows:
            _flag_complete_mismatch(rows, required, fresh, clock.flags)
    return ShotTable(
        rows=pd.concat([rows, pd.DataFrame(joined, index=rows.index)], axis=1)
        if joined
        else rows,
        clock=clock.column,
        record_for_shot=record_for,
        streams={source.device: source.stream for source in devices},
        flags=tuple(tuple(f) for f in clock.flags),
        extras=extras,
        held={name: tuple(sorted(i)) for name, i in clock.held.items()},
        ages=ages,
        device_flags=device_flags,
        columns=columns,
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
    budget: float
    flags: list[list[str]]
    held: dict[str, set[int]] = field(default_factory=dict)
    soft: bool = False  # a soft-tick (fly) run: synchronous records by tick interval

    @classmethod
    def of(cls, start, rows, offsets, new: bool, budget: float) -> "_RowClock":
        n = len(rows)
        column = shot_clock_column(start, list(rows.columns))
        stamps = np.full(n, np.nan)
        if column:
            stamps = clock_stamps(rows, column) - float(
                offsets.get(clock_device(column), 0.0)
            )
        loose, shared = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        flags: list[list[str]] = [[] for _ in range(n)]
        if new:
            loose = ~np.isfinite(stamps) | (column == "fire_stamp")
            for name in ("unwitnessed", "witness_late"):
                if name in rows:
                    loose |= rows[name].fillna(False).astype(bool).to_numpy()
            loose |= _count_held(rows)
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
            fires=clock_stamps(rows, "fire_stamp") if has else np.full(n, np.nan),
            counts=_column(rows, "witness_count") if new else None,
            epochs=rows["witness_epoch"].to_numpy()
            if new and "witness_epoch" in rows
            else None,
            period=float(start.get("shot_period") or DEFAULT_SHOT_PERIOD_S),
            budget=budget,
            flags=flags,
            soft=new and start.get("clock") == SOFT_TICK,
        )

    @property
    def sample_at(self) -> np.ndarray:
        """Where a free-running device is sampled: the clock, or fire + budget on a loose row."""
        late_fire = self.fires + self.budget
        return np.where(self.loose & np.isfinite(late_fire), late_fire, self.stamps)

    def assign(
        self, source: DeviceRecords, records: np.ndarray, *, positional: bool
    ) -> list[int | None]:
        """A synchronous device's record per row; flags its gaps."""
        n = self.stamps.size
        unresolved: set[int] = set()
        ambiguous: set[int] = set()
        if positional:
            assigned: list[int | None] = [
                k if k < records.size else None for k in range(n)
            ]
        else:
            if self.soft:
                assigned = _assign_ticks(records, self)
            else:
                join = join_frames_to_shots(
                    np.where(self.loose | self.shared, np.nan, self.stamps),
                    records,
                    windows=row_windows(self.stamps, self.period),
                )
                assigned = list(join.frame_for_shot)
            ambiguous = _assign_intervals(assigned, records, self)
            unresolved, self.held[source.device] = _assign_shared(
                assigned, records, source.counter, self
            )
        for k in range(n):
            if k in unresolved:
                self.flags[k].append(f"witness-stamp-shared-unresolved:{source.device}")
            elif k in ambiguous:
                self.flags[k].append(f"interval-ambiguous:{source.device}")
            elif assigned[k] is None:
                self.flags[k].append(f"missing:{source.device}")
        return assigned

    def audit(
        self,
        source: DeviceRecords,
        assigned,
        records: np.ndarray,
        *,
        attribute: bool,
    ) -> tuple[str, ...]:
        """The counter audit, then (not on a soft tick) the clock-step and attribute audits."""
        if self.counts is None:
            return ()
        name = source.device
        _audit(name, source, assigned, self.counts, self.epochs, self.loose, self.flags)
        if not attribute:
            return ()
        _clock_step_audit(name, source, assigned, records, self)
        return _attribute_audit(
            name, source, assigned, self.counts, self.loose, self.flags
        )


def _resolve_required(
    start: Mapping[str, Any],
    devices: Sequence[str],
    device_flags: dict[str, tuple[str, ...]],
) -> list[str]:
    """``required`` as device names, matched ignoring case and separators."""
    by_key = {_squash(device): device for device in devices}
    resolved = []
    for name in start.get("required") or ():
        device = by_key.get(_squash(str(name)))
        if device is None:
            logger.warning("required device %r has no records; not counted", name)
            device_flags[str(name)] = ("required-unresolved",)
        else:
            resolved.append(device)
    return resolved


def _flag_complete_mismatch(
    rows: pd.DataFrame,
    required: Sequence[str],
    fresh: Mapping[str, tuple[bool, ...]],
    flags: list[list[str]],
) -> None:
    """Flag rows whose ``complete`` disagrees with the join for the required devices.

    A free-running device fills a row only with a record that is not stale.
    """
    for k, complete in enumerate(rows["complete"].fillna(False).astype(bool)):
        filled = all(fresh[device][k] for device in required)
        flags[k] += ["complete-mismatch"] * (filled != complete)


def _count_held(rows: pd.DataFrame) -> np.ndarray:
    """Rows whose ``witness_count`` did not advance over the last counted row's.

    Such a row has no witness edge of its own (a fire nobody saw, a
    witness that dropped the count): its ``witness_stamp`` is that row's
    edge repeated, so it is timed by its fire, like an unwitnessed row,
    and the counted row keeps its edge and its record.  The comparison is
    with the last row whose count is finite: a failed witness read in
    between (``NaN``, epoch ``""``) does not hide the repeat.  A count
    across a ``witness_epoch`` change is a restart, not a repeat.
    """
    counts = _column(rows, "witness_count")
    held = np.zeros(len(rows), dtype=bool)
    if counts is None:
        return held
    epochs = (
        rows["witness_epoch"].to_numpy()
        if "witness_epoch" in rows
        else np.full(len(rows), None)
    )
    last: tuple[float, Any] | None = None
    for k, count in enumerate(counts):
        if not np.isfinite(count):
            continue  # a failed read: no count to compare, none to carry
        held[k] = last is not None and last == (count, epochs[k])
        last = (count, epochs[k])
    return held


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


def _assign_ticks(records: np.ndarray, clock: _RowClock) -> list[int | None]:
    """File a soft-tick run's synchronous records by tick interval [t_k − m, t_k+1 − m).

    A pressed device acquires after its tick and before the next (the
    plan's completeness rule, the same interval), however late within it:
    the nearest tick would hand a frame later than half a period to the
    next row.  Each row takes its latest record in its interval; the
    rest are extras.  A held tick's press can land after the wait closed,
    in the next interval and before that row's own press: the later
    record is the row's own.  *m* is the stamp budget (host clocks a few
    ms apart); the last row's interval closes a period after its tick.
    """
    ticks, n = clock.stamps, clock.stamps.size
    assigned: list[int | None] = [None] * n
    order = np.argsort(records, kind="stable")
    ordered = records[order]
    claimed: set[int] = set()
    for k in range(n):
        if not np.isfinite(ticks[k]):
            continue
        following = ticks[k + 1] if k + 1 < n else np.nan
        if not np.isfinite(following):
            following = ticks[k] + clock.period
        low, high = ticks[k] - clock.budget, following - clock.budget
        start = int(np.searchsorted(ordered, low, "left"))
        stop = int(np.searchsorted(ordered, high, "left"))
        free = [int(i) for i in order[start:stop] if int(i) not in claimed]
        if free:
            assigned[k] = free[-1]
            claimed.add(free[-1])
    return assigned


def _assign_intervals(
    assigned: list[int | None], records: np.ndarray, clock: _RowClock
) -> set[int]:
    """File unwitnessed rows by fire interval (fire k − m, fire k+1]; return the ambiguous rows.

    SINGLESHOT fires on the next external edge, so a fire leads its edge
    by up to a period and the nearest fire is the wrong rule.  Rows go in
    fire order, each taking its earliest free record; the low end opens
    the stamp budget *m* early, for a record stamped a few ms before its
    fire.  A record in the overlap (fire k+1 − m, fire k+1] is either row
    *k*'s late record (a phase-locked loop: fire k+1 waits for the slowest
    required device, so the edge sits a drain before it) or row *k*+1's
    early one with row *k* missed.  It goes to row *k* only when row *k*
    has nothing earlier and the run's records so far sit late (or none has
    been filed yet); when they sit early, row *k* missed and the record
    waits for row *k*+1.  With no evidence after a row already missed,
    nothing is guessed: rows *k* and *k*+1 are left empty and returned
    (*interval-ambiguous*), and their records stay extras.  The last row's
    interval closes a period after its fire.  The plan also records each
    row's pre-fire baseline per required device (``<device>-baseline``):
    a record newer than baseline *k* is row *k*'s or later.  This join
    does not read it.
    """
    claimed = {i for i in assigned if i is not None}
    fires, n = clock.fires, clock.fires.size

    def end(k: int) -> float:
        following = fires[k + 1] if k + 1 < n else np.nan
        return following if np.isfinite(following) else fires[k] + clock.period

    ambiguous: set[int] = set()
    phase, missed = "", False  # where this run's records sit: "early" / "late"
    for k in np.flatnonzero(clock.loose & np.isfinite(fires)):
        if k in ambiguous:
            continue
        low, high = fires[k] - clock.budget, end(k)
        inside = np.flatnonzero((records > low) & (records <= high))
        free = sorted(
            (int(i) for i in inside if int(i) not in claimed), key=lambda i: records[i]
        )
        late = bool(free) and records[free[0]] > high - clock.budget
        if not free or (late and phase == "early"):
            missed = True  # (early run, overlap record: row k+1's)
        elif late and not phase and missed:
            pair = [int(k)] + [int(k) + 1] * bool(k + 1 < n and clock.loose[k + 1])
            ambiguous.update(pair)
            claimed.update(
                int(i)
                for i in np.flatnonzero((records > low) & (records <= end(pair[-1])))
            )
        else:
            assigned[k] = free[0]
            claimed.add(free[0])
            phase = "late" if late else "early"
    return ambiguous


def _assign_shared(
    assigned: list[int | None],
    records: np.ndarray,
    counter: np.ndarray | None,
    clock: _RowClock,
) -> tuple[set[int], set[int]]:
    """File each group of rows sharing one witness stamp, or leave it unresolved.

    A strict row reads count and stamp from one CA reading, so it rarely
    lands here; this is the reader half of per-integer-count rows, the
    shape a gated rework would write (gated is frozen without it), where
    one +k update stamps k rows.  A +k update stamps k edges at the
    poll after the last one, so the group's records lie up to k − 1
    periods before the stamp.  Time cannot
    order them, and nothing is guessed: the group is filled in
    counter order only when it found exactly one record per row and their
    counters are consecutive.  Otherwise its rows stay empty for this
    device and are returned, with the records it found: left unclaimed
    (extras), and held (:attr:`ShotTable.held`).
    """
    claimed = {i for i in assigned if i is not None}
    unresolved: set[int] = set()
    held: set[int] = set()
    half = clock.period / 2.0
    for stamp in np.unique(clock.stamps[clock.shared]):
        group = list(np.flatnonzero(clock.shared & (clock.stamps == stamp)))
        low = stamp - (len(group) - 1) * clock.period - half
        inside = np.flatnonzero((records > low) & (records <= stamp + half))
        found = [int(i) for i in inside if int(i) not in claimed]
        ordered = sorted(found, key=lambda i: counter[i]) if counter is not None else []
        steps = np.diff([counter[i] for i in ordered]) if ordered else np.array([])
        if len(ordered) == len(group) and np.all(steps == 1):
            for k, index in zip(group, ordered):
                assigned[k] = index
                claimed.add(index)
        else:
            unresolved.update(int(k) for k in group)
            held.update(found)
    return unresolved, held


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


def _clock_step_audit(
    name: str,
    source: DeviceRecords,
    assigned: Sequence[int | None],
    records: np.ndarray,
    clock: _RowClock,
) -> None:
    """Flag a step in a synchronous device's residual, and every row it may have moved.

    The residual of a timed row (witnessed, not stamp-shared) is its
    record's stamp less the row's clock, both offset-corrected: a constant
    for a device whose clock tracks the box's.  Where it leaves its local
    level (the median of the last :data:`CLOCK_STEP_LEVEL_FRAMES`) by more
    than the step threshold (:data:`CLOCK_STEP_SHARE` of a period, never
    under two stamp budgets) and the next timed record confirms the new
    level (or there is none), the device's clock stepped there: that row is
    *clock-step*, and a new level begins.  A lone outlier is no step.

    A step is seen only modulo the period, and a counter cannot tell a
    record moved one shot from an edge the device missed (both advance the
    counter once over two shots), so every row of a level off the run's
    first is suspect.  The witness attribute decides it: a row whose
    record's attribute lags the row's count exactly as the first level's
    records did (their commonest lag) is verified; any other row of a
    displaced level — another lag, or no attribute to read — is
    *clock-step* too.

    A step of about a whole number of periods leaves the residual where it
    was, so the attribute is also read without one: every row of a run of
    consecutive timed rows lagging one more than the reference is
    *clock-step* when the run is two rows or longer, or follows a row the
    device has no record for.  A single one-behind row is the measured
    poll lag, and a lag off by more is already *audit-mismatch*
    (:func:`_attribute_audit`).  The operating limits: without the
    attribute a whole-period step moves records exactly as a missed edge
    does (one gap, then every record one row on, counters consistent), and
    it is not detected; with it, a move onto the last row alone after no
    gap reads as poll lag.  The flag never moves a record.
    """
    threshold = max(CLOCK_STEP_SHARE * clock.period, 2.0 * clock.budget)
    timed = [
        (k, i)
        for k, i in enumerate(assigned)
        if i is not None
        and not clock.loose[k]
        and not clock.shared[k]
        and np.isfinite(clock.stamps[k])
        and np.isfinite(records[i])
    ]
    residuals = np.array([records[i] - clock.stamps[k] for k, i in timed])
    if residuals.size < 2:
        return
    level_of, steps = _residual_levels(residuals, threshold)
    first = residuals[level_of == 0]
    base = float(np.median(first[:CLOCK_STEP_LEVEL_FRAMES]))
    attribute = (
        None
        if source.witness_count is None
        else np.asarray(source.witness_count, dtype=float)
    )

    def lag(j: int) -> float:
        k, i = timed[j]
        return np.nan if attribute is None else clock.counts[k] - attribute[i]

    lags = [lag(j) for j in np.flatnonzero(level_of == 0)]
    finite = [x for x in lags if np.isfinite(x)]
    reference = max(set(finite), key=finite.count) if finite else None
    levels = {
        n: float(np.median(residuals[level_of == n][:CLOCK_STEP_LEVEL_FRAMES]))
        for n in set(level_of.tolist())
    }
    flagged = set(steps)
    for j in range(len(timed)):
        displaced = abs(levels[level_of[j]] - base) > threshold
        if displaced and (reference is None or lag(j) != reference):
            flagged.add(j)
    if reference is not None:
        flagged.update(
            _lag_runs(timed, [lag(j) for j in range(len(timed))], reference, assigned)
        )
    for j in sorted(flagged):
        k = timed[j][0]
        if f"clock-step:{name}" not in clock.flags[k]:
            clock.flags[k].append(f"clock-step:{name}")


def _residual_levels(
    residuals: np.ndarray, threshold: float
) -> tuple[np.ndarray, list[int]]:
    """Each residual's level number, and the positions where a new level began.

    A residual off the median of its level's last
    :data:`CLOCK_STEP_LEVEL_FRAMES` by more than *threshold* begins a new
    level when the next one confirms it (or there is none); otherwise it
    is a lone outlier and joins nothing.
    """
    level_of = np.zeros(residuals.size, dtype=int)
    steps: list[int] = []
    window: list[float] = [float(residuals[0])]
    for j in range(1, residuals.size):
        level = float(np.median(window[-CLOCK_STEP_LEVEL_FRAMES:]))
        off = abs(residuals[j] - level) > threshold
        following = residuals[j + 1] if j + 1 < residuals.size else None
        confirmed = following is None or (
            abs(following - level) > threshold
            and abs(following - residuals[j]) <= threshold
        )
        if off and confirmed:
            steps.append(j)
            window = []
        level_of[j] = len(steps)
        if not off or confirmed:
            window.append(float(residuals[j]))
    return level_of, steps


def _lag_runs(
    timed: Sequence[tuple[int, int]],
    lags: Sequence[float],
    reference: float,
    assigned: Sequence[int | None],
) -> set[int]:
    """Timed positions in runs of adjacent rows lagging ``reference + 1``.

    A run counts when it is two rows or longer, or when the row before its
    first has no record from the device.
    """
    found: set[int] = set()
    run: list[int] = []

    def close() -> None:
        first = timed[run[0]][0] if run else 0
        after_gap = first > 0 and assigned[first - 1] is None
        if len(run) >= 2 or (run and after_gap):
            found.update(run)
        run.clear()

    for j, (k, _) in enumerate(timed):
        if lags[j] != reference + 1 or (run and timed[run[-1]][0] != k - 1):
            close()
        if lags[j] == reference + 1:
            run.append(j)
    close()
    return found


def _attribute_audit(
    name: str,
    source: DeviceRecords,
    assigned: Sequence[int | None],
    counts: np.ndarray,
    loose: np.ndarray,
    flags: list[list[str]],
) -> tuple[str, ...]:
    """The witness attribute against the row's count: equal, or one behind.

    An attribute that stopped advancing while the count did is
    *witness-stale*; any other disagreement is *audit-mismatch*.  Returns
    the run-level flags: ``no-attribute-audit`` (NaN on every record), or
    ``attribute-lag-systematic`` when more than
    :data:`SYSTEMATIC_LAG_SHARE` of at least
    :data:`SYSTEMATIC_LAG_MIN_FRAMES` audited records are one behind — the
    signature of every record filed one shot late (an offset wrong by a
    period), which the per-record rule accepts.
    """
    if source.witness_count is None:
        return ()
    attribute = np.asarray(source.witness_count, dtype=float)
    if attribute.size and not np.isfinite(attribute).any():
        logger.warning(
            "%s: the witness attribute is NaN all run; no attribute audit", name
        )
        return ("no-attribute-audit",)
    previous: tuple[float, float] | None = None
    audited = lagged = 0
    for k, i in enumerate(assigned):
        if (
            i is None
            or loose[k]
            or not (np.isfinite(attribute[i]) and np.isfinite(counts[k]))
        ):
            continue
        audited += 1
        lagged += counts[k] - attribute[i] == 1.0
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
    if audited >= SYSTEMATIC_LAG_MIN_FRAMES and lagged > SYSTEMATIC_LAG_SHARE * audited:
        logger.warning(
            "%s: %d of %d records one shot behind the witness; offset suspect",
            name,
            lagged,
            audited,
        )
        return ("attribute-lag-systematic",)
    return ()


__all__ = [
    "ASYNCHRONOUS",
    "COUNTER_KEYS",
    "STREAM_META_KEY",
    "WITNESS_ATTRIBUTE_KEY",
    "LAYOUT_KEY",
    "SHOT_TABLE_LAYOUT",
    "SOFT_TICK",
    "STAMP_BUDGET_S",
    "SYNCHRONOUS",
    "SYSTEMATIC_LAG_MIN_FRAMES",
    "SYSTEMATIC_LAG_SHARE",
    "DeviceRecords",
    "ShotTable",
    "frame_for",
    "fresh_rows",
    "device_records",
    "per_shot_event_streams",
    "shot_table",
]
