"""The stack-check callback.

At the stop document, checks every image stack the run's stream resources
reference against the documents (frame count and per-row stamps; a gated
stack's stamps against the ``shots`` rows).  A mismatch is a WARNING in
``scan.log``, never a failure.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geecs_data_utils.shot_join import SHOTS_STREAM

from geecs_bluesky.callbacks._base import (
    Document,
    _RunStreams,
    _StreamCallback,
    await_finalized,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = logging.getLogger(__name__)

#: Stamps closer than this are the same shot (ms rounding of a double).
_STAMP_TOLERANCE_S = 1e-3


class StackCheckCallback(_StreamCallback):
    """Assert, per image stack, that the frames on disk are what the documents reference.

    A plugin-backed camera's stream resource names its stack
    (``application/x-hdf5``, dataset ``FRAMES_DATASET``) and its data key
    ``<name>``; its stream datums say which rows own a frame
    (``seq_nums``, assigned by the RunEngine bundler) — a partial row owns
    none even when the camera delivered (its frame was rewound), so the
    rows are taken from the datums, never from the stamp column alone.
    Three shapes, one per way a stack can be referenced:

    - a stream **with event rows** (strict ``primary``): the stack's own
      stamps are read (LabVIEW epoch, the rows' epoch — the plugin stores
      Unix seconds) and compared with those rows' ``<name>-acq_timestamp``;
      the frame count must be the datums' total width and every referenced
      row's stamp its frame's;
    - a **gated** run's datum-only ``primary``: the count, plus the
      ``shots`` rows — the sampler ticked once per shot and the batch
      trimmed every stack to the quota, so *every* row must own exactly one
      frame within the join window and no frame may be orphaned;
    - any other datum-only stream (a non-essential ``<name>_stream``): the
      count alone — a non-essential camera's frame for shot *k* may land
      during *k+1* and an orphan there is normal, not a defect.

    A gated run's **native-saving essentials** (no plugin; their LabVIEW
    files are their record) get a files-versus-rows
    line of their own: the sampler writes each one's
    ``-nonscalar_save_path`` column into every ``shots`` row as a run-long
    constant, and every row's own stamp (``<owner>-acq_timestamp``) is
    matched against the files in that directory by the naming contract
    (``geecs_data_utils.native_files``: ``{stem}_{stamp:.3f}{tail}``, the
    tail the device's — so the listing is keyed by stamp, a per-shot
    sidecar rides with its shot and an Explorer ``Thumbs.db`` is nobody's
    file) — waiting, bounded, for every row's file first (there is no
    write-complete readback; the last file lands a LabVIEW loop period
    after the last edge).  Rows without a file (a dropped frame — a
    missing file, no retake) and file stamps with no row (the shot in
    flight at a pause, an in-flight edge's) are counted **separately**, so neither hides the
    other: WARNING on either, never a failure.

    The stop document precedes ``unstage`` (``Capture=0``, when the plugin
    finalizes and closes the file), and a run callback must not block the
    RunEngine — so the check runs on a small thread that waits, bounded,
    for the ``finalized`` root attribute before reading (lock-free, via
    ``geecs_data_utils.io.scan_stack.open_stack``).  By then
    ``scan.log`` is closed, so the verdict is appended to it directly as
    well as logged.

    Parameters
    ----------
    finalize_timeout :
        Seconds to wait for the plugin to finalize the file.
    """

    def on_streams(
        self, start: dict[str, Any], stop: Document, run: _RunStreams
    ) -> None:
        """Check every stack against its documents, off the RunEngine's thread."""
        gated = str(start.get("acquisition") or "") == "gated"
        owners = self._owners.get(str(start.get("uid") or ""), {})
        for stack in run.stacks.values():
            stream_rows = run.rows_by_seq(stack.stream)
            # One device acquires once, so a SECOND capture stream's frames
            # are stamped by the device's own acq_timestamp — there is no
            # `<device>-<variable>-acq_timestamp` column and never was, so
            # building the name from the data key reported every second
            # stream as "0 rows own a frame" against a column that cannot
            # exist (hardware, 26_0922 Scan005).
            #
            # The owner comes from the descriptor's `object_keys`, never from
            # parsing the name: a device's capture streams and its stamp
            # column are keys of the SAME object.  Stripping a `-<suffix>`
            # instead would be a guess, and a device whose NAME contains
            # hyphens could be resolved to a different device's stamps —
            # wrong data, silently (Codex review of #952).
            # The key is the object's own `<name>-acq_timestamp` — ophyd-async
            # names a child `<parent>-<attr>` — and it must belong to THIS
            # object, not merely exist.  Searching the object's keys for one
            # ending in `-acq_timestamp` would be looser for no gain: the
            # object also owns the file plugin's per-frame
            # `<device>-hdf-<variable>-frame_acq_timestamp`, which is spelled
            # with an underscore today and would become ambiguous the moment
            # anyone re-spelled it, silently dropping every plugin-backed
            # camera to the count-only check.
            owner = owners.get(stack.data_key)
            column = f"{owner}-acq_timestamp" if owner else None
            if column is not None and owners.get(column) != owner:
                column = None  # the device published no stamp of its own
            expected: list[float] | None = None
            shots: _ShotStamps | None = None
            if stream_rows and column:
                seqs = sorted({n for r in stack.seq_nums for n in r})
                expected = [
                    float(stream_rows[n][column])
                    for n in seqs
                    if n in stream_rows and column in stream_rows[n]
                ]
            elif gated and stack.stream == "primary":
                # The gated batch's own stacks: the shots rows ARE the frames'
                # rows, one per shot, so the stamps can be checked after all.
                shots = _shot_stamps(start, run, stack.data_key)
            elif stream_rows:
                # Rows, but the descriptors named no single stamp column for
                # this stack's device: say so rather than quietly dropping to
                # the count-only check, which would pass a stack whose frames
                # belong to nobody.
                _stack_verdict(
                    dict(start),
                    f"{stack.data_key}: no acq_timestamp column found for its "
                    f"device in the run's descriptors; frames checked by count only",
                    warning=True,
                )
            self.spawn(
                f"stack-check[{stack.data_key}]",
                self._check,
                dict(start),
                stack.data_key,
                stack.path,
                expected,
                shots,
                stack.width,
                self.finalize_timeout,
            )
        if gated:
            # The sampler's rows.  Stream-agnostic below this line: a strict
            # run's ``primary`` carries the same column with the same
            # semantics, and the day the strict line is wanted this reads
            # ``run.row_stream()`` instead (a partial strict row — a missed
            # shot, NaN stamp — would then count as a row without a file,
            # which it is).
            rows = run.stream_rows(SHOTS_STREAM)
            for owner, directory, stamps in _native_save_dirs(dict(start), rows):
                self.spawn(
                    f"native-files[{owner}]",
                    self._check_native_files,
                    dict(start),
                    owner,
                    directory,
                    stamps,
                    self.finalize_timeout,
                )
        # A non-essential native saver without a plugin (either mode): its
        # stream's EVENTS are matched to its files — one event per stamp it
        # published, so an event without a file is a lost save and a file
        # without an event a stamp the stream never saw.
        for stream in run.event_streams:
            rows = run.stream_rows(stream)
            for owner, directory, stamps in _native_save_dirs(dict(start), rows):
                self.spawn(
                    f"native-files[{owner}]",
                    self._check_native_files,
                    dict(start),
                    owner,
                    directory,
                    stamps,
                    self.finalize_timeout,
                    f"{stream} event",
                    True,
                )

    @staticmethod
    def _check_native_files(
        start: Mapping[str, Any],
        owner: str,
        directory: Path,
        stamps: Sequence[float],
        timeout: float,
        record: str = "shots row",
        stream_closes: bool = False,
    ) -> None:
        """Match the records' *stamps* against the native files in *directory*.

        *record* names what a stamp came from — a gated run's ``shots row``,
        or a non-essential stream's event (*stream_closes*).  Waits, bounded by *timeout*,
        for every record to have its file (the device writes a LabVIEW
        loop period behind the edge); then one line, WARNING on a record
        without a file or a file stamp with no record — never a failure.
        For a non-essential stream, files stamped **after** its last event
        are counted apart and are no defect: the device keeps saving from
        the stream's close (before ``close_run``) to its ``unstage``, and
        a slow device's last frame lands in that gap.
        """
        from geecs_data_utils.native_files import native_file_keys, timestamp_key

        deadline = time.monotonic() + timeout
        while True:
            keys = native_file_keys(directory)
            claimed, missing = _match_stamps(stamps, keys)
            if not missing or time.monotonic() >= deadline:
                break
            time.sleep(0.5)
        rows = len(stamps)
        short = record.split()[-1]  # "row", "event"
        unclaimed = set(keys) - claimed
        trailing = 0
        finite = [s for s in stamps if s == s]
        if stream_closes and finite:
            last = timestamp_key(max(finite)) + 1  # the %.3f rounding neighbour
            trailing = sum(1 for key in unclaimed if key > last)
        orphans = len(unclaimed) - trailing
        after = (
            f"; {trailing} file(s) after its last event (saved between the "
            "stream's close and its unstage)"
            if trailing
            else ""
        )
        if not directory.is_dir():
            message = f"{owner}: native directory {directory} missing"
            message += f" but {rows} {record}(s)" if rows else ""
            warning = bool(rows)
        elif not missing and not orphans:
            message = (
                f"{owner}: {rows} {record}(s), each with a native file in "
                f"{directory.name}/{after}"
            )
            warning = False
        else:
            message = (
                f"{owner}: {rows - len(missing)} of {rows} {record}(s) have a "
                f"native file in {directory.name}/ — MISMATCH ({len(missing)} "
                f"{short}(s) without a file, {orphans} file stamp(s) with no "
                f"{short}){after}"
            )
            warning = True
        _stack_verdict(start, message, warning=warning, kind="native files check")

    @staticmethod
    def _check(
        start: Mapping[str, Any],
        data_key: str,
        path: Path,
        expected: list[float] | None,
        shots: "_ShotStamps | None",
        width: int,
        finalize_timeout: float,
    ) -> None:
        from geecs_data_utils.io.scan_stack import read_stack_timestamps

        finalized = await_finalized(path, finalize_timeout)
        referenced = width if expected is None else len(expected)
        if not path.is_file():
            verdict = f"{data_key}: stack {path} missing" + (
                f" but {referenced} frame(s) are referenced" if referenced else ""
            )
            _stack_verdict(start, verdict, warning=bool(referenced))
            return
        if not finalized:
            _stack_verdict(
                start,
                f"{data_key}: {path.name} not finalized within {finalize_timeout:.0f} s; not checked",
                warning=True,
            )
            return
        stamps = read_stack_timestamps(path, labview_epoch=True)
        if expected is None:
            # A datum-only stream (gated primary, a non-essential stream):
            # no row of its own carries a stamp; the datums' width is the
            # contract.
            _stack_verdict(
                start,
                f"{data_key}: {len(stamps)} frame(s) in {path.name}, "
                f"{width} referenced by the stream's datums"
                + ("" if len(stamps) == width else " — MISMATCH"),
                warning=len(stamps) != width,
            )
            if shots is not None:
                message, warning = shots.verdict(data_key, path, stamps)
                _stack_verdict(start, message, warning=warning)
            return
        if len(stamps) != len(expected):
            _stack_verdict(
                start,
                f"{data_key}: {len(stamps)} frame(s) in {path.name} but {len(expected)} row(s) own a frame",
                warning=True,
            )
            return
        mismatched = [
            i
            for i, (a, b) in enumerate(zip(stamps, expected, strict=True))
            if abs(float(a) - b) > _STAMP_TOLERANCE_S
        ]
        if mismatched:
            _stack_verdict(
                start,
                f"{data_key}: {len(mismatched)} of {len(stamps)} frame(s) in {path.name} "
                f"do not carry their row's stamp (first at index {mismatched[0]})",
                warning=True,
            )
        else:
            _stack_verdict(
                start,
                f"{data_key}: {len(stamps)} frame(s) in {path.name} match the rows' stamps",
                warning=False,
            )


@dataclass(frozen=True)
class _ShotStamps:
    """The ``shots`` rows a gated stack's frames must fall on, one each.

    Attributes
    ----------
    stamps :
        The rows' clock stamps, LabVIEW epoch, in row order.
    windows :
        Per-row half-windows, seconds
        (``geecs_data_utils.shot_join.row_windows``).
    clock_offset, frame_offset :
        The clock device's and the camera's drain offsets, seconds.
    """

    stamps: Sequence[float]
    windows: "np.ndarray"
    clock_offset: float = 0.0
    frame_offset: float = 0.0

    def verdict(
        self, data_key: str, path: Path, frames: "np.ndarray"
    ) -> tuple[str, bool]:
        """``(message, warning)`` for the stamp comparison of this stack."""
        from geecs_data_utils.shot_join import join_frames_to_shots

        join = join_frames_to_shots(
            self.stamps,
            frames,
            windows=self.windows,
            shot_offset=self.clock_offset,
            frame_offset=self.frame_offset,
        )
        rows = len(join.frame_for_shot)
        if join.matched == rows and not join.orphans and not join.contested:
            return (
                f"{data_key}: {len(frames)} frame(s) in {path.name} match the "
                f"shots rows' stamps",
                False,
            )
        widest = float(max(self.windows)) if len(self.windows) else 0.0
        return (
            f"{data_key}: {join.matched} of {len(frames)} frame(s) in {path.name} "
            f"fall on a shots row (±{widest:.3f} s at most) — {len(join.orphans)} "
            f"orphan(s), {rows - join.matched} shot(s) with no frame",
            True,
        )


def _shot_stamps(
    start: Mapping[str, Any], run: _RunStreams, data_key: str
) -> "_ShotStamps | None":
    """The gated run's shot stamps and per-row windows, or ``None`` without rows."""
    from geecs_data_utils.shot_join import (
        DEFAULT_SHOT_PERIOD_S,
        clock_device,
        row_windows,
        shot_clock_column,
    )

    rows = run.stream_rows(SHOTS_STREAM)
    if not rows:
        return None
    clock = shot_clock_column(start, list(rows[0]))
    if clock is None:
        return None
    stamps = [float(row.get(clock, float("nan"))) for row in rows]
    period = float(start.get("shot_period") or DEFAULT_SHOT_PERIOD_S)
    return _ShotStamps(
        stamps=stamps,
        windows=row_windows(stamps, period),
        clock_offset=float(run.drain_offsets.get(clock_device(clock), 0.0)),
        frame_offset=float(run.drain_offsets.get(data_key, 0.0)),
    )


def _match_stamps(
    stamps: Sequence[float], keys: Mapping[int, Any]
) -> tuple[set[int], list[int]]:
    """``(claimed file keys, indices of rows without a file)`` for *stamps* over *keys*.

    A row claims the file rendering its stamp (``timestamp_key``, probed
    with the ``%.3f`` rounding neighbours — a canonicalisation, never a
    tolerance window); a row with no stamp (``NaN``) claims nothing.
    """
    import math

    from geecs_data_utils.native_files import timestamp_key, timestamp_key_candidates

    claimed: set[int] = set()
    missing: list[int] = []
    for index, stamp in enumerate(stamps):
        hit = None
        if not math.isnan(stamp):
            for candidate in timestamp_key_candidates(timestamp_key(stamp)):
                if candidate in keys:
                    hit = candidate
                    break
        if hit is None:
            missing.append(index)
        else:
            claimed.add(hit)
    return claimed, missing


def _native_save_dirs(
    start: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[tuple[str, Path, list[float]]]:
    """``(owner, directory, the rows' stamps)`` per native-saving device of *rows*.

    The device's ``-nonscalar_save_path`` column is a run-long constant
    (``EVENT_SCHEMA.md``); a column that is not constant is reported as a
    defect of its own and its first value is checked.  The stamps are the
    rows' own ``<owner>-acq_timestamp`` (``NaN`` where a row has none).
    """
    from geecs_bluesky.devices.detector import ACQ_TIMESTAMP, LvNativeFileDataLogic

    suffix = LvNativeFileDataLogic.datakey_suffix
    out: list[tuple[str, Path, list[float]]] = []
    for column in sorted({c for row in rows for c in row if c.endswith(suffix)}):
        owner = column[: -len(suffix)]
        values = [str(row[column]) for row in rows if row.get(column)]
        if not values:
            continue
        if len(set(values)) > 1:
            _stack_verdict(
                start,
                f"{owner}: {column} is not constant over the rows "
                f"({len(set(values))} values); checking {values[0]}",
                warning=True,
                kind="native files check",
            )
        stamp_column = f"{owner}-{ACQ_TIMESTAMP}"
        stamps = [_as_float(row.get(stamp_column)) for row in rows]
        out.append((owner, Path(values[0]), stamps))
    return out


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _stack_verdict(
    start: Mapping[str, Any], message: str, *, warning: bool, kind: str = "stack check"
) -> None:
    """Log the verdict and append it to the run's ``scan.log`` (already closed)."""
    scan = start.get("scan_number")
    line = f"scan {scan}: {message}"
    logger.log(logging.WARNING if warning else logging.INFO, "%s", line)
    folder = start.get("scan_folder")
    if not folder:
        return
    log_path = Path(str(folder)) / "scan.log"
    try:
        if log_path.parent.is_dir():
            with log_path.open("a", encoding="utf-8") as fh:
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                level = "WARNING" if warning else "INFO"
                fh.write(f"{stamp} {level} {kind}: {message}\n")
    except OSError:
        logger.debug(
            "could not append the stack verdict to %s", log_path, exc_info=True
        )
