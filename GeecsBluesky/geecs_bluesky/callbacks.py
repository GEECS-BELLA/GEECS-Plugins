"""The GEECS outputs of a run, as RunEngine callbacks (plan of record §4.C).

Three document callbacks, each best-effort — a failure is logged and never
raised back into the RunEngine, the scan itself is the priority:

- :class:`ScanInfoCallback` — ``ScanInfoScanNNN.ini`` at the start document
  (the ``[Scan Info]`` keys every downstream reader parses — ScanAnalysis's
  ``Scan Parameter``, the scans database's ``Start`` / ``End`` /
  ``Step size`` / ``Shots per step`` / ``ScanMode`` / ``ScanStartInfo``,
  ``ScanPaths.is_background_scan``'s ``Background``), rewritten at the stop
  document with ``ScanEndInfo`` filled in.
- :class:`SFileCallback` — the legacy scalar files
  (``ScanDataScanNNN.txt`` + ``analysis/sNNN.txt``) at the stop document,
  built from the run's own per-shot rows
  (:func:`geecs_data_utils.write_scalar_files`) — no Tiled round trip, so
  the files exist whether or not the catalog does.  Written for any exit
  status that produced rows: an aborted 500-shot scan's 300 rows are
  data, exactly as the legacy scanner left them.  The rows are the
  ``primary`` events of a strict run and the per-shot sampler's ``shots``
  events of a gated one, and the per-frame columns of every datum-only
  stream (a gated run's cameras, a non-essential camera) are joined onto
  them by offset-corrected stamp — one row per essential shot, orphan
  frames left in the stack (``08_gated_batch.md`` §4.5).
- :class:`ScanLogCallback` — ``scan.log`` attached from the start document
  to the stop document (:class:`geecs_bluesky.scan_log.ScanLogFile`).
- :class:`StackCheckCallback` — at the stop document, for every image
  stack the run's stream resources reference (the PVA gateway's file
  plugin, #806), asserts that the frames on disk are what the documents
  reference: for a stream with event rows (strict ``primary``) the same
  count and the same ``acq_timestamp`` per row; for a datum-only stream (a
  gated run's ``primary``, a non-essential ``<name>_stream``) the frame
  count equals the datums' total width, and a *gated* stack's stamps are
  compared with the ``shots`` rows besides (one frame per shot, none
  orphaned — the batch trims to the quota, so anything else is a defect).
  Synchronicity is checked per scan, never assumed
  (``06_pva_file_plugin.md`` §2.1); a mismatch is a warning in ``scan.log``.

All four read the GEECS keys the claim preprocessor put in the start
document (``scan_number``, ``scan_folder``, ``geecs_scalar_headers``) and
write **into** the claimed folder only — never creating it.  The two that
read the run's streams (the s-file and the stack check) share one piece of
document bookkeeping, :class:`_StreamCallback`, and both do their file
reading on a small thread that waits for the plugin to finalize — the stop
document precedes ``unstage``, and a run callback must never block the
RunEngine.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geecs_data_utils.shot_join import SHOTS_STREAM

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from geecs_data_utils.shot_join import FrameColumns

from geecs_bluesky.scan_log import ScanLogFile

logger = logging.getLogger(__name__)

Document = Mapping[str, Any]

#: Seconds to wait for the file plugin to finalize a stack before reading it.
DEFAULT_FINALIZE_TIMEOUT_S = 15.0

#: The streams whose events can be a run's per-shot rows, in preference
#: order: ``primary`` for a strict run, the per-shot sampler's ``shots`` for
#: a gated one.  No other stream's events are buffered — ``baseline``'s two
#: open/close rows are telemetry, not shots.
ROW_STREAMS = ("primary", SHOTS_STREAM)

#: How many runs' buffers to keep when a run never emits a stop document
#: (the RunEngine always does, even on an abort, so this is a backstop
#: against an unbounded process).
_MAX_OPEN_RUNS = 8


class _RunCallback:
    """Per-run bookkeeping shared by the three: start → stop, keyed by run uid."""

    def __init__(self) -> None:
        self._starts: dict[str, dict[str, Any]] = {}

    def __call__(self, name: str, doc: Document) -> None:
        """Dispatch one document; every failure is logged, never raised."""
        try:
            if name == "start":
                self._starts[str(doc["uid"])] = dict(doc)
                self.on_start(dict(doc))
            elif name == "descriptor":
                self.on_descriptor(doc)
            elif name == "event":
                self.on_event(doc)
            elif name == "event_page":
                # A ``collect``ed stream's rows arrive as pages (the gated
                # run's ``shots``); the hooks see one row at a time either way.
                from event_model import unpack_event_page

                for event in unpack_event_page(doc):  # type: ignore[arg-type]
                    self.on_event(event)
            elif name == "stream_resource":
                self.on_stream_resource(doc)
            elif name == "stream_datum":
                self.on_stream_datum(doc)
            elif name == "stop":
                start = self._starts.pop(str(doc.get("run_start")), None)
                if start is not None:
                    self.on_stop(start, doc)
        except Exception:
            logger.warning(
                "%s failed on a %s document", type(self).__name__, name, exc_info=True
            )

    def on_start(self, start: dict[str, Any]) -> None:
        """Hook: a run opened."""

    def on_descriptor(self, doc: Document) -> None:
        """Hook: a stream was described."""

    def on_event(self, doc: Document) -> None:
        """Hook: one event."""

    def on_stream_resource(self, doc: Document) -> None:
        """Hook: an external data resource was declared."""

    def on_stream_datum(self, doc: Document) -> None:
        """Hook: a slice of an external resource was referenced."""

    def on_stop(self, start: dict[str, Any], stop: Document) -> None:
        """Hook: the run closed."""


def _scan_folder(start: Mapping[str, Any]) -> Path | None:
    folder = start.get("scan_folder")
    if not folder:
        return None
    path = Path(str(folder))
    if not path.is_dir():
        logger.warning("scan folder %s does not exist; nothing written there", path)
        return None
    return path


# ------------------------------------------------- shared stream bookkeeping
@dataclass
class _Stack:
    """One image stack a run's stream resources reference.

    Attributes
    ----------
    data_key :
        The camera's data key — its ophyd name, the event-column prefix.
    stream :
        The stream whose datums reference it (``primary``, ``<name>_stream``).
    uri :
        The stack file's URI, from the stream resource.
    seq_nums :
        The event sequence numbers the datums cover (empty for a
        datum-only stream, whose rows do not exist).
    width :
        The total number of frames the datums reference.
    attributed :
        Whether a datum has yet said which stream owns the stack (a
        ``StreamResource`` does not name one, so the first datum does).
    """

    data_key: str
    stream: str
    uri: str
    seq_nums: list[range] = field(default_factory=list)
    width: int = 0
    attributed: bool = False

    @property
    def path(self) -> Path:
        """The stack file's local path."""
        from urllib.parse import unquote, urlparse

        return Path(unquote(urlparse(self.uri).path))


@dataclass
class _RunStreams:
    """What one run's documents said about its streams.

    Attributes
    ----------
    rows :
        Stream name → the stream's event rows **in arrival order**, each as
        ``(sequence number, data)``.  Both orders matter: the s-file takes
        the rows as they arrived (what the pre-2c writer did, and a partial
        row is data — ``EVENT_SCHEMA.md``), while the stack check maps a
        datum's sequence numbers onto rows and so needs them keyed.
    stacks :
        Data key → the stack it references.
    drain_offsets :
        Object name → its ``drain_offset`` config value, seconds (``03``
        §11.4; read from the streams' descriptor configuration).
    """

    rows: dict[str, list[tuple[int, dict[str, Any]]]] = field(default_factory=dict)
    stacks: dict[str, _Stack] = field(default_factory=dict)
    drain_offsets: dict[str, float] = field(default_factory=dict)

    def stream_rows(self, stream: str) -> list[dict[str, Any]]:
        """The event rows of *stream*, in arrival order (the s-file's rows)."""
        return [data for _seq, data in self.rows.get(stream) or ()]

    def rows_by_seq(self, stream: str) -> dict[int, dict[str, Any]]:
        """The event rows of *stream* keyed by sequence number, last one winning.

        The RunEngine reuses a sequence number after a rewind, and a datum
        that names it means the row emitted last — which is what plain
        assignment in arrival order gives.
        """
        return {seq: data for seq, data in self.rows.get(stream) or ()}

    def row_stream(self) -> str:
        """The stream the run's per-shot rows are in.

        ``primary`` when it carried events (strict), else the sampler's
        ``shots`` (gated), else ``""``.
        """
        for stream in ROW_STREAMS:
            if self.rows.get(stream):
                return stream
        return ""

    def datum_only_stacks(self) -> list[_Stack]:
        """The stacks whose stream carried no event rows — the ones to join."""
        return [s for s in self.stacks.values() if not self.rows.get(s.stream)]


class _StreamCallback(_RunCallback):
    """Per-run bookkeeping of the streams, rows and stacks a run's documents declare.

    Both file-reading outputs need the same four things — which descriptor
    belongs to which stream, the event rows per stream, the stacks the
    stream resources name with the frames their datums reference, and each
    object's ``drain_offset`` from the descriptors' configuration — so they
    are collected once, here, and handed to :meth:`on_streams` at the stop
    document.
    """

    def __init__(self, finalize_timeout: float = DEFAULT_FINALIZE_TIMEOUT_S) -> None:
        self.finalize_timeout = finalize_timeout
        super().__init__()
        self._runs: dict[str, _RunStreams] = {}
        self._streams: dict[str, tuple[str, str]] = {}  # descriptor → (run, stream)
        self._resources: dict[str, tuple[str, str]] = {}  # resource → (run, data key)
        self._threads: list[threading.Thread] = []

    def on_start(self, start: dict[str, Any]) -> None:
        """Open the run's buffers, evicting any run that never closed."""
        while len(self._runs) >= _MAX_OPEN_RUNS:
            stale, _ = self._runs.popitem()  # insertion-ordered: the oldest
            logger.warning(
                "%s: run %s never emitted a stop document; its buffers are dropped",
                type(self).__name__,
                stale,
            )
            self._streams = {k: v for k, v in self._streams.items() if v[0] != stale}
            self._resources = {
                k: v for k, v in self._resources.items() if v[0] != stale
            }
        self._runs[str(start["uid"])] = _RunStreams()

    def on_descriptor(self, doc: Document) -> None:
        """Index the descriptor's stream and harvest its drain offsets."""
        run_uid = str(doc["run_start"])
        run = self._runs.get(run_uid)
        if run is None:
            return
        self._streams[str(doc["uid"])] = (run_uid, str(doc.get("name")))
        for name, config in (doc.get("configuration") or {}).items():
            value = (config.get("data") or {}).get(f"{name}-drain_offset")
            if value is not None:
                run.drain_offsets[str(name)] = float(value)

    def on_event(self, doc: Document) -> None:
        """Buffer an event row under its stream, in arrival order."""
        owner = self._streams.get(str(doc.get("descriptor")))
        if owner is None:
            return
        run_uid, stream = owner
        if stream not in ROW_STREAMS:
            return  # baseline telemetry and monitors are never per-shot rows
        run = self._runs.get(run_uid)
        if run is not None:
            run.rows.setdefault(stream, []).append(
                (int(doc["seq_num"]), dict(doc.get("data") or {}))
            )

    def on_stream_resource(self, doc: Document) -> None:
        """Remember each image stack the run references (frames, not attributes)."""
        from geecs_data_utils.io.scan_stack import FRAMES_DATASET

        run_uid = str(doc.get("run_start"))
        run = self._runs.get(run_uid)
        parameters = doc.get("parameters") or {}
        if (
            run is None
            or doc.get("mimetype") != "application/x-hdf5"
            or parameters.get("dataset") != FRAMES_DATASET
        ):
            return
        key = str(doc["data_key"])
        # A StreamResource names no descriptor (``event_model``'s schema has no
        # such field), so every stack starts attributed to ``primary`` and its
        # first datum — which does carry one — says which stream really owns it.
        run.stacks[key] = _Stack(data_key=key, stream="primary", uri=str(doc["uri"]))
        self._resources[str(doc["uid"])] = (run_uid, key)

    def on_stream_datum(self, doc: Document) -> None:
        """Record which rows (and how many frames) a stack's datum covers."""
        owner = self._resources.get(str(doc.get("stream_resource")))
        if owner is None:
            return
        run_uid, key = owner
        run = self._runs.get(run_uid)
        stack = run.stacks.get(key) if run is not None else None
        if stack is None:
            return
        if not stack.attributed:
            stack.attributed = True
            stack.stream = self._streams.get(
                str(doc.get("descriptor")), (run_uid, stack.stream)
            )[1]
        seq = doc.get("seq_nums") or {}
        indices = doc.get("indices") or {}
        stack.seq_nums.append(range(int(seq.get("start", 0)), int(seq.get("stop", 0))))
        stack.width += int(indices.get("stop", 0)) - int(indices.get("start", 0))

    def on_stop(self, start: dict[str, Any], stop: Document) -> None:
        """Drop the run's indices and hand its streams to the subclass."""
        run_uid = str(start["uid"])
        run = self._runs.pop(run_uid, None) or _RunStreams()
        self._streams = {k: v for k, v in self._streams.items() if v[0] != run_uid}
        self._resources = {k: v for k, v in self._resources.items() if v[0] != run_uid}
        self.on_streams(start, stop, run)

    def on_streams(
        self, start: dict[str, Any], stop: Document, run: _RunStreams
    ) -> None:
        """Hook: the run closed; *run* is everything its documents said."""

    def spawn(self, name: str, target: Any, *args: Any) -> None:
        """Run *target* on a small daemon thread (reading files must not block the RE).

        The thread is as best-effort as the callback itself: a failure in it
        is logged, never left to die as an unhandled thread exception.
        """

        def guarded() -> None:
            try:
                target(*args)
            except Exception:
                logger.warning("%s failed", name, exc_info=True)

        self._threads = [t for t in self._threads if t.is_alive()]
        thread = threading.Thread(target=guarded, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def join(self, timeout: float | None = None) -> None:
        """Wait for the pending file work (tests, orderly shutdown)."""
        for thread in list(self._threads):
            thread.join(timeout)
        self._threads = [t for t in self._threads if t.is_alive()]


def _await_all_finalized(paths: Sequence[Path], timeout: float) -> list[bool]:
    """Wait once, for all of *paths*, returning each one's finalized state.

    One shared deadline: a run with several stacks must not pay the timeout
    per stack.
    """
    import time

    deadline = time.monotonic() + timeout
    done = [False] * len(paths)
    while True:
        for index, path in enumerate(paths):
            if not done[index]:
                done[index] = _is_finalized(path)
        if all(done) or time.monotonic() >= deadline:
            return done
        time.sleep(0.2)


def _is_finalized(path: Path) -> bool:
    """Whether the plugin has marked the stack at *path* finalized."""
    from geecs_data_utils.io.scan_stack import open_stack

    try:
        with open_stack(path) as f:
            return bool(f.attrs.get("finalized", False))
    except OSError:
        return False


def await_finalized(path: Path, timeout: float) -> bool:
    """Wait, bounded, for the file plugin to mark *path* finalized.

    The stop document precedes ``unstage`` (``Capture=0``, when the plugin
    closes the file), so a stack must never be read at the stop document
    itself; the ``finalized`` root attribute is the plugin's "done" flag
    (``06_pva_file_plugin.md`` §5 — read through
    :func:`~geecs_data_utils.io.scan_stack.open_stack`, lock-free).

    Parameters
    ----------
    path :
        The stack file.
    timeout :
        Seconds to wait.

    Returns
    -------
    bool
        Whether the file is finalized.
    """
    return _await_all_finalized([path], timeout)[0]


# ---------------------------------------------------------------- ScanInfo
def scan_parameter(start: Mapping[str, Any]) -> str:
    """The legacy ``Scan Parameter``: the first motor's ``Device Variable`` header.

    ``"Shotnumber"`` for a motionless run (ScanAnalysis reads that, or
    ``"noscan"``, as *no parameter varied*).  The header comes from the
    start document's ``geecs_scalar_headers`` (the motor's readback
    column); the ophyd name is the fallback.
    """
    motors = list(start.get("motors") or [])
    if not motors:
        return "Shotnumber"
    motor = str(motors[0])
    headers: Mapping[str, str] = start.get("geecs_scalar_headers") or {}
    for key, header in headers.items():
        if key == motor or key.startswith(motor + "-"):
            return header
    return motor


def first_axis(start: Mapping[str, Any]) -> tuple[float, float, float]:
    """``(start, end, step)`` of the outermost scanned axis, from the stock metadata.

    Reads ``plan_pattern`` / ``plan_pattern_args`` the way the stock plans
    write them (``inner_product`` for ``scan``, ``inner_list_product`` for
    ``list_scan``, ``outer_product`` / ``outer_list_product`` for the
    grids), then ``plan_args``' ``start`` / ``stop`` / ``num`` (``x2x_scan``,
    ``log_scan``), then ``extents`` + ``shape``; zeros for a motionless run
    or an unknown shape.  A relative plan's values are its offsets.
    """
    pattern = start.get("plan_pattern")
    pargs: Mapping[str, Any] = start.get("plan_pattern_args") or {}
    args = list(pargs.get("args") or [])
    try:
        if pattern in ("inner_list_product", "outer_list_product") and len(args) > 1:
            return _from_points(list(args[1]))
        if pattern == "inner_product" and len(args) > 2:
            num = int(pargs.get("num") or start.get("num_points") or 1)
            return _from_range(float(args[1]), float(args[2]), num)
        if pattern == "outer_product" and len(args) > 3:
            return _from_range(float(args[1]), float(args[2]), int(args[3]))
        plan_args: Mapping[str, Any] = start.get("plan_args") or {}
        if {"start", "stop", "num"} <= set(plan_args):
            return _from_range(
                float(plan_args["start"]),
                float(plan_args["stop"]),
                int(plan_args["num"]),
            )
        extents = start.get("extents")
        shape = start.get("shape")
        if extents and shape:
            lo, hi = extents[0]
            return _from_range(float(lo), float(hi), int(shape[0]))
    except (TypeError, ValueError, IndexError):
        logger.debug("could not derive the scan axis from %r", pattern, exc_info=True)
    return 0.0, 0.0, 0.0


def _from_range(start: float, stop: float, num: int) -> tuple[float, float, float]:
    step = (stop - start) / (num - 1) if num > 1 else 0.0
    return start, stop, step


def _from_points(points: list[Any]) -> tuple[float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0
    first, last = float(points[0]), float(points[-1])
    step = float(points[1]) - first if len(points) > 1 else 0.0
    return first, last, step


def shots_per_step(start: Mapping[str, Any]) -> int:
    """The legacy ``Shots per step``: ``count``'s ``num``, else the bound plan's value."""
    if start.get("plan_name") == "count" or not start.get("motors"):
        return int(start.get("num_points") or start.get("shots_per_step") or 1)
    return int(start.get("shots_per_step") or 1)


def scan_info_lines(start: Mapping[str, Any], *, end_info: str = "") -> list[str]:
    """The ``[Scan Info]`` ini lines for one run (the legacy key set, verbatim)."""
    background = bool(start.get("background", False))
    if not start.get("motors"):
        mode = "background" if background else "noscan"
    else:
        mode = "standard"
    first, last, step = first_axis(start)
    description = str(start.get("description") or "")
    return [
        "[Scan Info]\n",
        f"Scan No = {start.get('scan_number', 0)}\n",
        f'ScanStartInfo = "{description}"\n',
        f'Scan Parameter = "{scan_parameter(start)}"\n',
        f"Start = {first}\n",
        f"End = {last}\n",
        f"Step size = {step}\n",
        f"Shots per step = {shots_per_step(start)}\n",
        f'ScanEndInfo = "{end_info}"\n',
        f"Background = {str(background).lower()}\n",
        f'ScanMode = "{mode}"\n',
        'Scanner = "bluesky"\n',
        f'Plan = "{start.get("plan_name", "")}"\n',
        f'Trigger profile = "{start.get("trigger_profile", "")}"\n',
    ]


class ScanInfoCallback(_RunCallback):
    """Write ``ScanInfoScanNNN.ini`` at the start document; fill ``ScanEndInfo`` at the stop."""

    def on_start(self, start: dict[str, Any]) -> None:
        """Write the ini into the claimed folder."""
        self._write(start)

    def on_stop(self, start: dict[str, Any], stop: Document) -> None:
        """Rewrite the ini with the run's outcome."""
        exit_status = str(stop.get("exit_status") or "")
        reason = str(stop.get("reason") or "")
        end_info = exit_status + (f": {reason}" if reason else "")
        self._write(start, end_info=end_info)

    @staticmethod
    def _write(start: Mapping[str, Any], *, end_info: str = "") -> Path | None:
        folder = _scan_folder(start)
        if folder is None:
            return None
        path = folder / f"ScanInfo{folder.name}.ini"
        with path.open("w", encoding="utf-8") as fh:
            fh.writelines(scan_info_lines(start, end_info=end_info))
        logger.info("Scan info written to %s", path)
        return path


# ------------------------------------------------------------------ s-file
class SFileCallback(_StreamCallback):
    """Write the legacy scalar files at the stop document from the run's per-shot rows.

    The rows are the ``primary`` events when the run has them (strict:
    every essential device is read per shot) and the per-shot sampler's
    ``shots`` events otherwise (gated: ``primary`` carries only the
    cameras' frames).  The baseline stream's open/close telemetry is never
    a row source.  The columns are whatever the rows carried, renamed and
    ordered by ``geecs_scalar_headers`` inside
    :func:`geecs_data_utils.build_legacy_scalar_dataframe`.

    Every **datum-only** stream of the run — a gated run's cameras, a
    non-essential camera in either mode — has its per-frame columns joined
    onto those rows by offset-corrected stamp
    (:mod:`geecs_data_utils.shot_join`, ``08_gated_batch.md`` §4.5): one
    s-file row per essential shot, a camera's per-frame scalars spelled as
    a strict row spells them, and a frame with no shot inside the window
    left where it is (in the stack and in Tiled, out of the s-file).

    A run with no such stream is written **synchronously**, as before.  A
    run with one is written on a thread, because a stack may only be read
    after the plugin finalizes it and that happens at ``unstage``, after
    the stop document; a stack that never finalizes costs its own columns
    and a warning, never the s-file.

    Parameters
    ----------
    finalize_timeout :
        Seconds to wait for the plugin to finalize each stack.
    """

    def on_streams(
        self, start: dict[str, Any], stop: Document, run: _RunStreams
    ) -> None:
        """Write the files from the run's rows, joining the stacks if it has any."""
        stream = run.row_stream()
        rows = run.stream_rows(stream) if stream else []
        if not rows:
            logger.info(
                "scan %s: no per-shot rows in any stream, no scalar files "
                "(exit_status=%s)",
                start.get("scan_number"),
                stop.get("exit_status"),
            )
            return
        stacks = run.datum_only_stacks()
        if not stacks:
            self._write(start, rows, (), run.drain_offsets)
            return
        logger.info(
            "scan %s: s-file from the %s rows joined to %d stack(s): %s",
            start.get("scan_number"),
            stream,
            len(stacks),
            ", ".join(s.data_key for s in stacks),
        )
        self.spawn(
            f"s-file[{start.get('scan_number')}]",
            self._join_and_write,
            dict(start),
            rows,
            stacks,
            dict(run.drain_offsets),
            self.finalize_timeout,
        )

    def _join_and_write(
        self,
        start: Mapping[str, Any],
        rows: list[dict[str, Any]],
        stacks: list[_Stack],
        drain_offsets: Mapping[str, float],
        finalize_timeout: float,
    ) -> None:
        from geecs_data_utils.io.scan_stack import (
            LABVIEW_EPOCH_OFFSET,
            read_stack_attributes,
            stack_scalar_variables,
        )
        from geecs_data_utils.shot_join import frame_columns_from_attributes

        # One window for every stack, not one each: a gated run with three
        # cameras whose plugin never finalizes must not hold its s-file for
        # three timeouts.
        finalized = {
            stack.data_key: ready
            for stack, ready in zip(
                stacks,
                _await_all_finalized([s.path for s in stacks], finalize_timeout),
                strict=True,
            )
        }
        frames = []
        for stack in stacks:
            columns = None
            if not finalized[stack.data_key]:
                logger.warning(
                    "scan %s: %s not finalized within %.0f s (%s) — its per-frame "
                    "columns are absent from the s-file",
                    start.get("scan_number"),
                    stack.data_key,
                    finalize_timeout,
                    stack.path,
                )
            else:
                columns = frame_columns_from_attributes(
                    stack.data_key,
                    read_stack_attributes(stack.path),
                    variables=stack_scalar_variables(stack.path),
                    labview_epoch_offset=LABVIEW_EPOCH_OFFSET,
                )
            if columns is None:
                continue
            if stack.width and stack.width < len(columns):
                # Frames the documents do not reference: a non-essential camera
                # keeps writing between its ``collect`` and its ``unstage``, and
                # a value in the s-file for a frame Tiled has no datum for would
                # make the two disagree about the same shot.
                logger.info(
                    "scan %s: %s has %d frame(s) on disk and %d referenced by its "
                    "datums; the join uses the referenced ones",
                    start.get("scan_number"),
                    stack.data_key,
                    len(columns),
                    stack.width,
                )
                columns = columns.truncated(stack.width)
            frames.append(columns)
        self._write(start, rows, frames, drain_offsets)

    @staticmethod
    def _write(
        start: Mapping[str, Any],
        rows: list[dict[str, Any]],
        frames: "Sequence[FrameColumns]",
        drain_offsets: Mapping[str, float],
    ) -> None:
        import pandas as pd
        from geecs_data_utils import write_scalar_files

        result = write_scalar_files(
            dict(start),
            pd.DataFrame(rows),
            frames,
            drain_offsets=drain_offsets,
        )
        if result is None:
            logger.warning(
                "scan %s: scalar files not written", start.get("scan_number")
            )


# ---------------------------------------------------------------- scan.log
class ScanLogCallback(_RunCallback):
    """Attach ``scan.log`` in the claimed folder for the span of each run."""

    def __init__(self) -> None:
        super().__init__()
        self._log = ScanLogFile()

    def on_start(self, start: dict[str, Any]) -> None:
        """Open the file (a missing folder is a warning, never a mkdir)."""
        number = start.get("scan_number")
        folder = start.get("scan_folder")
        if number is None or not folder:
            logger.warning(
                "run %s has no scan number/folder; no scan.log", start.get("uid")
            )
            return
        self._log.open(int(number), str(folder))

    def on_stop(self, start: dict[str, Any], stop: Document) -> None:
        """Close the file with the run's outcome."""
        self._log.close(note=f"finished ({stop.get('exit_status', '?')})")


# ------------------------------------------------------------- stack check
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
      frame within the join window and no frame may be orphaned
      (``08_gated_batch.md`` §4.5);
    - any other datum-only stream (a non-essential ``<name>_stream``): the
      count alone — a non-essential camera's frame for shot *k* may land
      during *k+1* and an orphan there is normal, not a defect.

    The stop document precedes ``unstage`` (``Capture=0``, when the plugin
    finalizes and closes the file), and a run callback must not block the
    RunEngine — so the check runs on a small thread that waits, bounded,
    for the ``finalized`` root attribute before reading (lock-free, via
    ``geecs_data_utils.io.scan_stack.open_stack``; design §5).  By then
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
        for stack in run.stacks.values():
            stream_rows = run.rows_by_seq(stack.stream)
            column = f"{stack.data_key}-acq_timestamp"
            expected: list[float] | None = None
            shots: _ShotStamps | None = None
            if stream_rows:
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


def _stack_verdict(start: Mapping[str, Any], message: str, *, warning: bool) -> None:
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
                fh.write(f"{stamp} {level} stack check: {message}\n")
    except OSError:
        logger.debug(
            "could not append the stack verdict to %s", log_path, exc_info=True
        )


@dataclass(frozen=True)
class ScanOutputs:
    """The subscribed output callbacks of a RunEngine, and their tokens.

    Returned so a caller can reach the two that finish their work on a
    thread: :meth:`join` blocks until every pending stack read and s-file
    write is done, which an orderly shutdown (or a test) needs and a
    bare subscription token cannot give.
    """

    stack_check: StackCheckCallback
    scan_log: ScanLogCallback
    scan_info: ScanInfoCallback
    sfile: SFileCallback
    tokens: tuple[int, int, int, int]

    def join(self, timeout: float | None = None) -> None:
        """Wait for the pending stack reads and s-file writes."""
        self.stack_check.join(timeout)
        self.sfile.join(timeout)


def subscribe_scan_outputs(run_engine: Any) -> ScanOutputs:
    """Subscribe the output callbacks; return them with their tokens.

    Order is not load-bearing: the stack check appends its verdict to
    ``scan.log`` itself, after the log callback has closed the file.

    Returns
    -------
    ScanOutputs
        The four callbacks and their subscription tokens.  Iterating it is
        not the same as the pre-0.85 four-tuple of tokens — read
        ``.tokens`` for those.
    """
    stack_check = StackCheckCallback()
    scan_log = ScanLogCallback()
    scan_info = ScanInfoCallback()
    sfile = SFileCallback()
    return ScanOutputs(
        stack_check=stack_check,
        scan_log=scan_log,
        scan_info=scan_info,
        sfile=sfile,
        tokens=(
            run_engine.subscribe(stack_check),
            run_engine.subscribe(scan_log),
            run_engine.subscribe(scan_info),
            run_engine.subscribe(sfile),
        ),
    )


__all__ = [
    "DEFAULT_FINALIZE_TIMEOUT_S",
    "ROW_STREAMS",
    "SFileCallback",
    "ScanOutputs",
    "ScanInfoCallback",
    "ScanLogCallback",
    "StackCheckCallback",
    "await_finalized",
    "first_axis",
    "scan_info_lines",
    "scan_parameter",
    "shots_per_step",
    "subscribe_scan_outputs",
]
