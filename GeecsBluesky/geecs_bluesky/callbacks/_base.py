"""The per-run document bookkeeping the GEECS output callbacks share."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geecs_data_utils.shot_join import (
    SHOTS_STREAM,
    non_essential_stream,
    numeric_data_keys,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geecs_data_utils.shot_join import FrameColumns

logger = logging.getLogger(__name__)

Document = Mapping[str, Any]

#: Seconds to wait for the file plugin to finalize a stack before reading it.
DEFAULT_FINALIZE_TIMEOUT_S = 15.0

#: The streams whose events can be a run's per-shot rows, in preference
#: order: ``primary`` for a strict run, the per-shot sampler's ``shots`` for
#: a gated one.  No other stream's events are buffered (a monitor stream is
#: not shots).
ROW_STREAMS = ("primary", SHOTS_STREAM)

#: How many runs' buffers to keep when a run never emits a stop document
#: (the RunEngine always does, even on an abort, so this is a backstop
#: against an unbounded process).
_MAX_OPEN_RUNS = 8


class _RunCallback:
    """Per-run bookkeeping shared by the three: start → stop, keyed by run uid."""

    def __init__(self) -> None:
        self._starts: dict[str, dict[str, Any]] = {}

    def __call__(self, name: str, doc: Document) -> None:  # noqa: C901
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
        except Exception:  # noqa: BLE001 — callback never kills RunEngine
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
        the rows as they arrived (a partial row is data —
        ``EVENT_SCHEMA.md``), while the stack check maps a
        datum's sequence numbers onto rows and so needs them keyed.
    stacks :
        Data key → the stack it references.
    drain_offsets :
        Object name → its ``drain_offset`` config value, seconds (read from
        the streams' descriptor configuration).
    event_streams :
        Stream name → object name, for every non-essential stream the start
        document names (``<name>_stream``).  The ones that carry events are
        a triggered device without a file plugin, one event per stamp it
        published (GeecsBluesky's ``StampStream``): buffered like the rows
        and joined onto them by stamp.
    stream_keys :
        Stream name → its descriptor's numeric data keys, for those streams
        (so one that recorded nothing still contributes its ``NaN`` columns).
    """

    rows: dict[str, list[tuple[int, dict[str, Any]]]] = field(default_factory=dict)
    stacks: dict[str, _Stack] = field(default_factory=dict)
    drain_offsets: dict[str, float] = field(default_factory=dict)
    event_streams: dict[str, str] = field(default_factory=dict)
    stream_keys: dict[str, list[str]] = field(default_factory=dict)

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

    def event_stream_columns(self) -> list["FrameColumns"]:
        """Every non-essential event stream as join columns (in memory, no I/O).

        A stream that is a plugin camera's (it references a stack) is left
        to :meth:`datum_only_stacks`; the rest — a triggered device without
        a plugin — join by the stamp each event carries.
        """
        from geecs_data_utils.shot_join import (
            ACQ_TIMESTAMP_SUFFIX,
            frame_columns_from_events,
        )

        stacked = {s.stream for s in self.stacks.values()}
        out = []
        for stream, obj in self.event_streams.items():
            keys = self.stream_keys.get(stream, ())
            if stream in stacked or f"{obj}{ACQ_TIMESTAMP_SUFFIX}" not in keys:
                continue  # a plugin camera's datum stream (framed or not)
            columns = frame_columns_from_events(
                obj, self.stream_rows(stream), keys=keys
            )
            if columns is not None:
                out.append(columns)
        return out


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
        #: run → {data key: the descriptor object that owns it}.  A device's
        #: capture streams and its ``acq_timestamp`` column are keys of the
        #: SAME object, which is how a stream finds its stamp column without
        #: anyone parsing a name (``object_keys``, event-model).
        self._owners: dict[str, dict[str, str]] = {}
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
            self._owners.pop(stale, None)
        self._runs[str(start["uid"])] = _RunStreams(
            event_streams={
                non_essential_stream(str(name)): str(name)
                for name in start.get("non_essential") or ()
            }
        )

    def on_descriptor(self, doc: Document) -> None:
        """Index the descriptor's stream and harvest its drain offsets."""
        run_uid = str(doc["run_start"])
        run = self._runs.get(run_uid)
        if run is None:
            return
        stream = str(doc.get("name"))
        self._streams[str(doc["uid"])] = (run_uid, stream)
        if stream in run.event_streams:
            run.stream_keys[stream] = numeric_data_keys(doc.get("data_keys") or {})
        owners = self._owners.setdefault(run_uid, {})
        for obj, keys in (doc.get("object_keys") or {}).items():
            for key in keys or ():
                owners[str(key)] = str(obj)
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
        run = self._runs.get(run_uid)
        if stream not in ROW_STREAMS and (
            run is None or stream not in run.event_streams
        ):
            return  # a monitor stream is never per-shot rows
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
        try:
            self.on_streams(start, stop, run)
        finally:
            # After the subclass has read it: the owner map is what
            # ``StackCheckCallback`` resolves each stack's stamp column with.
            self._owners.pop(run_uid, None)

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
            except Exception:  # noqa: BLE001 — callback never kills RunEngine
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
    (read through
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
