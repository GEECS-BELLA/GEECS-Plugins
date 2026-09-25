"""The per-run document spool between the RunEngine and the Tiled writer service.

Registering a run in Tiled is ~500 serial HTTP calls at the stop document
(one register + one data-source update per external dataset — 28 plugin
streams × ~9 keys on a full HTU preset), ~25 s that used to run **on the
engine thread**, ahead of unstage and the trigger box's standby.  The
engine now writes every document of a run to one JSON Lines file in a
spool directory — microseconds per document — and a separate process,
``geecs-tiled-writer`` (:mod:`geecs_bluesky.tiled_writer`), registers the
run from that file once its stop document is on disk.

The spool is the writer's **only** source.  The live 0MQ document stream
(``launch_re_manager.sh``'s proxy) is best-effort by design — documents
published while a consumer is down are gone — so a writer fed from it
would need a second, durable path anyway, and two paths for one record
means deduplicating against Tiled's ``create_container(key=uid)`` and
cleaning up partial registrations.  One durable path is simpler and loses
nothing.  The cost is that a run appears in Tiled at its close plus the
registration time, not at its open; the stock ``TiledWriter`` batched
every table and dataset to the stop document already, so nothing that
read a *running* run's data from Tiled ever worked.

Layout (``GEECS_TILED_WRITER_STATE``, one directory both processes agree on;
the units set it explicitly — see :func:`default_state_dir`)::

    <state>/spool/<start time>-<run uid>.jsonl          being written / awaiting registration
    <state>/spool/<start time>-<run uid>.jsonl.done     registered (pruned after --keep-days)
    <state>/spool/<start time>-<run uid>.jsonl.failed   gave up after --max-attempts; an operator's call
    <state>/heartbeat.json                              the writer's liveness + backlog

A file is *complete* when its last line is a ``stop`` document.  The engine
flushes every line (a worker that dies mid-run leaves every document it
emitted) and calls ``fsync`` once at the stop, so a complete file is
durable before the run is reported finished.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Mapping
from enum import Enum
from pathlib import Path
from typing import IO, Any, Callable

logger = logging.getLogger(__name__)

#: The one knob: the writer's state directory (spool + heartbeat).  Set
#: explicitly in BOTH units (the qserver's and the writer's) — a two-process
#: contract is never left to a per-unit default.
ENV_STATE_DIR = "GEECS_TILED_WRITER_STATE"
#: The state directory when nothing names one (a developer running the
#: worker and the writer by hand): the XDG state home.
DEFAULT_STATE_DIR = Path("~/.local/state/geecs-tiled-writer")

SPOOL_SUBDIR = "spool"
HEARTBEAT_FILE = "heartbeat.json"
PENDING_SUFFIX = ".jsonl"
DONE_SUFFIX = ".jsonl.done"
FAILED_SUFFIX = ".jsonl.failed"

#: How much of a file's tail :func:`spool_state` reads to find its last
#: line — a stop document is a few hundred bytes; 64 KiB covers any
#: document the engine emits last.
_TAIL_BYTES = 64 * 1024


class SpoolError(RuntimeError):
    """A spool file that cannot be read back (a malformed line before the last)."""


class SpoolState(str, Enum):
    """What a pending spool file holds."""

    IN_PROGRESS = (
        "in_progress"  # no stop document yet (the run is on, or the worker died)
    )
    COMPLETE = "complete"  # the last line is the stop document


def default_state_dir(
    env: Mapping[str, str] | None = None, *, honour_state_directory: bool = False
) -> Path:
    """The writer's state directory: the env var, else systemd's, else the XDG default.

    Parameters
    ----------
    env :
        The environment to read (``os.environ`` by default).
    honour_state_directory :
        Whether systemd's ``$STATE_DIRECTORY`` counts.  ``True`` for the
        writer service, whose unit declares ``StateDirectory=`` and *owns*
        the directory.  ``False`` for the engine: the qserver unit may one
        day declare a state directory of its own, and the first entry of
        a colon-separated ``$STATE_DIRECTORY`` would then silently point
        the spool somewhere the writer never looks.  The engine reads the
        explicit variable or falls back to the developer default.
    """
    env = os.environ if env is None else env
    explicit = env.get(ENV_STATE_DIR)
    if explicit:
        return Path(explicit).expanduser()
    if honour_state_directory:
        systemd = env.get("STATE_DIRECTORY")
        if systemd:
            return Path(systemd.split(":")[0])
    return DEFAULT_STATE_DIR.expanduser()


class SpoolLayout:
    """The paths under one state directory (shared by the engine and the writer)."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.spool_dir = self.state_dir / SPOOL_SUBDIR
        self.heartbeat_path = self.state_dir / HEARTBEAT_FILE

    def ensure(self) -> None:
        """Create the spool directory (the state directory is ours to create)."""
        self.spool_dir.mkdir(parents=True, exist_ok=True)

    def file_for(self, run_uid: str, start_time: float) -> Path:
        """The pending file for a run: time-prefixed so a name sort is chronological."""
        return self.spool_dir / f"{int(start_time)}-{run_uid}{PENDING_SUFFIX}"

    def pending_files(self) -> list[Path]:
        """Every pending file (in progress or complete), oldest first."""
        if not self.spool_dir.is_dir():
            return []
        return sorted(
            p for p in self.spool_dir.iterdir() if p.name.endswith(PENDING_SUFFIX)
        )

    def done_files(self) -> list[Path]:
        """Every registered file still on disk."""
        if not self.spool_dir.is_dir():
            return []
        return sorted(
            p for p in self.spool_dir.iterdir() if p.name.endswith(DONE_SUFFIX)
        )

    def failed_files(self) -> list[Path]:
        """Every file the writer gave up on."""
        if not self.spool_dir.is_dir():
            return []
        return sorted(
            p for p in self.spool_dir.iterdir() if p.name.endswith(FAILED_SUFFIX)
        )


def run_uid_of(path: Path) -> str:
    """The run uid a spool file is named after (``<time>-<uid>.jsonl[.done|.failed]``)."""
    name = path.name
    for suffix in (DONE_SUFFIX, FAILED_SUFFIX, PENDING_SUFFIX):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    _time, sep, uid = name.partition("-")
    return uid if sep else name


def _json_default(value: Any) -> Any:
    """JSON for what bluesky documents carry beyond the JSON types.

    numpy scalars and arrays (readings), ``Path`` (a provider's directory).
    Anything else is a real error — a document the writer could not
    replay faithfully must not be spooled as its ``repr``.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy rides with bluesky
        np = None
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def encode_line(name: str, doc: Mapping[str, Any]) -> str:
    """One spool line: ``{"name": ..., "doc": ...}`` plus the newline."""
    return json.dumps({"name": name, "doc": doc}, default=_json_default) + "\n"


class SpoolCallback:
    """The engine-side RunEngine callback: one JSON Lines file per run.

    ``start`` opens the run's file; every document is written and flushed
    (the OS holds it from then on — a dying worker loses nothing it
    emitted); ``stop`` is written, fsynced and the file closed.  One
    run at a time: the plans here never nest runs, and a second ``start``
    while a file is open is an error rather than a silent second file.

    Subscribed through :class:`~geecs_bluesky.tiled_integration.SafeDocumentCallback`,
    so a spool failure (disk full, a document the encoder refuses) is
    logged, disables spooling for the rest of that run, and never fails
    the run itself.
    """

    def __init__(
        self, layout: SpoolLayout, *, fsync: Callable[[int], None] = os.fsync
    ) -> None:
        self._layout = layout
        self._fsync = fsync
        self._file: IO[str] | None = None
        self._run_uid: str | None = None

    @property
    def run_uid(self) -> str | None:
        """The run whose file is open, if any."""
        return self._run_uid

    def __call__(self, name: str, doc: Mapping[str, Any]) -> None:
        """Spool one document."""
        if name == "start":
            if self._file is not None:
                self._abandon()
                raise SpoolError(
                    f"start of run {doc.get('uid')} while run {self._run_uid} is "
                    "still open — nested runs are not spooled"
                )
            self._layout.ensure()
            path = self._layout.file_for(str(doc["uid"]), float(doc.get("time", 0.0)))
            self._file = open(path, "w", encoding="utf-8")
            self._run_uid = str(doc["uid"])
        if self._file is None:
            raise SpoolError(f"{name} document with no run open (run {self._run_uid})")
        try:
            self._file.write(encode_line(name, doc))
            self._file.flush()
            if name == "stop":
                self._fsync(self._file.fileno())
        except Exception:
            # Whatever went wrong, this run's file is not going to be
            # completed by us: close it so the writer never sees a complete
            # file that the engine kept appending to after a failure.
            self._abandon()
            raise
        if name == "stop":
            self._file.close()
            self._file = None
            self._run_uid = None

    def _abandon(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:  # pragma: no cover - best effort on the error path
                pass
        self._file = None
        self._run_uid = None


def spool_state(path: Path) -> SpoolState:
    """Whether a pending file is complete (its last line is a stop document).

    Reads only the tail.  A truncated last line (the engine died mid-write,
    or is mid-write right now) reads as *in progress*.
    """
    size = path.stat().st_size
    if size == 0:
        return SpoolState.IN_PROGRESS
    with open(path, "rb") as fh:
        fh.seek(max(0, size - _TAIL_BYTES))
        tail = fh.read()
    if not tail.endswith(b"\n"):
        return SpoolState.IN_PROGRESS
    last = tail.rstrip(b"\n").rsplit(b"\n", 1)[-1]
    try:
        record = json.loads(last)
    except ValueError:
        return SpoolState.IN_PROGRESS
    return (
        SpoolState.COMPLETE
        if isinstance(record, dict) and record.get("name") == "stop"
        else SpoolState.IN_PROGRESS
    )


def iter_documents(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(name, doc)`` per line, in order.

    A malformed **last** line is a truncated write (the engine died
    mid-line) and is skipped with a warning — the run is then registered
    without a stop document, which the writer synthesizes.  A malformed
    line anywhere else is corruption and raises :class:`SpoolError`.
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    last_index = len(lines) - 1
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
            name, doc = record["name"], record["doc"]
        except (ValueError, KeyError, TypeError) as exc:
            if index == last_index:
                logger.warning(
                    "%s: last line unreadable (%s) — treated as a truncated write",
                    path.name,
                    exc,
                )
                return
            raise SpoolError(
                f"{path.name}: line {index + 1} unreadable: {exc}"
            ) from exc
        yield str(name), dict(doc)


__all__ = [
    "DEFAULT_STATE_DIR",
    "DONE_SUFFIX",
    "ENV_STATE_DIR",
    "FAILED_SUFFIX",
    "HEARTBEAT_FILE",
    "PENDING_SUFFIX",
    "SpoolCallback",
    "SpoolError",
    "SpoolLayout",
    "SpoolState",
    "default_state_dir",
    "encode_line",
    "iter_documents",
    "run_uid_of",
    "spool_state",
]
