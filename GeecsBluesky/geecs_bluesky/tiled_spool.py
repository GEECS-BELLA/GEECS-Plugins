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
    <state>/spool/<start time>-<run uid>.jsonl.failed   set aside (corrupt, or gave up); an operator's call
    <state>/heartbeat.json                              the writer's liveness + backlog

A file is *complete* when its last line is a ``stop`` document.  The engine
flushes every line (a worker that dies mid-run leaves every document it
emitted) and calls ``fsync`` once at the stop, so a complete file is
durable before the run is reported finished.  While a run is open the
engine **holds an advisory lock** on its file (``flock``): that, not
silence, is how the writer tells a live run — paused for an hour, or a
long count — from one whose worker died (:func:`spool_is_held`).

The line format is the stock ``bluesky.callbacks.json_writer`` one,
``{"name": ..., "doc": ...}`` per line, so any bluesky JSON Lines reader
opens a spool file; what the stock writer lacks is the numpy-aware
encoder, the flush-per-document, the fsync and the completeness mark.

The writer's heartbeat model lives here too (:class:`WriterHeartbeat`,
:func:`read_heartbeat`): the scanner and ``fleet_status.sh`` read it, and
a reader of a JSON file has no business importing the service loop.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import IO, Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: no advisory locks
    fcntl = None  # type: ignore[assignment]

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

#: A heartbeat older than this many sweep intervals is stale (the writer
#: is down or wedged).
STALE_AFTER_SWEEPS = 3

#: How much of a file's tail :func:`spool_state` reads to find its last
#: line — a stop document is a few hundred bytes; 64 KiB covers any
#: document the engine emits last.
_TAIL_BYTES = 64 * 1024


class SpoolError(RuntimeError):
    """A spool file that cannot be registered from (corrupt, or without a start)."""


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
        """Every file the writer set aside."""
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
    replay faithfully must not be spooled as its ``repr``.  A numpy scalar
    comes back as the Python type (``float32`` → ``float``): the replayed
    table's dtype is the wide one, whatever ``dtype_numpy`` said.
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


def _hold(fh: IO[str]) -> None:
    """Take the run's advisory lock on its open file (released by close)."""
    if fcntl is not None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def spool_is_held(path: Path) -> bool:
    """Whether the engine still holds *path* open for a run (its advisory lock).

    The writer's liveness test for a file with no stop: a paused run and a
    long count both go silent for longer than any deadline, but the engine
    holds the lock until the stop is written or the process dies.  Without
    advisory locks (Windows) every file reads as not held and the writer
    falls back to the silence deadline alone.
    """
    if fcntl is None:  # pragma: no cover - Windows
        return False
    try:
        fh = open(path, "rb")
    except OSError:
        return False
    with fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False


class SpoolCallback:
    """The engine-side RunEngine callback: one JSON Lines file per run.

    ``start`` opens the run's file and takes its lock; every document is
    written and flushed (the OS holds it from then on — a dying worker
    loses nothing it emitted); ``stop`` is written, fsynced and the file
    closed, which releases the lock.  A start the encoder refuses leaves
    no file behind (the line is encoded before the file exists), so the
    writer never meets an empty file.  One run at a time: the plans here
    never nest runs, and a second ``start`` while a file is open is an
    error rather than a silent second file.

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
            self._open(doc)
            return
        if self._file is None:
            raise SpoolError(f"{name} document with no run open (run {self._run_uid})")
        try:
            self._file.write(encode_line(name, doc))
            self._file.flush()
            if name == "stop":
                self._fsync(self._file.fileno())
        except Exception:
            # Whatever went wrong, this run's file is not going to be
            # completed by us: close it (releasing the lock) so the writer
            # sees an unfinished run, never a complete file the engine
            # kept appending to after a failure.
            self._abandon()
            raise
        if name == "stop":
            self._file.close()
            self._file = None
            self._run_uid = None

    def _open(self, doc: Mapping[str, Any]) -> None:
        if self._file is not None:
            self._abandon()
            raise SpoolError(
                f"start of run {doc.get('uid')} while run {self._run_uid} is "
                "still open — nested runs are not spooled"
            )
        line = encode_line("start", doc)  # before the file exists
        self._layout.ensure()
        path = self._layout.file_for(str(doc["uid"]), float(doc.get("time", 0.0)))
        fh = open(path, "w", encoding="utf-8")
        try:
            _hold(fh)
            fh.write(line)
            fh.flush()
        except Exception:
            fh.close()
            path.unlink(missing_ok=True)
            raise
        self._file = fh
        self._run_uid = str(doc["uid"])

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


# ── the writer's heartbeat (written by the service, read by everyone else) ──


@dataclass
class WriterHeartbeat:
    """What the writer says about itself between sweeps (``heartbeat.json``)."""

    pid: int
    version: str
    started_at: float
    last_sweep: float
    sweep_interval: float
    tiled_uri: str
    tiled_reachable: bool
    last_ok: float | None = None
    last_error: str | None = None
    pending: int = 0
    in_progress: int = 0
    failed: int = 0
    done: int = 0
    registered: list[str] = field(default_factory=list)

    def is_stale(self, now: float | None = None) -> bool:
        """Whether the writer has missed :data:`STALE_AFTER_SWEEPS` sweeps."""
        now = time.time() if now is None else now
        return now - self.last_sweep > STALE_AFTER_SWEEPS * self.sweep_interval

    def to_json(self) -> str:
        """The file's content."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> WriterHeartbeat:
        """Parse a heartbeat file; unknown keys are ignored (a newer writer)."""
        raw = json.loads(text)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


def write_heartbeat(path: Path, heartbeat: WriterHeartbeat) -> None:
    """Atomic write (temp file + rename): a reader never sees a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(heartbeat.to_json(), encoding="utf-8")
    os.replace(tmp, path)


def read_heartbeat(path: Path) -> WriterHeartbeat | None:
    """The heartbeat on disk, or ``None`` when there is none (never ran here)."""
    try:
        return WriterHeartbeat.from_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (ValueError, TypeError) as exc:
        logger.warning("%s unreadable: %s", path, exc)
        return None


__all__ = [
    "DEFAULT_STATE_DIR",
    "DONE_SUFFIX",
    "ENV_STATE_DIR",
    "FAILED_SUFFIX",
    "HEARTBEAT_FILE",
    "PENDING_SUFFIX",
    "STALE_AFTER_SWEEPS",
    "SpoolCallback",
    "SpoolError",
    "SpoolLayout",
    "SpoolState",
    "WriterHeartbeat",
    "default_state_dir",
    "encode_line",
    "iter_documents",
    "read_heartbeat",
    "run_uid_of",
    "spool_is_held",
    "spool_state",
    "write_heartbeat",
]
