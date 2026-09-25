"""``geecs-tiled-writer``: registers spooled runs in Tiled, off the engine.

The service half of :mod:`geecs_bluesky.tiled_spool`.  Every few seconds
it sweeps the spool directory: complete files (last line a ``stop``) are
replayed, oldest first, through the stock ``TiledWriter`` — with the
stop-time dataset registration made concurrent
(:class:`ConcurrentRunWriter`) — and renamed ``.done``; a file the engine
never finished (the worker died mid-run) is registered after
``--orphan-after`` seconds of silence with a synthesized ``fail`` stop; a
file that fails ``--max-attempts`` times is renamed ``.failed`` and left
for an operator.  Between sweeps it writes ``heartbeat.json``: liveness,
backlog, the last error.  **Nothing reads the heartbeat to refuse a run**
— with the spool a dead writer loses nothing, so the heartbeat is a
warning surface (the scanner's status, ``fleet_status.sh``), never a gate.

Idempotent by construction: before a replay the run's container is
looked up and, when present (a writer that died between registering and
renaming), deleted and registered again from the spool, which holds the
whole record.

Run it as ``geecs-tiled-writer`` (the console script; the unit template
lives beside the qserver's) or ``geecs-tiled-writer --once`` for one
sweep from a shell.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from geecs_bluesky.tiled_spool import (
    DONE_SUFFIX,
    FAILED_SUFFIX,
    SpoolError,
    SpoolLayout,
    SpoolState,
    default_state_dir,
    iter_documents,
    run_uid_of,
    spool_state,
)

logger = logging.getLogger(__name__)

DEFAULT_SWEEP_INTERVAL_S = 2.0
DEFAULT_KEEP_DAYS = 7.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_ORPHAN_AFTER_S = 30 * 60.0
#: Concurrent stop-time registrations.  Tiled's catalog is SQLite: a
#: handful of writers overlap their HTTP round trips without contending
#: for the write lock the way dozens would.
DEFAULT_MAX_WORKERS = 4

#: A heartbeat older than this many sweep intervals is stale (the writer
#: is down or wedged).
STALE_AFTER_SWEEPS = 3


# ── the concurrent stop ──────────────────────────────────────────────────


def _load_tiled_writer_classes() -> tuple[type, type, int]:
    """``(TiledWriter, _RunWriter, BATCH_SIZE)`` — imported lazily (tiled extra)."""
    from bluesky.callbacks.tiled_writer import BATCH_SIZE, TiledWriter, _RunWriter

    return TiledWriter, _RunWriter, BATCH_SIZE


def external_groups(
    external_data_cache: dict[str, Any],
    stream_resource_cache: dict[str, Any],
    desc_nodes: dict[str, Any],
) -> list[list[Any]]:
    """Cached StreamDatums grouped by ``<stream>_<data_key>``, arrival order kept.

    Two stream resources of one data key (a re-prepare mid-run) are
    concatenated by the stock writer in arrival order, so they must stay
    sequential; different keys are independent and run concurrently.
    """
    groups: dict[str, list[Any]] = {}
    for sres_uid, datum in external_data_cache.items():
        sres_doc = stream_resource_cache.get(sres_uid)
        desc_node = desc_nodes.get(datum.get("descriptor")) if sres_doc else None
        if sres_doc is not None and desc_node is not None:
            key = f"{desc_node.item['id']}_{sres_doc['data_key']}"
        else:
            key = str(sres_uid)
        groups.setdefault(key, []).append(datum)
    return list(groups.values())


def make_concurrent_writer_classes(max_workers: int = DEFAULT_MAX_WORKERS):
    """Build the ``TiledWriter`` subclass whose run writer registers datasets concurrently.

    Built on demand rather than at import so the module imports without
    the ``tiled`` extra (the engine side never needs it).  Pinned to the
    stock ``_RunWriter.stop`` structure of bluesky 1.15: the external
    loop is drained concurrently *before* the stock ``stop`` runs, which
    then finds the cache empty and does the rest (internal tables, the
    validation pass, the stop metadata) unchanged.
    """
    TiledWriter, _RunWriter, BATCH_SIZE = _load_tiled_writer_classes()
    # A class body cannot read a closure variable it also assigns.
    workers_wanted = max_workers

    class ConcurrentRunWriter(_RunWriter):
        """The stock run writer with the stop-time registrations in a thread pool."""

        max_workers = workers_wanted

        def stop(self, doc):
            groups = external_groups(
                self._external_data_cache,
                self._stream_resource_cache,
                self._desc_nodes,
            )
            self._external_data_cache.clear()
            if groups:
                workers = max(1, min(self.max_workers, len(groups)))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    # list() re-raises the first failure once every worker
                    # has finished — no half-cancelled pool.
                    list(pool.map(self._write_group, groups))
            super().stop(doc)

        def _write_group(self, datums) -> None:
            for datum in datums:
                self._write_external_data(datum)

    class ConcurrentTiledWriter(TiledWriter):
        """``TiledWriter`` whose per-run writer is :class:`ConcurrentRunWriter`."""

        def _factory(self, name, doc):
            # The stock factory with the run writer swapped (bluesky 1.15).
            cb = run_writer = ConcurrentRunWriter(
                self.client, batch_size=self._batch_size
            )
            if self._normalizer:
                cb = self._normalizer(
                    patches=self.patches, spec_to_mimetype=self.spec_to_mimetype
                )
                cb.subscribe(run_writer)
            if self.backup_directory:
                from bluesky.callbacks.tiled_writer import (
                    JSONLinesWriter,
                    _ConditionalBackup,
                )

                cb = _ConditionalBackup(cb, [JSONLinesWriter(self.backup_directory)])
            return [cb], []

    return ConcurrentTiledWriter, ConcurrentRunWriter


# ── the heartbeat ────────────────────────────────────────────────────────


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


# ── the registrar ────────────────────────────────────────────────────────


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("geecs-bluesky")
    except Exception:  # pragma: no cover - not installed as a distribution
        return "unknown"


def synthesized_stop(run_uid: str, at: float) -> dict[str, Any]:
    """A schema-valid stop for a run the engine never closed (the worker died)."""
    return {
        "uid": f"{run_uid}-synthesized-stop",
        "time": at,
        "run_start": run_uid,
        "exit_status": "fail",
        "reason": (
            "no stop document was spooled: the worker exited during the run "
            "(stop synthesized by geecs-tiled-writer)"
        ),
        "num_events": {},
    }


class SpoolRegistrar:
    """The sweep: prune, classify, replay complete files, write the heartbeat.

    Parameters
    ----------
    layout :
        The state directory's paths.
    tiled_uri, api_key :
        The catalog.
    writer_factory :
        ``client -> callback``; the default builds a
        :func:`make_concurrent_writer_classes` writer.  Tests inject a
        recorder.
    client_factory :
        ``() -> client``; the default is ``tiled.client.from_uri``.
    reachable :
        ``uri -> bool``; the bounded TCP pre-check by default.  An
        unreachable server skips every replay this sweep (the backlog
        waits; nothing is counted as an attempt).
    clock :
        ``time.time`` unless a test says otherwise.
    """

    def __init__(
        self,
        layout: SpoolLayout,
        tiled_uri: str,
        api_key: str | None = None,
        *,
        writer_factory: Callable[[Any], Callable[[str, dict], None]] | None = None,
        client_factory: Callable[[], Any] | None = None,
        reachable: Callable[[str], bool] | None = None,
        sweep_interval: float = DEFAULT_SWEEP_INTERVAL_S,
        keep_days: float = DEFAULT_KEEP_DAYS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        orphan_after_s: float = DEFAULT_ORPHAN_AFTER_S,
        max_workers: int = DEFAULT_MAX_WORKERS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.layout = layout
        self.tiled_uri = tiled_uri
        self._api_key = api_key
        self._writer_factory = writer_factory or self._default_writer_factory
        self._client_factory = client_factory or self._default_client_factory
        if reachable is None:
            from geecs_bluesky.tiled_integration import tiled_server_reachable

            reachable = tiled_server_reachable
        self._reachable = reachable
        self.sweep_interval = float(sweep_interval)
        self.keep_days = float(keep_days)
        self.max_attempts = int(max_attempts)
        self.orphan_after_s = float(orphan_after_s)
        self.max_workers = int(max_workers)
        self._clock = clock
        self._client: Any = None
        self._attempts: dict[str, int] = {}
        self._started_at = clock()
        self._last_ok: float | None = None
        self._last_error: str | None = None
        self._done = 0
        self._registered: list[str] = []

    # -- factories ---------------------------------------------------------

    def _default_client_factory(self) -> Any:
        from tiled.client import from_uri

        return from_uri(self.tiled_uri, api_key=self._api_key)

    def _default_writer_factory(self, client: Any) -> Callable[[str, dict], None]:
        writer_cls, _run_writer_cls = make_concurrent_writer_classes(self.max_workers)
        return writer_cls(client)

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    # -- the sweep ---------------------------------------------------------

    def sweep(self) -> WriterHeartbeat:
        """One pass; returns (and writes) the heartbeat."""
        now = self._clock()
        # What went wrong in THIS sweep: a run that failed and is still
        # pending shows here even when a later run registered fine; a
        # clean sweep clears it (the journal keeps the history).
        self._last_error = None
        self._prune(now)
        complete: list[tuple[Path, bool]] = []
        in_progress = 0
        for path in self.layout.pending_files():
            try:
                state = spool_state(path)
            except OSError as exc:
                logger.warning("%s: cannot read (%s)", path.name, exc)
                in_progress += 1
                continue
            if state is SpoolState.COMPLETE:
                complete.append((path, False))
            elif now - path.stat().st_mtime > self.orphan_after_s:
                complete.append(
                    (path, True)
                )  # orphan: register with a synthesized stop
            else:
                in_progress += 1
        reachable = self._reachable(self.tiled_uri)
        if reachable:
            for path, orphan in complete:
                self._register(path, orphan=orphan, now=now)
        elif complete:
            self._last_error = (
                f"Tiled at {self.tiled_uri} unreachable; {len(complete)} run(s) waiting"
            )
        heartbeat = WriterHeartbeat(
            pid=os.getpid(),
            version=_package_version(),
            started_at=self._started_at,
            last_sweep=self._clock(),
            sweep_interval=self.sweep_interval,
            tiled_uri=self.tiled_uri,
            tiled_reachable=reachable,
            last_ok=self._last_ok,
            last_error=self._last_error,
            pending=len(self.layout.pending_files()) - in_progress,
            in_progress=in_progress,
            failed=len(self.layout.failed_files()),
            done=self._done,
            registered=self._registered[-20:],
        )
        try:
            write_heartbeat(self.layout.heartbeat_path, heartbeat)
        except OSError as exc:
            logger.error(
                "heartbeat not written to %s: %s", self.layout.heartbeat_path, exc
            )
        return heartbeat

    def _prune(self, now: float) -> None:
        cutoff = now - self.keep_days * 86400.0
        for path in self.layout.done_files():
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError as exc:  # pragma: no cover - a race with an operator
                logger.warning("%s: not pruned (%s)", path.name, exc)

    def _register(self, path: Path, *, orphan: bool, now: float) -> None:
        run_uid = run_uid_of(path)
        started = time.monotonic()
        label = run_uid
        try:
            client = self._get_client()
            self._delete_existing(client, run_uid)
            writer = self._writer_factory(client)
            count = 0
            saw_stop = False
            for name, doc in iter_documents(path):
                if name == "start":
                    scan = doc.get("scan_number")
                    if scan is not None:
                        label = f"Scan{int(scan):03d} ({run_uid})"
                writer(name, doc)
                count += 1
                saw_stop = name == "stop"
            if not saw_stop:
                writer("stop", synthesized_stop(run_uid, at=path.stat().st_mtime))
                logger.warning(
                    "%s: no stop document spooled%s — registered with a synthesized "
                    "fail stop",
                    label,
                    " (orphan)" if orphan else "",
                )
        except Exception as exc:
            self._client = None  # a fresh client next time: the old one may be wedged
            attempts = self._attempts.get(run_uid, 0) + 1
            self._attempts[run_uid] = attempts
            self._last_error = f"{label}: {type(exc).__name__}: {exc}"
            if attempts >= self.max_attempts:
                failed = path.with_name(path.name[: -len(".jsonl")] + FAILED_SUFFIX)
                path.rename(failed)
                logger.error(
                    "%s: registration failed %d time(s) — giving up, kept as %s",
                    label,
                    attempts,
                    failed.name,
                    exc_info=True,
                )
            else:
                logger.warning(
                    "%s: registration failed (attempt %d of %d): %s",
                    label,
                    attempts,
                    self.max_attempts,
                    exc,
                    exc_info=True,
                )
            return
        path.rename(path.with_name(path.name[: -len(".jsonl")] + DONE_SUFFIX))
        self._attempts.pop(run_uid, None)
        self._last_ok = self._clock()
        self._done += 1
        self._registered.append(run_uid)
        logger.info(
            "%s: registered in %.1f s (%d documents)",
            label,
            time.monotonic() - started,
            count,
        )

    def _delete_existing(self, client: Any, run_uid: str) -> None:
        """A container left by an earlier attempt goes; the spool holds the whole record."""
        try:
            node = client[run_uid]
        except KeyError:
            return
        logger.warning(
            "%s: a container already exists (an earlier registration was not "
            "marked done) — deleting it and registering again from the spool",
            run_uid,
        )
        node.delete(recursive=True, external_only=False)

    # -- the loop ----------------------------------------------------------

    def run_forever(self, stop: threading.Event | None = None) -> None:
        """Sweep every ``sweep_interval`` seconds until *stop* is set."""
        stop = stop or threading.Event()
        while not stop.is_set():
            try:
                self.sweep()
            except SpoolError:
                logger.exception("sweep aborted by a corrupt spool file")
            except Exception:
                logger.exception("sweep failed")
            stop.wait(self.sweep_interval)


# ── the command ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """The ``geecs-tiled-writer`` command line."""
    parser = argparse.ArgumentParser(
        prog="geecs-tiled-writer",
        description="Register spooled bluesky runs in Tiled, off the engine thread.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help=(
            "the spool + heartbeat directory (default: $GEECS_TILED_WRITER_STATE, "
            "else systemd's $STATE_DIRECTORY, else ~/.local/state/geecs-tiled-writer)"
        ),
    )
    parser.add_argument(
        "--tiled-uri", default=None, help="the catalog (default: config.ini [tiled])"
    )
    parser.add_argument(
        "--sweep-interval", type=float, default=DEFAULT_SWEEP_INTERVAL_S, help="seconds"
    )
    parser.add_argument(
        "--keep-days",
        type=float,
        default=DEFAULT_KEEP_DAYS,
        help="how long registered spool files stay on disk",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="registration failures before a run is set aside as .failed",
    )
    parser.add_argument(
        "--orphan-after",
        type=float,
        default=DEFAULT_ORPHAN_AFTER_S,
        help="seconds of silence before an unfinished run is registered as failed",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="concurrent dataset registrations at a run's stop",
    )
    parser.add_argument(
        "--once", action="store_true", help="one sweep, then exit (0 = the sweep ran)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    tiled_uri, api_key = args.tiled_uri, None
    if tiled_uri is None:
        from geecs_bluesky.tiled_integration import read_tiled_config

        tiled_uri, api_key = read_tiled_config()
    if not tiled_uri:
        logger.error(
            "no Tiled URI: pass --tiled-uri or configure [tiled] uri in "
            "~/.config/geecs_python_api/config.ini"
        )
        return 2
    state_dir = args.state_dir or default_state_dir(honour_state_directory=True)
    layout = SpoolLayout(state_dir)
    layout.ensure()
    registrar = SpoolRegistrar(
        layout,
        tiled_uri,
        api_key,
        sweep_interval=args.sweep_interval,
        keep_days=args.keep_days,
        max_attempts=args.max_attempts,
        orphan_after_s=args.orphan_after,
        max_workers=args.max_workers,
    )
    logger.info(
        "geecs-tiled-writer %s: spool %s → %s (every %.1f s)",
        _package_version(),
        layout.spool_dir,
        tiled_uri,
        registrar.sweep_interval,
    )
    if args.once:
        heartbeat = registrar.sweep()
        logger.info(
            "swept: %d registered, %d pending, %d in progress, %d failed",
            heartbeat.done,
            heartbeat.pending,
            heartbeat.in_progress,
            heartbeat.failed,
        )
        return 0
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    registrar.run_forever(stop)
    return 0


__all__ = [
    "DEFAULT_MAX_WORKERS",
    "STALE_AFTER_SWEEPS",
    "SpoolRegistrar",
    "WriterHeartbeat",
    "build_parser",
    "external_groups",
    "main",
    "make_concurrent_writer_classes",
    "read_heartbeat",
    "synthesized_stop",
    "write_heartbeat",
]
