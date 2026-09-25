"""``geecs-tiled-writer``: registers spooled runs in Tiled, off the engine.

The service half of :mod:`geecs_bluesky.tiled_spool`.  Every few seconds
it sweeps the spool directory: complete files (last line a ``stop``) are
replayed, oldest first, through the stock ``TiledWriter`` and renamed
``.done``.  A file with no stop whose engine no longer holds it (the
worker died mid-run; :func:`~geecs_bluesky.tiled_spool.spool_is_held`)
is registered after ``--orphan-after`` seconds of silence with a
synthesized ``fail`` stop; a held file is a live run, however long it
stays quiet.  Between sweeps it writes ``heartbeat.json``: liveness,
backlog, the last error — and once more just before each registration,
naming the run (``registering``), since a registration is ~25 s of
silence that a reader must not mistake for death.  **Nothing reads the heartbeat to refuse a run**
— with the spool a dead writer loses nothing, so the heartbeat is a
warning surface (the scanner's status, ``fleet_status.sh``), never a gate.

Registration is the stock writer's, serial: one register plus one
data-source update per external dataset at the stop, ~230 datasets and
~25 s for a 23-device run (measured 2026-09-25).  A concurrent variant
was tried on hardware and made no difference — the SQLite catalog
commits one write at a time — and was removed rather than kept as dead
machinery; on a catalog that takes parallel writes (Postgres) it would
be worth bringing back (git history of #999).

Two kinds of failure, treated differently.  A **corrupt file** (a
malformed line before the last, no start document) will not heal: it is
set aside as ``.jsonl.failed`` at once, for an operator.  Everything else
— Tiled answering 5xx through a restart, a rotated key, full storage, a
transient — is retried per run with **exponential backoff** (the sweep
interval doubling per attempt, capped at ``--max-backoff``) for
``--max-attempts`` attempts, roughly an hour and a quarter at the
defaults, before the file is set aside and any half-registered
container removed.  The spool is durable; giving up early gains nothing.

Idempotent by construction: before a replay the run's container is
looked up and, when present (a writer that died between registering and
renaming, or an earlier failed attempt), deleted and registered again
from the spool, which holds the whole record.

Run it as ``geecs-tiled-writer`` (the console script; the unit template
lives beside the qserver's), ``python -m geecs_bluesky.tiled_writer`` on
a checkout with no reinstall, or either with ``--once`` for one sweep.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from geecs_bluesky.tiled_spool import (
    DONE_SUFFIX,
    FAILED_SUFFIX,
    SpoolError,
    SpoolLayout,
    SpoolState,
    WriterHeartbeat,
    default_state_dir,
    iter_documents,
    run_uid_of,
    spool_is_held,
    spool_state,
    write_heartbeat,
)

logger = logging.getLogger(__name__)

DEFAULT_SWEEP_INTERVAL_S = 2.0
DEFAULT_KEEP_DAYS = 7.0
#: Attempts before a run that keeps failing at Tiled is set aside.  With
#: the backoff doubling from the sweep interval and capped at
#: :data:`DEFAULT_MAX_BACKOFF_S`, fifteen attempts span ~75 minutes — a
#: Tiled restart or migration, not a blip.
DEFAULT_MAX_ATTEMPTS = 15
DEFAULT_MAX_BACKOFF_S = 600.0
DEFAULT_ORPHAN_AFTER_S = 30 * 60.0


def make_tiled_writer(client: Any) -> Callable[[str, dict], None]:
    """The stock ``TiledWriter`` over *client* (imported lazily: the tiled extra)."""
    from bluesky.callbacks.tiled_writer import TiledWriter

    return TiledWriter(client)


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
        ``client -> callback``; the default is :func:`make_tiled_writer`.
        Tests inject a recorder.
    client_factory :
        ``() -> client``; the default is ``tiled.client.from_uri``.
    reachable :
        ``uri -> bool``; the bounded TCP pre-check by default.  An
        unreachable server skips every replay this sweep (the backlog
        waits; nothing is counted as an attempt).
    held :
        ``path -> bool``; whether the engine still holds a file open
        (:func:`~geecs_bluesky.tiled_spool.spool_is_held`).
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
        held: Callable[[Path], bool] = spool_is_held,
        sweep_interval: float = DEFAULT_SWEEP_INTERVAL_S,
        keep_days: float = DEFAULT_KEEP_DAYS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_backoff_s: float = DEFAULT_MAX_BACKOFF_S,
        orphan_after_s: float = DEFAULT_ORPHAN_AFTER_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.layout = layout
        self.tiled_uri = tiled_uri
        self._api_key = api_key
        self._writer_factory = writer_factory or make_tiled_writer
        self._client_factory = client_factory or self._default_client_factory
        if reachable is None:
            from geecs_bluesky.tiled_integration import tiled_server_reachable

            reachable = tiled_server_reachable
        self._reachable = reachable
        self._held = held
        self.sweep_interval = float(sweep_interval)
        self.keep_days = float(keep_days)
        self.max_attempts = int(max_attempts)
        self.max_backoff_s = float(max_backoff_s)
        self.orphan_after_s = float(orphan_after_s)
        self._clock = clock
        self._client: Any = None
        self._attempts: dict[str, int] = {}
        self._next_attempt: dict[str, float] = {}
        #: The failure each backing-off run is waiting out (uid → text): a
        #: run in a retry cycle stays a finding between its attempts.
        self._backoff_errors: dict[str, str] = {}
        self._started_at = clock()
        self._last_ok: float | None = None
        self._last_error: str | None = None
        self._done = 0
        self._registered: list[str] = []

    # -- factories ---------------------------------------------------------

    def _default_client_factory(self) -> Any:
        from tiled.client import from_uri

        return from_uri(self.tiled_uri, api_key=self._api_key)

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
        # clean sweep clears it (the journal keeps the history) — except
        # that a run waiting out its backoff keeps its failure on the
        # heartbeat (``_backoff_errors``): a backlog that is failing, not
        # draining, must read as one between the attempts too.
        self._last_error = None
        self._prune(now)
        complete, in_progress = self._classify(now)
        reachable = self._reachable(self.tiled_uri)
        started = 0
        if reachable:
            for path, orphan in complete:
                run_uid = run_uid_of(path)
                if self._next_attempt.get(run_uid, 0.0) > now:
                    continue  # waiting out its backoff; still pending
                # A registration is ~25 s of silence (one HTTP call at a
                # time on the SQLite catalog): say what is being done before
                # it starts, so a reader's stale rule (``is_stale`` honours
                # ``registering``) does not call work death. ``pending`` is
                # what waits BEHIND this one: every complete file not yet
                # registered this sweep, the backing-off ones included.
                self._write_heartbeat(
                    reachable,
                    pending=len(complete) - started - 1,
                    in_progress=in_progress,
                    registering=run_uid,
                )
                self._register(path, orphan=orphan, now=now)
                started += 1
        elif complete:
            self._last_error = (
                f"Tiled at {self.tiled_uri} unreachable; {len(complete)} run(s) waiting"
            )
        if started:
            # Registrations took seconds; a run may have started meanwhile
            # — recount so its file is in_progress, not pending.
            _complete, in_progress = self._classify(self._clock())
        return self._write_heartbeat(
            reachable,
            pending=len(self.layout.pending_files()) - in_progress,
            in_progress=in_progress,
        )

    def _classify(self, now: float) -> tuple[list[tuple[Path, bool]], int]:
        """The pending files: ``(complete files (path, orphan), in-progress count)``."""
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
            elif self._held(path):
                # The engine holds it: a live run, paused or long — alive.
                in_progress += 1
            elif now - path.stat().st_mtime > self.orphan_after_s:
                complete.append(
                    (path, True)
                )  # orphan: register with a synthesized stop
            else:
                in_progress += 1
        return complete, in_progress

    def _write_heartbeat(
        self,
        reachable: bool,
        *,
        pending: int,
        in_progress: int,
        registering: str | None = None,
    ) -> WriterHeartbeat:
        """The heartbeat as of now, written atomically (a failure is logged, never raised)."""
        now = self._clock()
        # This sweep's failure, else the latest one a backing-off run is
        # still waiting out (the journal has them all).
        last_error = self._last_error
        if last_error is None and self._backoff_errors:
            last_error = next(reversed(self._backoff_errors.values()))
        heartbeat = WriterHeartbeat(
            pid=os.getpid(),
            version=_package_version(),
            started_at=self._started_at,
            last_sweep=now,
            sweep_interval=self.sweep_interval,
            tiled_uri=self.tiled_uri,
            tiled_reachable=reachable,
            last_ok=self._last_ok,
            last_error=last_error,
            backing_off=len(self._backoff_errors),
            pending=pending,
            in_progress=in_progress,
            failed=len(self.layout.failed_files()),
            done=self._done,
            registered=self._registered[-20:],
            registering=registering,
            registering_since=now if registering else None,
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
            saw_start = saw_stop = False
            for name, doc in iter_documents(path):
                if name == "start":
                    saw_start = True
                    scan = doc.get("scan_number")
                    if scan is not None:
                        label = f"Scan{int(scan):03d} ({run_uid})"
                writer(name, doc)
                count += 1
                saw_stop = name == "stop"
            if not saw_start:
                raise SpoolError(
                    f"{path.name}: no start document — nothing to register"
                )
            if not saw_stop:
                writer("stop", synthesized_stop(run_uid, at=path.stat().st_mtime))
                logger.warning(
                    "%s: no stop document spooled%s — registered with a synthesized "
                    "fail stop",
                    label,
                    " (orphan)" if orphan else "",
                )
        except SpoolError as exc:
            # The file itself is the problem; a retry reads the same bytes.
            self._set_aside(path, run_uid, label, exc, attempts=None)
            return
        except Exception as exc:
            self._client = None  # a fresh client next time: the old one may be wedged
            attempts = self._attempts.get(run_uid, 0) + 1
            self._attempts[run_uid] = attempts
            self._last_error = f"{label}: {type(exc).__name__}: {exc}"
            if attempts >= self.max_attempts:
                self._set_aside(path, run_uid, label, exc, attempts=attempts)
                return
            delay = min(self.sweep_interval * (2**attempts), self.max_backoff_s)
            self._next_attempt[run_uid] = self._clock() + delay
            self._backoff_errors[run_uid] = self._last_error
            logger.warning(
                "%s: registration failed (attempt %d of %d) — next try in %.0f s: %s",
                label,
                attempts,
                self.max_attempts,
                delay,
                exc,
                exc_info=True,
            )
            return
        path.rename(path.with_name(path.name[: -len(".jsonl")] + DONE_SUFFIX))
        self._attempts.pop(run_uid, None)
        self._next_attempt.pop(run_uid, None)
        self._backoff_errors.pop(run_uid, None)
        self._last_ok = self._clock()
        self._done += 1
        self._registered.append(run_uid)
        logger.info(
            "%s: registered in %.1f s (%d documents)",
            label,
            time.monotonic() - started,
            count,
        )

    def _set_aside(
        self,
        path: Path,
        run_uid: str,
        label: str,
        exc: BaseException,
        *,
        attempts: int | None,
    ) -> None:
        """Rename ``.failed``, drop a half-registered container, say so loudly."""
        failed = path.with_name(path.name[: -len(".jsonl")] + FAILED_SUFFIX)
        path.rename(failed)
        self._attempts.pop(run_uid, None)
        self._next_attempt.pop(run_uid, None)
        self._backoff_errors.pop(run_uid, None)
        self._last_error = (
            f"{label}: set aside as {failed.name}: {type(exc).__name__}: {exc}"
        )
        try:
            self._delete_existing(self._get_client(), run_uid)
        except Exception as cleanup_exc:  # noqa: BLE001 - best effort, reported
            logger.warning(
                "%s: a partial container may remain in Tiled (%s)", label, cleanup_exc
            )
            self._client = None
        if attempts is None:
            logger.error(
                "%s: the spool file cannot be registered from — kept as %s: %s",
                label,
                failed.name,
                exc,
            )
        else:
            logger.error(
                "%s: registration failed %d time(s) — giving up, kept as %s",
                label,
                attempts,
                failed.name,
                exc_info=exc,
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
        help="registration failures (on the backoff schedule) before a run is set aside as .failed",
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=DEFAULT_MAX_BACKOFF_S,
        help="longest wait between two attempts on one run, seconds",
    )
    parser.add_argument(
        "--orphan-after",
        type=float,
        default=DEFAULT_ORPHAN_AFTER_S,
        help="seconds of silence before an unfinished run its engine no longer holds is registered as failed",
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
        max_backoff_s=args.max_backoff,
        orphan_after_s=args.orphan_after,
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
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_BACKOFF_S",
    "SpoolRegistrar",
    "build_parser",
    "main",
    "make_tiled_writer",
    "synthesized_stop",
]


if (
    __name__ == "__main__"
):  # `python -m geecs_bluesky.tiled_writer` — a checkout with no reinstall
    sys.exit(main())
