"""Analysis runs: ScanAnalysis executed directly from the portal.

The run model of ``Planning/data_portal/04_analysis_run_design.md``:
a browser click runs ONE scan analyzer on ONE scan by calling
``ScanAnalyzer.run_analysis`` directly — no task-queue participation
(no status records, claim locks, heartbeats or Google Doc uploads;
those live in ``scan_analysis.task_queue.run_worklist``, which this
module deliberately bypasses). Outputs are whatever the analyzer
writes on its own: figures under the scan's analysis folder and
columns in the s-file (ScanAnalysis's warn-and-overwrite protocol).

Pieces:

- :class:`AnalysisJob` — the portal-private, in-memory job record
  (state / artifacts / error / captured log). Lost on restart; the
  outputs on disk are the durable part.
- :class:`AnalysisRunner` — one worker thread, one job per scan at a
  time, log capture scoped to the worker thread.
- :func:`scan_analysis_factory` — the default analyzer factory
  (``load_diagnostic`` + ``create_scan_analyzer``, imported lazily so
  the portal still boots without the ``analysis`` extra). The app
  takes any factory with the same shape, which is how the tests run
  the whole endpoint ladder against a fake analyzer.
- :func:`analysis_folder_for` / :func:`contained_artifact` — pure
  path helpers for serving what a run produced without ever serving
  outside the scan's own analysis folder.

Nothing here creates anything under ``scans/ScanNNN/`` (repo
scan-folder invariant); the analyzer's own ``ScanPaths(read_mode=True)``
enforces it on the run side.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Protocol

logger = logging.getLogger(__name__)

#: Job states — the wire vocabulary the tab renders. The same words as
#: the task queue's status contract (queued → claimed/running →
#: done / failed / no_data) so a reader of either recognises them.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
NO_DATA = "no_data"
ACTIVE = (QUEUED, RUNNING)


class ScanAnalyzerLike(Protocol):
    """The two calls a run needs — ``ScanAnalyzer``'s public run contract."""

    def run_analysis(self, scan_tag: object) -> Optional[list]:
        """Run on *scan_tag*; return notable artifact paths/labels or None."""

    def cleanup(self) -> None:
        """Release per-scan memory (the caller's duty after every run)."""


#: ``factory(analyzer_id, config_dir) -> analyzer`` — built INSIDE the
#: worker thread so config/instantiation failures land in the job
#: record as ``failed`` rather than as a request error.
AnalyzerFactory = Callable[[str, Path], ScanAnalyzerLike]


def scan_analysis_factory(analyzer_id: str, config_dir: Path) -> ScanAnalyzerLike:
    """Build the real ScanAnalysis analyzer for one diagnostic ID.

    The single-analyzer path the group loader runs in a loop:
    :func:`image_analysis.config.load_diagnostic` (the unified
    diagnostic YAML under ``<config_dir>/analyzers/``) wrapped by
    :func:`scan_analysis.config.create_scan_analyzer`.

    Parameters
    ----------
    analyzer_id : str
        Diagnostic ID (YAML stem, or ``namespace/stem`` when ambiguous).
    config_dir : Path
        The configs tree root (the parent of ``analyzers/``).

    Returns
    -------
    ScanAnalyzerLike
        A runnable ``Array1DScanAnalyzer`` / ``Array2DScanAnalyzer``.

    Raises
    ------
    ImportError
        When the ``analysis`` extra (ImageAnalysis + ScanAnalysis) is
        not installed.
    """
    import matplotlib

    # The contract travels with the behaviour: ScanAnalysis's renderers
    # use pyplot, and a GUI backend picked lazily on a worker thread is
    # a crash — pin Agg (idempotent) before the import that needs it,
    # whatever the entry point (``__main__`` pins it too, earlier).
    matplotlib.use("Agg")
    from image_analysis.config import load_diagnostic
    from scan_analysis.config import create_scan_analyzer

    diag = load_diagnostic(analyzer_id, config_dir=config_dir)
    return create_scan_analyzer(diag)


@dataclasses.dataclass
class AnalysisJob:
    """One analysis run's record — what the tab polls."""

    uid: str
    analyzer_id: str
    state: str = QUEUED
    submitted: float = dataclasses.field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    #: Paths returned by ``run_analysis`` — relative to the analysis
    #: folder when they live under it (servable), verbatim otherwise
    #: (labels, or files elsewhere: shown, never served).
    artifacts: list[str] = dataclasses.field(default_factory=list)
    error: Optional[str] = None
    #: Formatted log records emitted by the run (last N lines) — what the
    #: process's logging levels admit (the portal's ``--log-level``);
    #: the capture never changes a logger's level.
    log: list[str] = dataclasses.field(default_factory=list)

    def to_json(self) -> dict:
        """The JSON shape the ``/api/run/{uid}/analysis`` endpoints emit."""
        return dataclasses.asdict(self)


class RunInProgress(RuntimeError):
    """A second run was requested while one is running for the scan."""

    def __init__(self, job: AnalysisJob):
        super().__init__(f"{job.analyzer_id} is running for {job.uid}")
        self.job = job


#: Logger-name prefixes that are the portal's own traffic, never the
#: run's: excluded from a job's captured log.
_CAPTURE_EXCLUDE = ("geecs_portal", "uvicorn", "httpx", "httpcore", "asyncio")
#: Thread-name prefixes that serve requests / warm caches in this
#: process — a record from one of them during a run is a concurrent
#: browser's (the Images tab's ephemeral path logs under
#: ``image_analysis.*`` too), not the run's. Analyzer pool threads are
#: ``ThreadPoolExecutor-*``; the worker itself is ``portal-analysis_*``.
_CAPTURE_EXCLUDE_THREADS = ("AnyIO worker thread", "MainThread", "warm-", "uvicorn")


class _RunLogCapture(logging.Handler):
    """Capture every record emitted during a run, minus the portal's own.

    Jobs are serialised on one worker, so the window is the run —
    including the per-shot lines the analyzers emit from their own
    thread pools (a worker-thread-id filter would drop those) — minus
    what the request threads and cache warmers emit meanwhile (a
    concurrent browser's traffic, filtered by thread name). Records
    from process-pool children never reach this process.
    """

    def __init__(self, max_lines: int):
        super().__init__(level=logging.DEBUG)
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(_CAPTURE_EXCLUDE) or str(
            record.threadName
        ).startswith(_CAPTURE_EXCLUDE_THREADS):
            return
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001 — a broken record must not kill the run
            self.handleError(record)


def _relativize(artifact: object, relative_to: Optional[Path]) -> str:
    """Artifact → servable relative path when under *relative_to*, else str."""
    if relative_to is None or not isinstance(artifact, (str, Path)):
        return str(artifact)
    try:
        path = Path(artifact)
        if not path.is_absolute():
            return str(artifact)
        return path.resolve().relative_to(relative_to.resolve()).as_posix()
    except (ValueError, OSError):
        return str(artifact)


class AnalysisRunner:
    """One worker thread; one job per scan at a time; records kept in memory.

    Parameters
    ----------
    max_log_lines : int
        How many trailing log lines each job keeps.
    """

    def __init__(self, *, max_log_lines: int = 400):
        self._jobs: dict[tuple[str, str], AnalysisJob] = {}
        self._lock = threading.Lock()
        self._max_log_lines = max_log_lines
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="portal-analysis"
        )

    def job(self, uid: str, analyzer_id: str) -> Optional[AnalysisJob]:
        """The record for (*uid*, *analyzer_id*), or None when never run."""
        with self._lock:
            return self._jobs.get((uid, analyzer_id))

    def jobs_for(self, uid: str) -> dict[str, AnalysisJob]:
        """All records for *uid*, keyed by analyzer id."""
        with self._lock:
            return {
                analyzer_id: job
                for (job_uid, analyzer_id), job in self._jobs.items()
                if job_uid == uid
            }

    def running_for(self, uid: str) -> Optional[AnalysisJob]:
        """The running job for *uid*, if any (the one-per-scan gate)."""
        with self._lock:
            return self._running_locked(uid)

    def _running_locked(self, uid: str) -> Optional[AnalysisJob]:
        for (job_uid, _), job in self._jobs.items():
            if job_uid == uid and job.state in ACTIVE:
                return job
        return None

    def start(
        self,
        uid: str,
        analyzer_id: str,
        run: Callable[[], Optional[list]],
        *,
        relative_to: Optional[Path] = None,
    ) -> AnalysisJob:
        """Submit *run* for (*uid*, *analyzer_id*); return its fresh record.

        Parameters
        ----------
        uid : str
            The run's catalog uid.
        analyzer_id : str
            The diagnostic ID (the record key with *uid*).
        run : callable
            Executes the analysis and returns the artifact list (the
            factory + ``run_analysis`` + ``cleanup`` composition is the
            caller's — see :func:`run_scan_analyzer`).
        relative_to : Path, optional
            The analysis folder: artifacts under it are recorded
            relative to it so the artifact endpoint can serve them.

        Raises
        ------
        RunInProgress
            When a job is already running for *uid* — one per scan.
        RuntimeError
            After :meth:`shutdown` — the app is stopping.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("analysis runner is shutting down")
            running = self._running_locked(uid)
            if running is not None:
                raise RunInProgress(running)
            job = AnalysisJob(uid=uid, analyzer_id=analyzer_id)
            self._jobs[(uid, analyzer_id)] = job
        self._executor.submit(self._execute, job, run, relative_to)
        return job

    def _execute(
        self,
        job: AnalysisJob,
        run: Callable[[], Optional[list]],
        relative_to: Optional[Path],
    ) -> None:
        capture = _RunLogCapture(self._max_log_lines)
        root = logging.getLogger()
        root.addHandler(capture)
        job.started = time.time()
        job.state = RUNNING
        # The final state is assigned LAST (after log/finished/artifacts)
        # so a poller that sees an inactive state sees a complete record.
        final = FAILED
        reraise: Optional[BaseException] = None
        try:
            artifacts = run()
            if artifacts is None:
                # run_analysis's "inputs missing" return (no s-file / ini /
                # scan parameter): a skip, not a success — the worklist
                # runner's own no_data mapping.
                job.error = "analysis skipped: inputs missing (see log)"
                final = NO_DATA
            else:
                job.artifacts = [_relativize(a, relative_to) for a in artifacts]
                final = DONE
        except BaseException as exc:  # noqa: BLE001 — every outcome becomes a record
            job.error = f"{type(exc).__name__}: {exc}"
            final = NO_DATA if _is_no_data(exc) else FAILED
            if not isinstance(exc, Exception):
                # SystemExit / KeyboardInterrupt from an analyzer: record
                # it (never a record stuck at ``running`` → permanent 409),
                # then let the executor see it.
                reraise = exc
            logger.log(
                logging.INFO if final == NO_DATA else logging.WARNING,
                "analysis %s on %s %s: %s",
                job.analyzer_id,
                job.uid,
                final,
                job.error,
            )
        finally:
            root.removeHandler(capture)
            job.log = list(capture.lines)
            job.finished = time.time()
            job.state = final
        if reraise is not None:
            raise reraise

    def shutdown(self) -> None:
        """Refuse new jobs and stop the worker (app shutdown; tests).

        A job already running cannot be interrupted — the interpreter
        joins the worker at exit, so a service stop waits for it (see
        ``DEPLOYMENT.md`` on the stop timeout). Logged so the operator
        knows what the wait is.
        """
        with self._lock:
            self._closed = True
            active = [j for j in self._jobs.values() if j.state in ACTIVE]
        for job in active:
            logger.warning(
                "shutdown with analysis %s on %s still %s — waiting for it",
                job.analyzer_id,
                job.uid,
                job.state,
            )
        self._executor.shutdown(wait=False, cancel_futures=True)


def _is_no_data(exc: BaseException) -> bool:
    """True for ScanAnalysis's ``DataUnavailableWarning`` (device folder missing/empty).

    Matched by name so the runner module imports nothing from
    ScanAnalysis at import time (the extra is optional).
    """
    return type(exc).__name__ == "DataUnavailableWarning"


def run_scan_analyzer(
    factory: AnalyzerFactory, analyzer_id: str, config_dir: Path, scan_tag: object
) -> Optional[list]:
    """Build, run and clean up one analyzer — the body of a job.

    ``cleanup()`` runs whether or not ``run_analysis`` raised: it is the
    task runner's duty in the queue path and ours here.
    """
    analyzer = factory(analyzer_id, config_dir)
    try:
        return analyzer.run_analysis(scan_tag)
    finally:
        analyzer.cleanup()


def analysis_folder_for(scan_folder: Path) -> Path:
    """``{day}/analysis/ScanNNN`` for a ``{day}/scans/ScanNNN`` folder.

    Pure path construction — the same ``parts[-2] = "analysis"`` rule
    as ``ScanPaths.get_analysis_folder`` without its create-if-missing
    side effect (the analyzer creates it when it runs; the portal only
    reads it).
    """
    return scan_folder.parents[1] / "analysis" / scan_folder.name


def contained_artifact(analysis_folder: Path, relative: str) -> Optional[Path]:
    """Resolve *relative* inside *analysis_folder*, or None when it escapes.

    The artifact endpoint's containment check: the resolved path must
    stay under the resolved analysis folder (symlinks included) and be
    an existing regular file. Absolute, ``..``-climbing and dangling
    paths all return None — never an exception.
    """
    if not relative or Path(relative).is_absolute():
        return None
    try:
        root = analysis_folder.resolve()
        candidate = (analysis_folder / relative).resolve()
        candidate.relative_to(root)
    except (ValueError, OSError):
        return None
    return candidate if candidate.is_file() else None


#: Artifact types the artifact endpoint renders inline (raster only —
#: SVG can carry script and is served as a download like everything else).
INLINE_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})


#: ScanAnalysis output-name conventions (``renderers/*``): per-bin
#: visuals are ``<name>_<bin>_processed_visual.<ext>``; the average /
#: grid / summary / animation files are the scan-level figures.
_BIN_FILE = re.compile(r"_(\d+)_processed_visual\.[A-Za-z0-9]+$")
_SUMMARY_MARKERS = (
    "_average_processed_visual.",
    "_averaged_image_grid.",
    "_summary_",
    "_animation.",
)


def classify_artifact(artifact: str) -> tuple[str, Optional[int]]:
    """``("bin", n)`` / ``("summary", None)`` / ``("other", None)`` from the name.

    The tab shows summaries automatically and steps through bins one
    at a time (owner ruling 2026-09-02); the split is by ScanAnalysis's
    own filename conventions, decided here rather than in page JS.
    """
    name = Path(artifact).name
    match = _BIN_FILE.search(name)
    if match:
        return "bin", int(match.group(1))
    if any(marker in name for marker in _SUMMARY_MARKERS) or name.lower().endswith(
        ".gif"
    ):
        return "summary", None
    return "other", None


def describe_artifact(analysis_folder: Path, artifact: str) -> dict:
    """The wire shape of one artifact: ``{path, servable, inline, kind, bin}``.

    ``servable`` = it resolves to a file inside the analysis folder (the
    artifact endpoint would serve it); ``inline`` = a raster image the
    tab may show in an ``<img>``. Labels and files elsewhere are neither
    — the tab shows them as text. ``kind``/``bin`` per
    :func:`classify_artifact`. Decided HERE so the page never re-derives
    the policy from a path's shape.
    """
    servable = contained_artifact(analysis_folder, artifact) is not None
    kind, bin_number = classify_artifact(artifact)
    return {
        "path": artifact,
        "servable": servable,
        "inline": servable and Path(artifact).suffix.lower() in INLINE_IMAGE_SUFFIXES,
        "kind": kind,
        "bin": bin_number,
    }


_KIND_ORDER = {"summary": 0, "other": 1, "bin": 2}


def describe_artifacts(analysis_folder: Path, artifacts: list[str]) -> list[dict]:
    """Describe + order for the tab: summaries, then others, then bins by number."""
    described = [describe_artifact(analysis_folder, a) for a in artifacts]
    return sorted(
        described, key=lambda d: (_KIND_ORDER[d["kind"]], d["bin"] or 0, d["path"])
    )


def list_artifacts(
    analysis_folder: Path, output_name: str, *, limit: int = 200
) -> list[str]:
    """Files under the analyzer's output directory, relative to the analysis folder.

    ScanAnalysis writes per-analyzer outputs under
    ``analysis/ScanNNN/<output_name>/…`` — this lists them (sorted,
    capped) so a page loaded after a portal restart still shows what
    an earlier run produced.
    """
    out_dir = analysis_folder / output_name
    if not out_dir.is_dir():
        return []
    try:
        files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    except OSError:
        return []
    return [p.relative_to(analysis_folder).as_posix() for p in files[:limit]]
