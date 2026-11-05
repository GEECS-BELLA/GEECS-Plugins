"""The analysis worker's jobs: one analysis group over one day's scans.

A job is a request — *run group G over day D (or some of its scans), and
redo finished tasks or not* — expanded into the task queue's worklist
(:func:`scan_analysis.task_queue.build_worklist`) when it starts, and run
one task at a time through :func:`~scan_analysis.task_queue.run_task`,
which claims the task, writes its status record
(``scans/ScanNNN/analysis_status/<analyzer>.yaml``) and runs the analyzer.
So a finished task is skipped by the next job unless it asks to redo, two
workers never run one task twice, and every other reader of those records
(MCP, the portal) sees the same states.

Jobs run in submission order on one thread; a cancel takes effect between
tasks (an analyzer already running finishes).  The job list lives in memory
— the status records on the share are the durable result.

A scan still being recorded is left out: only a scan with closed-run
evidence (:func:`geecs_data_utils.data.sfile.run_closed_evidence`, its
``ScanData`` table or s-file) is analysed.  Nothing here creates a scan
folder; the day's ``scans/`` folder is listed, never made.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import itertools
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from geecs_data_utils import ScanPaths, ScanTag
from geecs_data_utils.data.sfile import run_closed_evidence

from scan_analysis.task_queue import (
    analyzer_task_id,
    build_worklist,
    load_analyzers_from_config,
    read_statuses,
    run_task,
)

logger = logging.getLogger(__name__)

__all__ = ["Job", "JobSpec", "TaskRecord", "Worker", "day_scan_tags"]

#: Job states.  ``queued`` → ``running`` → ``done`` | ``cancelled`` | ``failed``
#: (``failed`` is the job itself — an unknown group, a missing day — never a task).
JOB_ACTIVE = ("queued", "running")

_SCAN_FOLDER = re.compile(r"Scan(\d+)")


def day_scan_tags(
    day: date, experiment: str, *, base_directory: Optional[Path] = None
) -> list[ScanTag]:
    """Every ``ScanNNN`` folder of *day*, as scan tags, by number.

    Raises ``FileNotFoundError`` when the day has no ``scans/`` folder.
    Lists only; never creates.
    """
    probe = ScanTag(
        year=day.year, month=day.month, day=day.day, number=1, experiment=experiment
    )
    scans = ScanPaths.get_scan_folder_path(
        tag=probe, base_directory=base_directory
    ).parent
    if not scans.is_dir():
        raise FileNotFoundError(f"no scans folder for {day.isoformat()} ({scans})")
    numbers = sorted(
        int(m.group(1))
        for p in scans.iterdir()
        if p.is_dir() and (m := _SCAN_FOLDER.fullmatch(p.name))
    )
    return [
        ScanTag(
            year=day.year, month=day.month, day=day.day, number=n, experiment=experiment
        )
        for n in numbers
    ]


def release_memory() -> None:
    """Hand a finished task's freed memory back to the OS (glibc only)."""
    gc.collect()
    with contextlib.suppress(OSError, AttributeError):  # no glibc: macOS, Windows
        ctypes.CDLL("libc.so.6").malloc_trim(0)


@dataclass(frozen=True)
class JobSpec:
    """What a job asks for."""

    day: date
    group: str
    experiment: str
    scans: Optional[tuple[int, ...]] = None
    redo: bool = False


@dataclass
class TaskRecord:
    """One (scan, analyzer) task of a job and how it ended."""

    scan: int
    analyzer: str
    #: ``queued`` | ``running`` | ``done`` | ``no_data`` | ``failed`` |
    #: ``skipped`` (another runner held it) | ``cancelled``
    state: str = "queued"
    error: Optional[str] = None
    seconds: Optional[float] = None

    def to_json(self) -> dict:
        """The wire shape of one task."""
        return {
            "scan": self.scan,
            "analyzer": self.analyzer,
            "state": self.state,
            "error": self.error,
            "seconds": self.seconds,
        }


@dataclass
class Job:
    """A submitted job and its progress."""

    id: str
    spec: JobSpec
    state: str = "queued"
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    tasks: list[TaskRecord] = field(default_factory=list)
    #: scans left out because they are still being recorded
    open_scans: list[int] = field(default_factory=list)
    #: tasks the worklist left out because they already finished
    already_done: int = 0
    cancel_requested: bool = False

    def counts(self) -> dict[str, int]:
        """Tasks per state."""
        out: dict[str, int] = {}
        for task in self.tasks:
            out[task.state] = out.get(task.state, 0) + 1
        return out

    def to_json(self, *, with_tasks: bool = True) -> dict:
        """The wire shape of the job (its task list unless ``with_tasks=False``)."""
        current = next((t for t in self.tasks if t.state == "running"), None)
        body = {
            "id": self.id,
            "day": self.spec.day.isoformat(),
            "group": self.spec.group,
            "experiment": self.spec.experiment,
            "scans": list(self.spec.scans) if self.spec.scans is not None else None,
            "redo": self.spec.redo,
            "state": self.state,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "total": len(self.tasks),
            "counts": self.counts(),
            "current": current.to_json() if current else None,
            "open_scans": self.open_scans,
            "already_done": self.already_done,
            "cancel_requested": self.cancel_requested,
        }
        if with_tasks:
            body["tasks"] = [t.to_json() for t in self.tasks]
        return body


#: ``loader(group, config_dir=...) -> [analyzer]`` — the task queue's own loader by default.
AnalyzerLoader = Callable[..., list]
#: ``runner(priority, tag, analyzer, base_directory=...) -> state | None`` — ``run_task`` by default.
TaskRunner = Callable[..., Optional[str]]


class Worker:
    """Runs jobs, one at a time, on its own thread.

    Parameters
    ----------
    config_dir : Path
        The analysis configs tree (``scan_analysis_configs``): its groups
        name the analyzers a job runs.
    default_experiment : str
        The experiment a job names when it gives none.
    base_directory : Path, optional
        The data root; default the configured one (``config.ini``).
    loader, runner : callable, optional
        Seams for tests: how a group becomes analyzers, how one task runs.
    """

    def __init__(
        self,
        config_dir: Path,
        *,
        default_experiment: str,
        base_directory: Optional[Path] = None,
        loader: AnalyzerLoader = load_analyzers_from_config,
        runner: TaskRunner = run_task,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.default_experiment = default_experiment
        self.base_directory = base_directory
        self._loader = loader
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._ids = itertools.count(1)
        self._thread: Optional[threading.Thread] = None
        self._stopping = False

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the job thread (idempotent)."""
        if self._thread is None or not self._thread.is_alive():
            self._stopping = False
            self._thread = threading.Thread(
                target=self._loop, name="analysis-worker", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Cancel every job and stop the thread after the running task."""
        self._stopping = True
        with self._lock:
            for job in self._jobs.values():
                if job.state in JOB_ACTIVE:
                    job.cancel_requested = True
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout)

    # ------------------------------------------------------------ requests

    def submit(self, spec: JobSpec) -> Job:
        """Queue a job; it runs after the ones before it."""
        with self._lock:
            job = Job(id=f"j{next(self._ids)}", spec=spec)
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def jobs(self) -> list[Job]:
        """Every job, newest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def get(self, job_id: str) -> Optional[Job]:
        """One job by id, or ``None``."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[Job]:
        """Ask a job to stop: before its next task, or before it starts."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.state in JOB_ACTIVE:
                job.cancel_requested = True
            return job

    def plan(self, spec: JobSpec) -> Job:
        """What a job would run now, without running it (a job never queued)."""
        job = Job(id="plan", spec=spec)
        self._expand(job)
        return job

    def groups(self) -> list[dict]:
        """The configs tree's loadable groups: ``id`` (``PW/frog_only``), name, members.

        Each file once, by its path under ``groups/`` (unique, unlike the
        stem), validated and resolved as a run would load it; a group that
        would fail is left out with a log line.
        """
        from scan_analysis.config import discover_groups, load_analysis_group

        try:
            index = discover_groups(self.config_dir)
        except (FileNotFoundError, OSError):
            return []
        groups_dir = self.config_dir / "groups"
        out = []
        for path in sorted(set(index.values()), key=str):
            try:
                group = load_analysis_group(path, config_dir=self.config_dir)
            except Exception as exc:  # noqa: BLE001 — one bad group must not hide the rest
                logger.info("group %s not loadable: %s", path, exc)
                continue
            out.append(
                {
                    "id": Path(path).relative_to(groups_dir).with_suffix("").as_posix(),
                    "name": group.name,
                    "description": group.description or "",
                    "members": [r.id for r in group.analyzers],
                }
            )
        return out

    # ------------------------------------------------------------ the thread

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None or self._stopping:
                return
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._run(job)
            except Exception as exc:  # a job's failure is its record (logged)
                logger.exception("job %s failed", job.id)
                job.error = f"{type(exc).__name__}: {exc}"
                job.state = "failed"
                job.finished = time.time()

    def _expand(self, job: Job) -> list[tuple]:
        """The job's worklist; fills its tasks, open scans and done count."""
        spec = job.spec
        tags = day_scan_tags(
            spec.day, spec.experiment, base_directory=self.base_directory
        )
        if spec.scans is not None:
            wanted = set(spec.scans)
            tags = [t for t in tags if t.number in wanted]
        closed = []
        for tag in tags:
            folder = ScanPaths.get_scan_folder_path(
                tag=tag, base_directory=self.base_directory
            )
            if run_closed_evidence(folder) is not None:
                closed.append(tag)
            else:
                job.open_scans.append(tag.number)
        analyzers = self._loader(spec.group, config_dir=self.config_dir)
        work = build_worklist(
            closed,
            analyzers,
            base_directory=self.base_directory,
            rerun_completed=spec.redo,
            rerun_failed=spec.redo,
        )
        job.tasks = [
            TaskRecord(scan=tag.number, analyzer=analyzer_task_id(a))
            for _, tag, a in work
        ]
        job.already_done = len(closed) * len(analyzers) - len(work)
        return work

    def _run(self, job: Job) -> None:
        if job.cancel_requested:
            job.state = "cancelled"
            job.finished = time.time()
            return
        job.state = "running"
        job.started = time.time()
        work = self._expand(job)
        for (priority, tag, analyzer), record in zip(work, job.tasks):
            if job.cancel_requested:
                for rest in job.tasks:
                    if rest.state == "queued":
                        rest.state = "cancelled"
                break
            record.state = "running"
            began = time.monotonic()
            try:
                state = self._runner(
                    priority, tag, analyzer, base_directory=self.base_directory
                )
            # run_task records its own failures; a seam might raise instead
            except Exception as exc:  # noqa: BLE001
                state, record.error = "failed", f"{type(exc).__name__}: {exc}"
            record.seconds = round(time.monotonic() - began, 1)
            record.state = state or "skipped"
            if record.state == "failed" and record.error is None:
                record.error = self._recorded_error(tag, record.analyzer)
            release_memory()
        job.state = "cancelled" if job.cancel_requested else "done"
        job.finished = time.time()

    def _recorded_error(self, tag: ScanTag, analyzer_id: str) -> Optional[str]:
        """The failed task's error, as its status record holds it."""
        folder = ScanPaths.get_scan_folder_path(
            tag=tag, base_directory=self.base_directory
        )
        for status in read_statuses(folder):
            if status.analyzer_id == analyzer_id:
                return status.error
        return None
