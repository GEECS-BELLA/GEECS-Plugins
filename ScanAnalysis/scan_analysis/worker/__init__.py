"""The analysis worker's job runner: batch-runs analysis groups over scans.

One job = one analysis group over one day's scans (or some of them),
expanded into the task queue's worklist and run task by task under its
claims and status records (``scans/ScanNNN/analysis_status/``), so a
finished task is skipped unless the job asks to redo it.  ``jobs`` is the
whole of it, pure Python.  Its HTTP API and the ``scan-analysis-worker``
command are the GeecsAnalysisWorker package's (``geecs_analysis_worker``):
no web code lives here.

A watch (automatic analysis of new scans, on this queue) submits a job
whenever its day's newly closed scans have work for its ticked analyzers.
It exists only once a caller starts one (``Worker.watch``): a ``Worker``
is built and started with none.  A trigger from the engine's run-stop
document is a later step; it would submit jobs like any client.
"""

from scan_analysis.worker.jobs import (
    Job,
    JobSpec,
    TaskRecord,
    Watch,
    WatchSpec,
    Worker,
    day_scan_tags,
)

__all__ = [
    "Job",
    "JobSpec",
    "TaskRecord",
    "Watch",
    "WatchSpec",
    "Worker",
    "day_scan_tags",
]
