"""The analysis worker: a long-running service that batch-runs analysis groups over scans.

One job = one analysis group over one day's scans (or some of them),
expanded into the task queue's worklist and run task by task under its
claims and status records (``scans/ScanNNN/analysis_status/``), so a
finished task is skipped unless the job asks to redo it.  ``jobs`` is the
pure-Python runner, ``app`` its HTTP API (the Data Portal's day page is
one client), ``__main__`` the ``scan-analysis-worker`` command.

A watch (automatic analysis of new scans, on this queue) submits a job whenever its
day's newly closed scans have work for its ticked analyzers.  A trigger
from the engine's run-stop document is a later step; it would submit jobs
like any client.
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
