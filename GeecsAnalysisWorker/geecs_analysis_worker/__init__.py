"""The analysis worker's HTTP service (``scan-analysis-worker``).

The job runner is ScanAnalysis' (``scan_analysis.worker``: one job = one
analysis group over one day's scans, run task by task through the task
queue).  This package is the web side of it and nothing else: ``app`` is
the HTTP API over a ``Worker``, ``__main__`` the command.  Clients (the
Data Portal's day page is one) reach it over HTTP alone and never import
it.
"""

from geecs_analysis_worker.app import JobRequest, WatchRequest, create_worker_app

__all__ = ["JobRequest", "WatchRequest", "create_worker_app"]
