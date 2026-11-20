"""``scan-analysis-worker``: serve the analysis worker.

::

    scan-analysis-worker --configs <scan_analysis_configs> --experiment <name> [--port 8900]

Binds 127.0.0.1 unless ``--host`` says otherwise (the API is
unauthenticated).  ``--host`` and ``--port`` default to the environment's
``GEECS_ANALYSIS_WORKER_BIND`` and ``GEECS_ANALYSIS_WORKER_PORT`` (the
service unit's ``site.env``) when those are set.  The data root is the
client ``config.ini``'s, as for every analysis run.  The worker starts
idle: it runs the jobs it is sent and watches a day only once asked to.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


#: The worker's port when neither ``--port`` nor the environment names one.
DEFAULT_PORT = 8900


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and serve until stopped."""
    parser = argparse.ArgumentParser(
        prog="scan-analysis-worker", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--configs", type=Path, required=True, help="the scan_analysis_configs tree"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="the experiment folder a job names by default",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("GEECS_ANALYSIS_WORKER_BIND") or "127.0.0.1",
        help="bind address (default: GEECS_ANALYSIS_WORKER_BIND, else 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        # a string default goes through type=int, so a malformed value is
        # argparse's usage error, not a traceback
        default=os.environ.get("GEECS_ANALYSIS_WORKER_PORT") or DEFAULT_PORT,
        help="HTTP port (default: GEECS_ANALYSIS_WORKER_PORT, else 8900)",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    # Before any figure is drawn: ScanAnalysis' analyzers draw with pyplot,
    # and the one job thread is the only pyplot user in this process.
    import matplotlib

    matplotlib.use("Agg")
    import uvicorn

    from scan_analysis.worker import Worker

    from geecs_analysis_worker.app import create_worker_app

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.configs.is_dir():
        parser.error(f"--configs {args.configs} is not a folder")
    worker = Worker(args.configs, default_experiment=args.experiment)
    uvicorn.run(
        create_worker_app(worker),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
