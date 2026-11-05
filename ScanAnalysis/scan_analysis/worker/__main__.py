"""``scan-analysis-worker``: serve the analysis worker.

::

    scan-analysis-worker --configs <scan_analysis_configs> --experiment <name> [--port 8600]

Binds 127.0.0.1 unless ``--host`` says otherwise (the API is
unauthenticated).  The data root is the client ``config.ini``'s, as for
every analysis run.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    # Before ScanAnalysis: its analyzers draw with pyplot, and the one job
    # thread is the only pyplot user in this process.
    import matplotlib

    matplotlib.use("Agg")
    import uvicorn

    from scan_analysis.worker.app import create_worker_app
    from scan_analysis.worker.jobs import Worker

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
