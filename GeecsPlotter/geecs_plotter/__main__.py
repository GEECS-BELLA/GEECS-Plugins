"""Run the plotter: ``geecs-plotter`` / ``python -m geecs_plotter``.

Binds ``127.0.0.1`` by default — localhost first; deployment is a later
PR. The data root and experiment come from ``config.ini`` unless given.
"""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    """Parse CLI arguments, build the app, serve."""
    parser = argparse.ArgumentParser(description="GEECS plotter")
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8600, help="HTTP port")
    parser.add_argument(
        "--experiment",
        default="",
        help="the experiment folder to read (default: config.ini [Experiment] expt)",
    )
    parser.add_argument(
        "--root-path",
        default="",
        help="URL prefix behind a prefix-preserving reverse proxy",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    import uvicorn

    from geecs_plotter.app import create_app

    uvicorn.run(
        create_app(args.experiment, root_path=args.root_path),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
