"""``geecs-scanner`` — serve the web scanner.

Real mode builds the manager client from ``~/.config/geecs_python_api/config.ini``
(``[qserver]``) and the configs resolver from the scanner-configs root, and
starts the two stream consumers.  ``--demo`` serves the same API over an
in-memory manager that runs scans by itself — nothing is configured,
nothing is contacted, and the page behaves as it will against the worker.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from geecs_scanner import __version__


def build_service(
    *,
    experiment: str,
    identity: str,
    demo: bool,
    demo_period: float,
    portal_url: str = "",
):
    """Assemble the service for real or demo mode."""
    from geecs_scanner.service import ProgressCache, ScannerService

    streams = ProgressCache()
    if demo:
        from geecs_scanner.service.demo import (
            DemoQueueClient,
            DemoResolver,
            demo_preflight,
        )

        client = DemoQueueClient(streams, period=demo_period, user=identity)
        return ScannerService(
            client,
            DemoResolver(),
            experiment=experiment or "Demo",
            identity=identity,
            streams=streams,
            preflight=demo_preflight,
            version=__version__,
            portal_url=portal_url,
        )
    from geecs_bluesky.config_resolver import ConfigsRepoResolver
    from geecs_bluesky.qs_client import make_queue_client

    client = make_queue_client(experiment, user=identity)
    streams.ensure_started(client.doc_addr, client.info_addr)
    return ScannerService(
        client,
        ConfigsRepoResolver(experiment),
        experiment=experiment,
        identity=identity,
        streams=streams,
        version=__version__,
        portal_url=portal_url,
    )


def main() -> None:
    """Parse arguments, build the app, serve."""
    parser = argparse.ArgumentParser(description="GEECS web scanner")
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=8300, help="HTTP port")
    parser.add_argument(
        "--experiment",
        default="",
        help="the experiment this scanner serves (a site value: GEECS_EXPERIMENT in site.env)",
    )
    parser.add_argument(
        "--identity",
        default=f"geecs-scanner {__version__}",
        help="what the manager records as the submitting user on every queue item",
    )
    parser.add_argument(
        "--root-path",
        default="",
        help="URL prefix behind a reverse proxy (e.g. /scan); X-Forwarded-Prefix overrides per request",
    )
    parser.add_argument(
        "--portal-url",
        default="",
        help="the Data Portal's base URL for the run-page links (a site value; "
        "behind the front door it is the /portal prefix); empty hides them",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="serve over an in-memory manager that runs scans by itself; contacts nothing",
    )
    parser.add_argument(
        "--demo-period", type=float, default=1.0, help="seconds per shot in demo mode"
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.demo and not args.experiment:
        parser.error("--experiment is required (or pass --demo)")

    from geecs_scanner.web import create_app

    service = build_service(
        experiment=args.experiment,
        identity=args.identity,
        demo=args.demo,
        demo_period=args.demo_period,
        portal_url=args.portal_url,
    )
    app = create_app(service, root_path=args.root_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
