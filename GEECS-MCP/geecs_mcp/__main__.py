"""Entry point: ``python -m geecs_mcp`` runs the server.

Transports:

- ``stdio`` (default) — the per-machine mode: osprey's ``profile.yml``
  ``command:`` launches this process per session.
- ``http`` — the central-deployment mode (one server on the qserver
  box, every osprey machine's profile points ``url:`` at it; see
  ``deploy/DEPLOYMENT.md``).  Interim posture matches the manager: no
  transport auth, lab-network-internal (#660 covers the fleet-wide
  answer).
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("geecs_mcp.main")


def warm_progress_stream() -> None:
    """Start the ``scan_progress`` stream consumers before the first run (#685).

    HTTP mode only: a long-lived service must be consuming BEFORE its
    first run's start document passes — the lazy start behind
    ``scan_progress`` (stdio keeps it — see ``_stream_snapshot``) left
    the first scan after a service restart with no scan number or counts.
    Best-effort like the cache itself: nothing here can stop the server
    from coming up.  The cache logs what it did (consuming from which
    address, or latched unconfigured); the guard below only covers a
    client that will not build — a path the seams underneath promise not
    to take, kept so the boot can never die on it.
    """
    try:
        from geecs_mcp import runtime
        from geecs_mcp.scans import progress_stream

        progress_stream.start_for_client(runtime.get_queue_client())
    except Exception as exc:
        logger.warning(
            "progress stream not warmed at startup (%s) — scan_progress "
            "will start it lazily on first call",
            exc,
        )


def main() -> None:
    """Parse transport options and run the server."""
    parser = argparse.ArgumentParser(description="GEECS MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="stdio (per-session process, default) or http (central service)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="http bind host")
    parser.add_argument("--port", type=int, default=8100, help="http bind port")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from geecs_mcp.server import create_server

    server = create_server()
    if args.transport == "http":
        warm_progress_stream()
        server.run(transport="http", host=args.host, port=args.port)
    else:
        server.run()


if __name__ == "__main__":
    main()
