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
        server.run(transport="http", host=args.host, port=args.port)
    else:
        server.run()


if __name__ == "__main__":
    main()
