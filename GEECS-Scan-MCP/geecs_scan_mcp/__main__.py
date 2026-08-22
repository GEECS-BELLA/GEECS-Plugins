"""Entry point: ``python -m geecs_scan_mcp`` runs the stdio server."""

from __future__ import annotations

import logging


def main() -> None:
    """Run the server on stdio (FastMCP's default transport)."""
    logging.basicConfig(level=logging.INFO)
    from geecs_scan_mcp.server import create_server

    create_server().run()


if __name__ == "__main__":
    main()
