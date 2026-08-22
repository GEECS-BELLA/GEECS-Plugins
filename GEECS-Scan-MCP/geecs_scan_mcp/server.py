"""The GEECS scan MCP server — FastMCP instance + tool registration.

The osprey house pattern (its native bluesky server): a module-level
:class:`FastMCP`, tool modules that self-register via ``@mcp.tool()``,
and a :func:`create_server` that imports them.  Usage::

    python -m geecs_scan_mcp

Osprey's ``profile.yml`` points a stdio ``command:`` at that invocation;
permission lists import :mod:`geecs_scan_mcp.tool_names` symbols.
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP

logger = logging.getLogger("geecs_scan_mcp")

mcp = FastMCP(
    "geecs-scan",
    instructions=(
        "Read-only access to the GEECS scan service (v0): manager/queue "
        "status, recent scan history, completed-run results from the Tiled "
        "archive, the experiment's config catalogs (save sets, trigger "
        "profiles, presets, scan variables, actions), and full dry-run "
        "validation of a ScanRequest. Names must come from "
        "list_scan_configs — never invent catalog names. Submission and "
        "control verbs arrive in v1; today this server cannot start, stop, "
        "or modify anything."
    ),
)


def create_server() -> FastMCP:
    """Register every tool module and return the server."""
    from geecs_scan_mcp.tools import read_tools  # noqa: F401 — self-registers

    logger.info("geecs-scan MCP server initialised (v0 read-only tools)")
    return mcp
