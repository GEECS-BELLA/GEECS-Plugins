"""The GEECS MCP server — FastMCP instance + tool registration.

One general server, domains as subpackages: ``scans/`` today; future
domains (health, db, logs, analysis) register the same way.

The osprey house pattern (its native bluesky server): a module-level
:class:`FastMCP`, tool modules that self-register via ``@mcp.tool()``,
and a :func:`create_server` that imports them.  Usage::

    python -m geecs_mcp

Osprey's ``profile.yml`` points a stdio ``command:`` at that invocation;
permission lists import :mod:`geecs_mcp.tool_names` symbols.
"""

from __future__ import annotations

import logging

from fastmcp import FastMCP

logger = logging.getLogger("geecs_mcp")

mcp = FastMCP(
    "geecs",
    instructions=(
        "Access to the GEECS scan service. Read tools: manager/queue "
        "status, recent scan history, completed-run results from the Tiled "
        "archive, the experiment's config catalogs (save sets, trigger "
        "profiles, presets, scan variables, actions), full dry-run "
        "validation of a ScanRequest. Control verbs: "
        "submit_scan (one scan in flight, capped, preflight warnings need "
        "explicit acknowledgement), stop_scan (graceful; another client's "
        "scan needs force=true and an operator's say-so), clear_queue "
        "(the only remover), scan_progress. Analysis domain: "
        "get_scan_analysis (task statuses + output tree) and "
        "get_scan_figure (a figure REFERENCE — metadata plus a fetch "
        "URL served by this same server; thumbnail=true for a bounded "
        "inline preview, never pull full figures through context), "
        "run_scan_analysis (execute a ScanAnalysis diagnostic or group "
        "for one existing scan, detached — poll get_scan_analysis for "
        "progress; analyzer/group names come from list_analyzers / "
        "list_analysis_groups). "
        "Names must come from the listing tools — never invent catalog "
        "names."
    ),
)


def create_server() -> FastMCP:
    """Register every tool module and return the server."""
    from geecs_mcp.analysis import (  # noqa: F401 — self-register
        read_tools as analysis_read_tools,
        run_tools as analysis_run_tools,
    )
    from geecs_mcp.scans import (  # noqa: F401 — self-register
        control_tools,
        read_tools,
    )

    logger.info("geecs MCP server initialised (v0 read + v1 control tools)")
    return mcp
