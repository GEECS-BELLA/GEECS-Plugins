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
        "Read access to the GEECS scan service, plus the halt verbs. Read "
        "tools: manager/queue status, recent scan history, completed-run "
        "results from the Tiled archive, the experiment's config catalogs "
        "(trigger profiles, presets, optimizer configs, scan variables, "
        "actions), scan_progress. Control verbs: stop_scan "
        "(graceful; another client's scan needs force=true and an "
        "operator's say-so), pause_scan / resume_scan (same ownership "
        "etiquette), clear_queue (the only remover). Analysis domain: "
        "get_scan_analysis (task statuses + output tree) and "
        "get_scan_figure (a figure REFERENCE — metadata plus a fetch "
        "URL served by this same server; thumbnail=true for a bounded "
        "inline preview, never pull full figures through context), "
        "run_scan_analysis (execute a ScanAnalysis diagnostic or group "
        "for one existing scan, detached — poll get_scan_analysis for "
        "progress; analyzer/group names come from list_analyzers / "
        "list_analysis_groups). "
        "There is NO submit verb: scans are submitted from the web "
        "scanner, not from here. Names must come from the listing tools "
        "— never invent catalog names."
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

    logger.info("geecs MCP server initialised (read + halt tools; no submit verb)")
    return mcp
