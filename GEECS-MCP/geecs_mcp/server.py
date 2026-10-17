"""The GEECS MCP server — FastMCP instance + tool registration.

One general server, domains as subpackages: ``scans/`` (the scanner's
HTTP API), ``analysis/`` (the portal's), ``logbook/`` (the logbook's,
with figures sent through the portal).

The osprey house pattern (its native bluesky server): a module-level
:class:`FastMCP`, tool modules that self-register via ``@mcp.tool()``,
and a :func:`create_server` that imports them.  Usage::

    python -m geecs_mcp

Osprey's ``profile.yml`` points a stdio ``command:`` at that invocation;
permission lists import :mod:`geecs_mcp.tool_names` symbols.
"""

from __future__ import annotations

import importlib
import logging

from fastmcp import FastMCP

logger = logging.getLogger("geecs_mcp")

#: The domain subpackages whose ``tools`` modules carry the verbs.
DOMAINS = ("scans", "analysis", "logbook")

mcp = FastMCP(
    "geecs",
    instructions=(
        "The GEECS facility server: a thin client of the web scanner, the "
        "Data Portal and the logbook. Reads: scan_status, scan_progress, "
        "scan_history (queue rows with item_uid, scan_numbers, run_uids), "
        "list_scan_configs (the catalogs — names must come from here, never "
        "invented), wait_for_scan (blocks until an item finishes; a timeout "
        "is a result with finished=false, call again), get_scan_analysis / "
        "get_scan_figure / list_analyzers / list_analysis_groups. Writes "
        "(every one takes the supervising operator's name): submit_preset "
        "(the scanner's preflight runs; its questions come back verbatim "
        "under needs_acknowledgement and are never acknowledged here — show "
        "them to the operator, then pass the check names back), move (the "
        "scanner's mv plan, idle-only, returns the readback), resume_scan, "
        "clear_queue, run_analysis (one analysis document on one scan "
        "through the portal; wait_for_analysis follows it), "
        "post_logbook_entry (a draft in the scans book; figures are artifact "
        "references the portal attaches — ok once the entry exists, so never "
        "post twice). Numbers come "
        "from get_analysis_artifact (a run's JSON/text sidecar, bounded) "
        "and get_scan_scalars (per-bin statistics of up to four columns); "
        "figures from get_scan_figure (a reference, thumbnail=true for a "
        "bounded look). The halt family — stop_scan, pause_scan — always "
        "works. Every refusal carries the service's own words in `message` "
        "and a kind to branch on."
    ),
)


def create_server() -> FastMCP:
    """Register every tool module and return the server."""
    for domain in DOMAINS:  # each tools module self-registers on import
        importlib.import_module(f"geecs_mcp.{domain}.tools")

    logger.info(
        "geecs MCP server initialised (a client of the scanner, the portal and the logbook)"
    )
    return mcp
