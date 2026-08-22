"""GEECS scan MCP server — AI-agent access to the GEECS scan service.

The server exposes MCP verbs over the queueserver client seam
(:mod:`geecs_bluesky.qs_client`), the config resolver, and the Tiled
archive.  v0 is read-only: status, history, results, config discovery,
and request validation — zero write risk.  See ``CLAUDE.md`` for the
verb roadmap (submit/stop in v1) and the safety doctrine.
"""

__version__ = "0.2.0"
