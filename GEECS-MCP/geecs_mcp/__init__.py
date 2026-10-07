"""The general GEECS MCP server — AI-agent access to GEECS.

One server process, domains as subpackages: ``scans/`` (v0 read tools +
v1 control verbs — submit/stop/clear/progress) today; future domains
(health, db, logs, analysis) register on the same server.  See
``CLAUDE.md`` for the domain roadmap and the safety doctrine.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geecs-mcp")
except PackageNotFoundError:  # pragma: no cover — source checkout, not installed
    __version__ = "0.0.0+source"
