"""The general GEECS MCP server — AI-agent access to GEECS.

One server process, domains as subpackages: ``scans/`` (v0 read tools +
v1 control verbs — submit/stop/clear/progress) today; future domains
(health, db, logs, analysis) register on the same server.  See
``CLAUDE.md`` for the domain roadmap and the safety doctrine.
"""

__version__ = "0.4.0"
