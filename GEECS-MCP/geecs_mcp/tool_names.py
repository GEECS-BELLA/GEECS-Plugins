"""Tool-name constants — the one place tool names are spelled.

The osprey ``bluesky_tool_names.py`` pattern: profile permission lists and
hook matchers import these symbols instead of retyping strings, so a
rename cannot silently strand a permission entry.  The import path
``geecs_mcp.tool_names`` is stable: the HTU profile derives its allow /
ask lists and its ``write_tools`` from the three tuples below.

Safety classes: **R** read (auto-allow), **W** write (the tier decides:
on allow at the admin tier only, per the deployment's DEVIATION marker),
**S** the halt family (allow wherever the server renders, never behind
the headless gate).
"""

from __future__ import annotations

# reads
SCAN_STATUS = "scan_status"
SCAN_PROGRESS = "scan_progress"
SCAN_HISTORY = "scan_history"
LIST_SCAN_CONFIGS = "list_scan_configs"
PREFLIGHT_PRESET = "preflight_preset"
WAIT_FOR_SCAN = "wait_for_scan"
GET_SCAN_ANALYSIS = "get_scan_analysis"
GET_SCAN_FIGURE = "get_scan_figure"
GET_ANALYSIS_ARTIFACT = "get_analysis_artifact"
GET_SCAN_SCALARS = "get_scan_scalars"
LIST_ANALYZERS = "list_analyzers"
LIST_ANALYSIS_GROUPS = "list_analysis_groups"
WAIT_FOR_ANALYSIS = "wait_for_analysis"

# writes
SUBMIT_PRESET = "submit_preset"
MOVE = "move"
RESUME_SCAN = "resume_scan"
CLEAR_QUEUE = "clear_queue"
RUN_ANALYSIS = "run_analysis"
POST_LOGBOOK_ENTRY = "post_logbook_entry"

# the halt family
STOP_SCAN = "stop_scan"
PAUSE_SCAN = "pause_scan"

#: Read-only tools (R) — safe to auto-allow in profile.yml.  ``wait_for_scan``
#: is a read: it polls the queue and changes nothing; ``preflight_preset``
#: runs the scanner's preflight (manager and gateway reads) and submits
#: nothing.
READ_TOOLS = (
    SCAN_STATUS,
    SCAN_PROGRESS,
    SCAN_HISTORY,
    LIST_SCAN_CONFIGS,
    PREFLIGHT_PRESET,
    WAIT_FOR_SCAN,
    GET_SCAN_ANALYSIS,
    GET_SCAN_FIGURE,
    GET_ANALYSIS_ARTIFACT,
    GET_SCAN_SCALARS,
    LIST_ANALYZERS,
    LIST_ANALYSIS_GROUPS,
    WAIT_FOR_ANALYSIS,
)

#: Write tools (W) — every verb that queues work, moves hardware or
#: writes a record: the deployment lists these in ``write_tools`` (the
#: headless gate) and decides their interactive posture per tier.
#: ``resume_scan`` is a write, not a halt: it restarts motion;
#: ``run_analysis`` writes the analysis tree and burns the portal's CPU;
#: ``post_logbook_entry`` writes a (draft) record in the scans book.
WRITE_TOOLS = (
    SUBMIT_PRESET,
    MOVE,
    RESUME_SCAN,
    CLEAR_QUEUE,
    RUN_ANALYSIS,
    POST_LOGBOOK_ENTRY,
)

#: The halt family (S) — NEVER listed in ``write_tools``: exempt by
#: omission from the headless gate BY DESIGN, so a halt is never blocked
#: on any path (verified osprey semantics 2026-08-22; deploy/DEPLOYMENT.md).
#: ``pause_scan`` sits here with ``stop_scan``: a pause makes the machine
#: strictly quieter.
STOP_TOOLS = (STOP_SCAN, PAUSE_SCAN)
