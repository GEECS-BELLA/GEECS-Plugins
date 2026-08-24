"""Tool-name constants — the one place tool names are spelled.

The osprey ``bluesky_tool_names.py`` pattern: profile permission lists and
hook matchers import these symbols instead of retyping strings, so a
rename cannot silently strand a permission entry.

Safety classes (the planning doc's vocabulary): **R** read-only
(auto-allow), **Q** queueing (`ask`), **S** stop direction (`ask`; see
the verified gating semantics on ``STOP_TOOLS``).
"""

from __future__ import annotations

SCAN_STATUS = "scan_status"
SCAN_HISTORY = "scan_history"
GET_SCAN_RESULT = "get_scan_result"
LIST_SCAN_CONFIGS = "list_scan_configs"
VALIDATE_SCAN_REQUEST = "validate_scan_request"
SCAN_PROGRESS = "scan_progress"
GET_SCAN_ANALYSIS = "get_scan_analysis"
GET_SCAN_FIGURE = "get_scan_figure"
SUBMIT_SCAN = "submit_scan"
STOP_SCAN = "stop_scan"
CLEAR_QUEUE = "clear_queue"

#: Read-only tools (R) — safe to auto-allow in profile.yml.
READ_TOOLS = (
    SCAN_STATUS,
    SCAN_HISTORY,
    GET_SCAN_RESULT,
    LIST_SCAN_CONFIGS,
    VALIDATE_SCAN_REQUEST,
    SCAN_PROGRESS,
    GET_SCAN_ANALYSIS,
    GET_SCAN_FIGURE,
)

#: Queueing tools (Q) — `ask` interactively; listed in `write_tools`
#: (hook_config.json, from the profile's `config:`) for the headless
#: gate (hook presets do NOT attach to custom servers — see STOP_TOOLS
#: below / DEPLOYMENT.md).
QUEUE_TOOLS = (
    SUBMIT_SCAN,
    CLEAR_QUEUE,
)

#: Stop direction (S) — `ask` only, and NEVER listed in `write_tools`:
#: exempt by omission from the headless gate BY DESIGN, and outside the
#: interactive kill switch because it does not cover custom servers
#: (upstream gap) — so a halt is never blocked on any path.  VERIFIED
#: osprey semantics 2026-08-22; see deploy/DEPLOYMENT.md.
STOP_TOOLS = (STOP_SCAN,)
