"""Tool-name constants — the one place tool names are spelled.

The osprey ``bluesky_tool_names.py`` pattern: profile permission lists and
hook matchers import these symbols instead of retyping strings, so a
rename cannot silently strand a permission entry.

Safety classes (the planning doc's vocabulary): **R** read-only
(auto-allow), **Q** queueing (`ask` + `writes_check`), **S** stop
direction (`ask`; see the kill-switch caveat on ``STOP_TOOLS``).
"""

from __future__ import annotations

SCAN_STATUS = "scan_status"
SCAN_HISTORY = "scan_history"
GET_SCAN_RESULT = "get_scan_result"
LIST_SCAN_CONFIGS = "list_scan_configs"
VALIDATE_SCAN_REQUEST = "validate_scan_request"
SCAN_PROGRESS = "scan_progress"
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
)

#: Queueing tools (Q) — `ask` + the writes_check (kill switch) preset.
QUEUE_TOOLS = (
    SUBMIT_SCAN,
    CLEAR_QUEUE,
)

#: Stop direction (S) — `ask` only.  Doctrine: a halt should never sit
#: behind the kill switch — but osprey hook presets attach SERVER-WIDE
#: for custom servers, so a `writes_check` on this server gates stop too
#: (this server cannot see the kill switch to exempt itself in-tool the
#: way osprey's native bluesky server does).  Honest posture: the
#: console/manager remain the kill-switch-proof stop paths; per-tool
#: matchers are the osprey-side fix.  See deploy/DEPLOYMENT.md.
STOP_TOOLS = (STOP_SCAN,)
