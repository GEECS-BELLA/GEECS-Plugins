"""Tool-name constants — the one place tool names are spelled.

The osprey ``bluesky_tool_names.py`` pattern: profile permission lists and
hook matchers import these symbols instead of retyping strings, so a
rename cannot silently strand a permission entry.

Safety classes (the planning doc's vocabulary): every v0 tool is **R**
(read-only, auto-allow).  The v1 queueing/stop verbs will be added here
with their classes when they land.
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

#: Stop direction (S) — `ask` ONLY, never behind the kill switch: a halt
#: must always be possible (the osprey bluesky-server doctrine; the
#: exemption is enforced in-tool since per-tool hook matchers don't
#: exist for custom servers).
STOP_TOOLS = (STOP_SCAN,)
