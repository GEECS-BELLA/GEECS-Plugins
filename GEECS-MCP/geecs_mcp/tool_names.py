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

#: Queueing tools (Q) — `ask` interactively; listed in the profile's
#: `config:` `write_tools` for the headless gate (hook presets do NOT
#: attach to custom servers — see STOP_TOOLS below / DEPLOYMENT.md).
QUEUE_TOOLS = (
    SUBMIT_SCAN,
    CLEAR_QUEUE,
)

#: Stop direction (S) — `ask` only.  VERIFIED osprey semantics
#: (2026-08-22): hook presets and the interactive kill switch do NOT
#: apply to custom-server tools at all, so stop is not kill-switch-
#: blocked (the halt doctrine holds, by upstream gap rather than
#: design) and the native ask prompt is the interactive gate on every
#: control verb.  See deploy/DEPLOYMENT.md for the full semantics and
#: the two upstream gaps to file.
STOP_TOOLS = (STOP_SCAN,)
