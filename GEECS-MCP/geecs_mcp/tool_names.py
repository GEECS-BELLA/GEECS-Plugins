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
LIST_ANALYZERS = "list_analyzers"
LIST_ANALYSIS_GROUPS = "list_analysis_groups"
RUN_SCAN_ANALYSIS = "run_scan_analysis"
SUBMIT_SCAN = "submit_scan"
STOP_SCAN = "stop_scan"
CLEAR_QUEUE = "clear_queue"
RUN_ACTION = "run_action"
DESCRIBE_ACTION = "describe_action"
MOVE_SCAN_VARIABLE = "move_scan_variable"
PAUSE_SCAN = "pause_scan"
RESUME_SCAN = "resume_scan"

#: Read-only tools (R) — safe to auto-allow in profile.yml.
#: describe_action is R by effect (a worker-side dry-run that changes
#: nothing), though it does require an idle manager to answer.
READ_TOOLS = (
    SCAN_STATUS,
    SCAN_HISTORY,
    GET_SCAN_RESULT,
    LIST_SCAN_CONFIGS,
    VALIDATE_SCAN_REQUEST,
    SCAN_PROGRESS,
    GET_SCAN_ANALYSIS,
    GET_SCAN_FIGURE,
    LIST_ANALYZERS,
    LIST_ANALYSIS_GROUPS,
    DESCRIBE_ACTION,
)

#: Queueing tools (Q) — `ask` interactively; listed in `write_tools`
#: (hook_config.json, from the profile's `config:`) for the headless
#: gate (hook presets do NOT attach to custom servers — see STOP_TOOLS
#: below / DEPLOYMENT.md).  resume_scan is Q, not S: it *restarts*
#: motion/acquisition (and retries the failed move), so it belongs
#: behind the headless gate like any other go verb.
#: run_scan_analysis is Q by effect, not by queueing semantics: it writes
#: to the data share (analysis outputs + analysis_status/ inside an
#: EXISTING scan folder) and burns CPU on the box shared with the
#: production manager (#686).
QUEUE_TOOLS = (
    SUBMIT_SCAN,
    CLEAR_QUEUE,
    RUN_ACTION,
    MOVE_SCAN_VARIABLE,
    RESUME_SCAN,
    RUN_SCAN_ANALYSIS,
)

#: Stop direction (S) — `ask` only, and NEVER listed in `write_tools`:
#: exempt by omission from the headless gate BY DESIGN, and outside the
#: interactive kill switch because it does not cover custom servers
#: (upstream gap) — so a halt is never blocked on any path.  VERIFIED
#: osprey semantics 2026-08-22; see deploy/DEPLOYMENT.md.  pause_scan
#: joins stop_scan here: a pause makes the machine strictly quieter.
STOP_TOOLS = (STOP_SCAN, PAUSE_SCAN)
