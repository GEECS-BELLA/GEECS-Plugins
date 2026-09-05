"""The queueserver plan and function-verb names — import-light on purpose.

One spelling of every name the GEECS worker registers with the RE Manager
(``qserver/startup/startup.py`` builds its ``__all__`` from these), shared
by the two sides that must agree with it and never with each other's heavy
imports: the client (``qs_client`` submits :data:`SCAN_REQUEST_PLAN` and
asks the manager whether it is *allowed*) and the service-start readiness
check (``qserver_ready`` asserts the manager lists :data:`GEECS_PLAN_NAMES`
after the environment opens — the invariant #793 found violated).  The plan
*functions* live in ``geecs_bluesky.plans``; the pin that each name here
is a real plan there is ``tests/test_plan_names.py``.

This module may depend on nothing heavier than the standard library (the
same rule as :mod:`geecs_bluesky.log_markers`).
"""

from __future__ import annotations

#: The funnel: every ``ScanRequest`` (step, noscan, optimize) runs through
#: it; the one plan the clients submit (``QueueClient.submit_scan``).
SCAN_REQUEST_PLAN = "geecs_scan_request_plan"
#: On-demand ActionPlan execution as a queue item (decision 2).
RUN_ACTION_PLAN = "geecs_run_action_plan"
#: The named per-mode plans (Phase 2b-ii) — same execution underneath.
NOSCAN_PLAN = "geecs_noscan_plan"
SCAN_PLAN = "geecs_scan_plan"
OPTIMIZE_PLAN = "geecs_optimize_plan"

#: Every plan the worker registers, in the startup profile's export order.
GEECS_PLAN_NAMES: tuple[str, ...] = (
    SCAN_REQUEST_PLAN,
    RUN_ACTION_PLAN,
    NOSCAN_PLAN,
    SCAN_PLAN,
    OPTIMIZE_PLAN,
)

#: The ``function_execute`` manual verbs (not plans; idle-manager only).
MOVE_VARIABLE_FUNCTION = "geecs_move_variable"
DESCRIBE_ACTION_FUNCTION = "geecs_describe_action"
GEECS_WORKER_FUNCTIONS: tuple[str, ...] = (
    MOVE_VARIABLE_FUNCTION,
    DESCRIBE_ACTION_FUNCTION,
)

__all__ = [
    "SCAN_REQUEST_PLAN",
    "RUN_ACTION_PLAN",
    "NOSCAN_PLAN",
    "SCAN_PLAN",
    "OPTIMIZE_PLAN",
    "GEECS_PLAN_NAMES",
    "MOVE_VARIABLE_FUNCTION",
    "DESCRIBE_ACTION_FUNCTION",
    "GEECS_WORKER_FUNCTIONS",
]
