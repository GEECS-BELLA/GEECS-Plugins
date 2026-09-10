"""The queueserver plan names the GEECS worker registers — import-light on purpose.

One spelling of every name the worker's startup profile exports, shared by
the two sides that must agree with it and never with each other's heavy
imports: the client (``qs_client``) and the service-start readiness check
(``qserver_ready``, which asserts the manager lists :data:`GEECS_PLAN_NAMES`
after the environment opens — the invariant #793 found violated).

**Phase 1 of the native-Bluesky rebuild (#807):** the worker registers the
**stock** ``bluesky.plans`` verbs over the device namespace and nothing
GEECS-named.  The funnel (``geecs_scan_request_plan``) and the named plans
are deleted; the plan layer (phase 1, PR 2 —
``Planning/native_bluesky/03_clean_room_rebuild.md`` §4.D, §10.5)
re-registers these same names with the strict ``take_reading`` pre-bound,
so the names here are already the final ones.  Until then the client's
submit verbs still name the retired funnel, so the pre-submit
``worker_ready`` check refuses against this worker — correctly: it cannot
run a ``ScanRequest``.  The client seam is rewired with the table.

This module may depend on nothing heavier than the standard library (the
same rule as :mod:`geecs_bluesky.log_markers`).
"""

from __future__ import annotations

#: The plan ``QueueClient.submit_scan`` queues — the retired funnel's name,
#: kept only until the registration table (PR 2) rewires the client seam.
SCAN_REQUEST_PLAN = "geecs_scan_request_plan"
#: The plan ``QueueClient.submit_action`` queues — same status.
RUN_ACTION_PLAN = "geecs_run_action_plan"

#: Every plan the worker registers: the stock ``bluesky.plans`` scan verbs
#: (absolute and relative) plus ``mv``, the manual move as a queue item.
GEECS_PLAN_NAMES: tuple[str, ...] = (
    "count",
    "scan",
    "rel_scan",
    "list_scan",
    "rel_list_scan",
    "grid_scan",
    "rel_grid_scan",
    "list_grid_scan",
    "rel_list_grid_scan",
    "mv",
)

__all__ = [
    "SCAN_REQUEST_PLAN",
    "RUN_ACTION_PLAN",
    "GEECS_PLAN_NAMES",
]
