"""The queueserver plan names the GEECS worker registers — import-light on purpose.

One spelling of every name the worker's startup profile exports, shared by
the two sides that must agree with it and never with each other's heavy
imports: the client (``qs_client``) and the service-start readiness check
(``qserver_ready``, which asserts the manager lists :data:`GEECS_PLAN_NAMES`
after the environment opens — the invariant #793 found violated).

The names are the stock ``bluesky.plans`` verbs — registered by the plan
layer with the strict ``take_reading`` pre-bound
(:mod:`geecs_bluesky.plans.registry`; plan of record §4.D, §10.5) — plus
``mv``, the manual move as a queue item.  The table is every stock verb
exposing ``per_step`` / ``per_shot`` that a queue item can express;
``tests/test_plan_registry.py`` pins this tuple to that derivation.

This module may depend on nothing heavier than the standard library (the
same rule as :mod:`geecs_bluesky.log_markers`).
"""

from __future__ import annotations

#: Every plan the worker registers: the stock ``bluesky.plans`` scan verbs
#: (absolute and relative) bound strict, plus ``mv``.
GEECS_PLAN_NAMES: tuple[str, ...] = (
    "count",
    "scan",
    "rel_scan",
    "list_scan",
    "rel_list_scan",
    "log_scan",
    "rel_log_scan",
    "grid_scan",
    "rel_grid_scan",
    "list_grid_scan",
    "rel_list_grid_scan",
    "spiral",
    "rel_spiral",
    "spiral_fermat",
    "rel_spiral_fermat",
    "spiral_square",
    "rel_spiral_square",
    "x2x_scan",
    "mv",
)

__all__ = ["GEECS_PLAN_NAMES"]
