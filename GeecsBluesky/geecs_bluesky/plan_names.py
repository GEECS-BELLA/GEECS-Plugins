"""The queueserver plan names the GEECS worker registers — import-light on purpose.

One spelling of every name the worker's startup profile exports, shared by
the two sides that must agree with it and never with each other's heavy
imports: the client (``qs_client``) and the service-start readiness check
(``qserver_ready``, which asserts the manager lists :data:`GEECS_PLAN_NAMES`
after the environment opens — the invariant #793 found violated).

The names are the stock ``bluesky.plans`` verbs — registered by the plan
layer with the strict ``take_reading`` pre-bound
(:mod:`geecs_bluesky.plans.registry`; plan of record §4.D, §10.5) — plus
the two GEECS queue items that are not scans: ``mv``, the manual move,
and ``run_action``, a named action plan from the experiment's action
library (:data:`NON_SCAN_PLAN_NAMES`).  The scan verbs are every stock
verb exposing ``per_step`` / ``per_shot`` that a queue item can express;
``tests/test_plan_registry.py`` pins this tuple to that derivation.

This module may depend on nothing heavier than the standard library (the
same rule as :mod:`geecs_bluesky.log_markers`).
"""

from __future__ import annotations

#: The registered plans that are not scans: a queue item naming one runs no
#: run, claims no scan number and takes no detector list — so a preset
#: cannot name them (``qs_client.presets.PRESET_PLAN_NAMES``).
NON_SCAN_PLAN_NAMES: tuple[str, ...] = ("mv", "run_action")

#: Every plan the worker registers: the stock ``bluesky.plans`` scan verbs
#: (absolute and relative) bound strict, plus :data:`NON_SCAN_PLAN_NAMES`.
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
    *NON_SCAN_PLAN_NAMES,
)

#: The acquisition modes every bound scan verb accepts (``acquisition=``,
#: ``08_gated_batch.md`` §4.1): strict single-shot, or the gated batch.
#: Shared by the registry (the plan) and the client seam (the preset).
ACQUISITION_MODES: tuple[str, ...] = ("strict", "gated")

__all__ = ["ACQUISITION_MODES", "GEECS_PLAN_NAMES", "NON_SCAN_PLAN_NAMES"]
