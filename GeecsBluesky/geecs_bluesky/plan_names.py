"""The queueserver plan names the GEECS worker registers — import-light on purpose.

One spelling of every name the worker's startup profile exports, shared by
the two sides that must agree with it and never with each other's heavy
imports: the client (``qs_client``) and the service-start readiness check
(``qserver_ready``, which asserts the manager lists :data:`GEECS_PLAN_NAMES`
after the environment opens — the invariant #793 found violated).

The scan choices are count, sweep and optimize. Utilities do not open runs.
Moving stock verbs are internal implementation details, never registrations.

This module may depend on nothing heavier than the standard library (the
same rule as :mod:`geecs_bluesky.log_markers`).
"""

from __future__ import annotations

#: The registered plans that are not scans: a queue item naming one runs no
#: run and claims no scan number — so a preset cannot name them
#: (``qs_client.presets.PRESET_PLAN_NAMES``).  ``mv`` and ``run_action``
#: take no detector list at all; the two calibration plans
#: (:data:`CALIBRATION_PLAN_NAMES`) do take one, but they take no positions
#: and write no data, so a preset — which describes a *scan* — still cannot
#: express them.
NON_SCAN_PLAN_NAMES: tuple[str, ...] = (
    "mv",
    "run_action",
    "measure_shot_offsets",
    "check_shot_sync",
)

#: The once-run shot-offset plans (``03_clean_room_rebuild.md`` §4.F): the
#: calibration that measures each device's edge-to-stamp latency, and the
#: preflight that says whether the stored measurement still holds.  Named
#: apart because both drive the trigger box OFF and cost at least the
#: longest device timeout (§11.2) — never a step inside a scan.
CALIBRATION_PLAN_NAMES: tuple[str, ...] = (
    "measure_shot_offsets",
    "check_shot_sync",
)

#: Native scan plans that open runs but have no stock Bluesky counterpart.
NATIVE_SCAN_PLAN_NAMES: tuple[str, ...] = (
    "sweep",
    "optimize",
)

#: Every public plan the worker registers.
GEECS_PLAN_NAMES: tuple[str, ...] = (
    "count",
    *NATIVE_SCAN_PLAN_NAMES,
    *NON_SCAN_PLAN_NAMES,
)

#: The acquisition modes every bound scan verb accepts (``acquisition=``,
#: ``08_gated_batch.md`` §4.1): strict single-shot, or the gated batch.
#: Shared by the registry (the plan) and the client seam (the preset).
ACQUISITION_MODES: tuple[str, ...] = ("strict", "gated")

__all__ = [
    "ACQUISITION_MODES",
    "CALIBRATION_PLAN_NAMES",
    "GEECS_PLAN_NAMES",
    "NON_SCAN_PLAN_NAMES",
    "NATIVE_SCAN_PLAN_NAMES",
]
