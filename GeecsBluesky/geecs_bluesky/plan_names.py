"""The queueserver plan names the GEECS worker registers — import-light on purpose.

One spelling of every name the worker's startup profile exports, shared by
the two sides that must agree with it and never with each other's heavy
imports: the client (``qs_client``) and the service-start readiness check
(``qs_client.ready``, which asserts the manager lists :data:`GEECS_PLAN_NAMES`
after the environment opens).

The scan choices are count, sweep and optimize. Utilities do not open runs.
Moving stock verbs are internal implementation details, never registrations.

This module may depend on nothing heavier than the standard library (the
same rule as :mod:`geecs_bluesky.log_markers`).
"""

from __future__ import annotations

#: The registered plans that are not scans: a queue item naming one opens
#: no run and claims no scan number, so a preset cannot name them
#: (``qs_client.presets.PRESET_PLAN_NAMES``).  ``mv`` and ``apply_setup``
#: take no detector list; the calibration plans take one but no positions
#: and write no data; ``release_scan_mode`` ends GEECS scan mode on the
#: devices of a run that never ended it (a halt skips the finalizers).
NON_SCAN_PLAN_NAMES: tuple[str, ...] = (
    "mv",
    "apply_setup",
    "measure_shot_offsets",
    "check_shot_sync",
    "release_scan_mode",
)

#: The once-run shot-offset plans: the
#: calibration that measures each device's edge-to-stamp latency, and the
#: preflight that says whether the stored measurement still holds.  Named
#: apart because both drive the trigger box OFF and cost at least the
#: longest device timeout — never a step inside a scan.
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

#: The acquisition modes every bound scan verb accepts (``acquisition=``):
#: strict single-shot, or fly (a software tick, no box).  Shared by the
#: registry (the plan) and the client seam (the preset).
ACQUISITION_MODES: tuple[str, ...] = ("strict", "fly")

#: Modes that existed and were deleted, with why — refused by name rather
#: than reported as unknown, so a preset or request still asking for one
#: reads the reason and never runs as something else.
RETIRED_ACQUISITION_MODES: dict[str, str] = {
    "gated": (
        "strict is the one box-driven mode (a sweep may run under 1 Hz) "
        "— use acquisition='strict'"
    ),
}


def acquisition_refusal(acquisition: object) -> str | None:
    """Why *acquisition* is refused, or ``None`` when it is a mode this worker runs.

    The one wording for the worker's bind and the client's expansion.
    """
    if acquisition in ACQUISITION_MODES:
        return None
    retired = RETIRED_ACQUISITION_MODES.get(str(acquisition))
    if retired is not None:
        return f"acquisition={acquisition!r} is retired: {retired}"
    return f"acquisition={acquisition!r} is not one of {ACQUISITION_MODES}"


__all__ = [
    "ACQUISITION_MODES",
    "RETIRED_ACQUISITION_MODES",
    "acquisition_refusal",
    "CALIBRATION_PLAN_NAMES",
    "GEECS_PLAN_NAMES",
    "NON_SCAN_PLAN_NAMES",
    "NATIVE_SCAN_PLAN_NAMES",
]
