"""A native EPICS camera's run-shape scope, worded once — import-light on purpose.

A native camera (the configs repo's ``native_devices.yaml``) runs as an
essential detector of a strict scan only for now (``NATIVE_DEVICES.md``,
"Scope of the first build").  Two sides refuse the other shapes and must
say the same thing: the worker at bind
(:func:`~geecs_bluesky.devices.native.refuse_native_cameras`, over device
objects) and the client's preset expansion
(:func:`~geecs_bluesky.qs_client.presets.expand_preset`, over registry
names) — which must name the true reason before the client's generic
device rules read a native camera as an unstamped device.

This module may depend on nothing heavier than the standard library (the
same rule as :mod:`geecs_bluesky.plan_names`).
"""

from __future__ import annotations

from collections.abc import Iterable

#: Why each refused shape is refused — the tail of :func:`native_refusal`.
NOT_NON_ESSENTIAL = "not as a non-essential stream"
NOT_GATED = "not in a gated batch"
NOT_SCAN_MODE = (
    "not under a scan-mode profile (the IOC never acknowledges "
    "ScanStart, and a box-less profile fires no edge)"
)
NOT_OPTIMIZE = "not in optimize (its live frames are matched on gateway PVs)"
NOT_CALIBRATION = "not in a shot-offset calibration (it reads GEECS stamps)"


def native_refusal(names: Iterable[str], why: str) -> str:
    """The refusal sentence naming the native camera(s) *names* and *why*."""
    return (
        f"{', '.join(names)}: native EPICS cameras run in strict scans only "
        f"for now — {why}"
    )


__all__ = [
    "NOT_CALIBRATION",
    "NOT_GATED",
    "NOT_NON_ESSENTIAL",
    "NOT_OPTIMIZE",
    "NOT_SCAN_MODE",
    "native_refusal",
]
