"""The one in-plan liveness read for built devices.

The device family carries a non-readable ``connected_status`` child on
the gateway's per-device ``CONNECTED`` PV — the authoritative liveness
signal.  Plan-context consumers (the strict refire gate in
:mod:`~geecs_bluesky.plans.single_shot`, the t0-sync seed gate in
:mod:`~geecs_bluesky.plans.t0_sync`) read it through this one stub so the
fail-open convention cannot drift between hand-rolled copies: only the
exact ``Disconnected`` choice string is ever a verdict.  Out-of-plan
probes (no device built yet) use
:func:`geecs_bluesky.devices.ca.liveness.probe_disconnected` instead.
"""

from __future__ import annotations

import logging
from typing import Any

import bluesky.plan_stubs as bps

from geecs_bluesky.devices.ca._pv import GATEWAY_DISCONNECTED

logger = logging.getLogger(__name__)


def rd_confirmed_down(device: Any):
    """Plan stub: ``True`` iff *device*'s ``connected_status`` reads down.

    **Fail-open**: no ``connected_status`` attribute (older device
    shapes, bare fakes) or a failed read is not a verdict and returns
    ``False``; only the exact ``Disconnected`` choice string confirms —
    a mock backend's ``""`` default reads live.
    """
    signal = getattr(device, "connected_status", None)
    if signal is None:
        return False
    try:
        value = yield from bps.rd(signal)
    except Exception:
        logger.debug(
            "CONNECTED read failed for %s; assuming live (fail-open)",
            getattr(device, "name", device),
            exc_info=True,
        )
        return False
    return value == GATEWAY_DISCONNECTED
