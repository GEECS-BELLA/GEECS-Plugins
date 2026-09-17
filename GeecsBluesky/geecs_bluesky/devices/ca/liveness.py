"""The CONNECTED liveness reads — one verdict rule, out of plan and in plan.

``CONNECTED`` is the authoritative liveness signal (the gateway serves
every DB device's data PVs whether or not the device is up, so CA-connect
success never implies liveness).  Two readers, one rule — **fail-open**:
only the exact ``"Disconnected"`` choice string is a verdict; an
unreadable PV, a missing signal or a mock backend's ``""`` default all
read live.

- :func:`probe_disconnected` — the out-of-plan probe over device *names*
  (the client-side pre-submit preflight,
  :mod:`geecs_bluesky.qs_client.submit_preflight`).  Its sharp edge is
  kept here in exactly one place: the PV is a **DBR_ENUM**, so the read
  must pass ``datatype=str`` — a native read returns the integer index,
  which can never match the choice string.
- :func:`read_disconnected` — the in-plan read over *built* devices'
  ``connected_status`` signals (the run's liveness gate before its first
  move, :mod:`geecs_bluesky.plans.registry`; the strict refire gate,
  :mod:`geecs_bluesky.plans.strict`).  A typed ``str`` signal has no
  enum edge to trip over.

``aioca`` is imported lazily on first use (the ``ca`` extra).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Iterable

import bluesky.plan_stubs as bps

from geecs_bluesky.devices.ca._pv import GATEWAY_DISCONNECTED

logger = logging.getLogger(__name__)

#: CA read budget for one probe batch (seconds) — concurrent, so N dead
#: PVs cost one budget, not N.
DEFAULT_PROBE_TIMEOUT_S = 2.0


def read_disconnected(
    signals: Mapping[str, Any], *, unreadable_level: int = logging.DEBUG
):
    """Plan: the names whose ``CONNECTED`` signal reads ``Disconnected``, in input order.

    One ``bps.rd`` per signal.  Fail-open per signal: a read that raises is
    logged at *unreadable_level* and counts as live — the gateway serves
    ``CONNECTED`` for every DB device, so an unreadable one is a transport
    question (a device added to the DB after the gateway started, a
    gateway going away), not a liveness verdict.

    Parameters
    ----------
    signals :
        GEECS device name → its ``connected_status`` signal (a ``str``
        ``SignalR`` on the gateway's ``CONNECTED`` PV).
    unreadable_level :
        Log level for a read that raises: the run's gate passes WARNING
        (abnormal before a run, and it cost the connect timeout); the
        mid-scan refire gate keeps DEBUG.

    Yields
    ------
    Bluesky messages.

    Returns
    -------
    list of str
        The devices confirmed down.
    """
    down: list[str] = []
    for device, signal in signals.items():
        try:
            value = yield from bps.rd(signal)
        except Exception as exc:
            logger.log(
                unreadable_level,
                "CONNECTED read failed for %s (%s: %s); assuming live (fail-open)",
                device,
                type(exc).__name__,
                exc,
                exc_info=unreadable_level < logging.WARNING,
            )
            continue
        if value == GATEWAY_DISCONNECTED:
            down.append(device)
    return down


def probe_disconnected(
    experiment: str,
    device_names: Iterable[str],
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_S,
) -> list[str]:
    """Return the devices whose gateway ``CONNECTED`` PV reads ``Disconnected``.

    Fail-open per the liveness doctrine: an unreadable PV is not a
    verdict — only the exact ``Disconnected`` choice string counts.  All
    reads run concurrently on the shared one-shot loop, so the worst case
    costs one *timeout* budget regardless of device count.

    Parameters
    ----------
    experiment : str
        The experiment PV prefix.
    device_names :
        GEECS device names to probe.
    timeout : float
        CA read budget for the whole batch, in seconds.

    Returns
    -------
    list of str
        The subset of *device_names* confirmed down, in input order.
    """
    from geecs_bluesky.devices.ca._pv import GATEWAY_DISCONNECTED, ca_pv
    from geecs_bluesky.devices.ca.gateway_put import bare_pv
    from geecs_bluesky.devices.ca.oneshot import try_caget_many

    names = list(device_names)
    if not names:
        return []
    pvs = [bare_pv(ca_pv(experiment, device, "CONNECTED")) for device in names]
    # datatype=str is load-bearing — see the module docstring.
    readings = try_caget_many(pvs, timeout=timeout, datatype=str)
    return [
        device
        for device, reading in zip(names, readings)
        if reading is not None and str(reading) == GATEWAY_DISCONNECTED
    ]
