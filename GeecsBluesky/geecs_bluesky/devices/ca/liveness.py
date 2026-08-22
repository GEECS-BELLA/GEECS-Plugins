"""The one out-of-plan CONNECTED liveness probe.

``CONNECTED`` is the authoritative liveness signal (the gateway serves
every DB device's data PVs whether or not the device is up, so CA-connect
success never implies liveness), and reading it correctly has one sharp
edge worth keeping in exactly one place: the PV is a **DBR_ENUM**, so the
read must pass ``datatype=str`` — a native read returns the integer index,
which can never match the ``"Disconnected"`` choice string.

Consumers: the client-side pre-submit preflight
(:mod:`geecs_bluesky.qs_client.submit_preflight`) and the worker's
pre-claim re-check (:func:`geecs_bluesky.scan_request_runner._preflight_connected`).
In-plan liveness reads on *built* devices go through their
``connected_status`` signal instead (:mod:`geecs_bluesky.plans.liveness`).

``aioca`` is imported lazily on first use (the ``ca`` extra).
"""

from __future__ import annotations

from typing import Iterable

#: CA read budget for one probe batch (seconds) — concurrent, so N dead
#: PVs cost one budget, not N.
DEFAULT_PROBE_TIMEOUT_S = 2.0


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
