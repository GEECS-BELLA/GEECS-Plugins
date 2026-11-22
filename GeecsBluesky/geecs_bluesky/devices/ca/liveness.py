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
- :func:`down_natives` — native devices (no ``CONNECTED``): a judge that raises is down.
- :func:`probe_stale` — a frozen loop (``systimestamp`` stale): a warning only.
- :func:`probe_gateway_monitors` — does the gateway deliver monitors at all.

``aioca`` is imported lazily on first use (the ``ca`` extra).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Iterable, Literal, Sequence

import asyncio

import bluesky.plan_stubs as bps

from geecs_bluesky.devices.ca._pv import GATEWAY_DISCONNECTED

logger = logging.getLogger(__name__)

#: CA read budget for one probe batch (seconds) — concurrent, so N dead
#: PVs cost one budget, not N.
DEFAULT_PROBE_TIMEOUT_S = 2.0

#: A device whose ``systimestamp`` is older than this (seconds) has a
#: frozen loop: connected, but publishing nothing.
STALE_AFTER_S = 30.0


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
        except Exception as exc:  # noqa: BLE001 — liveness read fails open
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


#: :func:`probe_gateway_monitors` verdicts.
MonitorVerdict = Literal["delivered", "withheld", "unreachable"]


def probe_gateway_monitors(
    experiment: str, *, timeout: float = DEFAULT_PROBE_TIMEOUT_S, grace: float = 1.0
) -> MonitorVerdict:
    """Whether the CA gateway delivers monitors, judged on its heartbeat over one held subscription.

    A get first: one that fails is ``"unreachable"`` at once (not a verdict
    on monitors; fail-open, as the ``CONNECTED`` probes), and one that
    answers proves the channel connected.  Then one subscription, held to
    the end: its value on subscribe within *timeout* is ``"delivered"``.
    If nothing came, a second get on the same open subscription (failing:
    ``"unreachable"``) and a further *grace* for the late value
    (``"delivered"``).  ``"withheld"`` therefore means a get answered
    before the subscribe and another after *timeout* of silence, and the
    subscription stayed silent through *grace* too: the gateway serves
    gets but delivers no monitor events.

    Costs two round trips when healthy; worst cases: unreachable one
    *timeout*, or three plus a *grace* when the second get fails;
    withheld two *timeout* budgets (at most) plus *grace*.

    Raises
    ------
    ImportError
        ``aioca`` is missing (the ``ca`` extra): the probe cannot run.
    """
    from geecs_core.pv_naming import GATEWAY_HEARTBEAT, gateway_status_pv

    from geecs_bluesky.devices.ca.oneshot import FirstValueWatch, try_caget_once

    pv = gateway_status_pv(experiment, GATEWAY_HEARTBEAT)
    if try_caget_once(pv, timeout=timeout) is None:
        return "unreachable"
    with FirstValueWatch(pv) as watch:
        if watch.wait(timeout):
            return "delivered"
        if try_caget_once(pv, timeout=timeout) is None:
            return "unreachable"
        return "delivered" if watch.wait(grace) else "withheld"


def probe_stale_pvs(
    stamp_pvs: Mapping[str, str],
    *,
    window: float = STALE_AFTER_S,
    timeout: float = DEFAULT_PROBE_TIMEOUT_S,
    now: float | None = None,
) -> list[tuple[str, float | None]]:
    """The devices whose loop stamp is older than *window*: ``(name, age s)``, ``None`` when never set.

    *stamp_pvs* is GEECS device name → its ``systimestamp`` PV (bare).
    One concurrent read; a read that fails is not judged (fail-open, as
    :func:`probe_disconnected`).  The age is against this host's clock, so
    a skew between hosts larger than *window* would read as stale.
    """
    import time

    from geecs_core.db.variable_types import LABVIEW_EPOCH_OFFSET

    from geecs_bluesky.devices.ca.oneshot import try_caget_many

    names = list(stamp_pvs)
    if not names:
        return []
    readings = try_caget_many([stamp_pvs[n] for n in names], timeout=timeout)
    wall = (time.time() if now is None else now) + LABVIEW_EPOCH_OFFSET
    stale: list[tuple[str, float | None]] = []
    for name, reading in zip(names, readings):
        try:
            stamp = float(reading)
        except (TypeError, ValueError):
            continue  # unreadable: not judged
        if stamp <= 0:
            stale.append((name, None))  # the gateway's placeholder: never published
        elif wall - stamp > window:
            stale.append((name, wall - stamp))
    return stale


def probe_stale(
    experiment: str,
    device_names: Iterable[str],
    *,
    window: float = STALE_AFTER_S,
    timeout: float = DEFAULT_PROBE_TIMEOUT_S,
) -> list[tuple[str, float | None]]:
    """:func:`probe_stale_pvs` over device *names* (the client-side preflight)."""
    from geecs_bluesky.devices.ca._pv import ca_pv
    from geecs_bluesky.devices.ca.gateway_put import bare_pv

    return probe_stale_pvs(
        {d: bare_pv(ca_pv(experiment, d, "systimestamp")) for d in device_names},
        window=window,
        timeout=timeout,
    )


def stale_text(stale: Sequence[tuple[str, float | None]]) -> str:
    """``"A (not updated in 95 s), B (never updated)"`` — the one wording."""
    return ", ".join(
        f"{name} (never updated)"
        if age is None
        else f"{name} (not updated in {age:.0f} s)"
        for name, age in stale
    )


def down_natives(found: Sequence[Any], judge: Callable[[Any], Awaitable[bool]]):
    """Plan: the GEECS names of the native devices in *found* that *judge* says are down.

    Not the gateway's fail-open rule: a native device's word comes from its
    own IOC, so a judge that raises (an IOC that never answered, or died
    after an earlier connect) means down, logged at WARNING.  The judges run
    concurrently; each bounds its own wait.
    """
    if not found:
        return []

    async def run() -> list[bool | BaseException]:
        return await asyncio.gather(*(judge(d) for d in found), return_exceptions=True)

    (task,) = yield from bps.wait_for([run])
    down: list[str] = []
    for device, verdict in zip(found, task.result()):
        if isinstance(verdict, BaseException):
            logger.warning(
                "liveness: %s IOC did not answer (%s: %s) — down",
                device._geecs_device_name,
                type(verdict).__name__,
                " ".join(str(verdict).split()),
            )
        if isinstance(verdict, BaseException) or verdict:
            down.append(device._geecs_device_name)
    return down
