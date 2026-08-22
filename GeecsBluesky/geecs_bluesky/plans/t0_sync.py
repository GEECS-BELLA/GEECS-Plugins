"""geecs_t0_sync — coordinated t0 capture for cross-device shot matching.

With the shot-control outputs disarmed, every sync device's cache holds the
frame from the last physical trigger.  The control machines are NTP-synced
far better than the trigger period, so cached ``acq_timestamp`` values
within a small acceptance window came from the **same physical trigger** —
each device's value becomes its own t0 (physical shot 1), making derived
shot IDs directly comparable across devices (see
:class:`~geecs_bluesky.devices.shot_id.ShotIdTracker`).

Run once per free-run scan, before the first trigger arm.  Strict-mode scans
may run it too; if skipped there, devices self-seed on their first awaited
shot.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import bluesky.plan_stubs as bps

from geecs_bluesky.exceptions import GeecsT0SyncError
from geecs_bluesky.plans.liveness import rd_confirmed_down

logger = logging.getLogger(__name__)


def _refuse_disconnected_devices(devices: Sequence[Any]):
    """Plan stub: refuse to seed from a device the gateway reports down.

    The seed comes from each device's **cached** ``acq_timestamp`` under a
    quiescent trigger, and a dead device serves its stale cache forever —
    with one sync device (or all dead) the spread check is trivially
    satisfied and a dead reference "seeds" from an hours-old frame,
    deferring the failure to a cryptic mid-scan trigger timeout (#664,
    live incident 2026-08-22: a camera server rebooted uncleanly).
    Timestamp freshness cannot be the guard — at rest with the trigger
    parked (e.g. laser-off operation) a *healthy* cache is legitimately
    old — so liveness is read from ``connected_status`` via the shared
    :func:`~geecs_bluesky.plans.liveness.rd_confirmed_down` (fail-open;
    only the exact ``Disconnected`` choice string refuses, naming the
    device(s)).  Every device here is synchronous, so any dead one would
    also fail the spread check — refusing them all pre-seed, by name, is
    the honest version of that failure.
    """
    down: list[str] = []
    for dev in devices:
        confirmed = yield from rd_confirmed_down(dev)
        if confirmed:
            down.append(getattr(dev, "_geecs_device_name", dev.name))
    if down:
        raise GeecsT0SyncError(
            f"cannot seed t0: the gateway reports {', '.join(sorted(down))} "
            "as Disconnected — the cached acq_timestamp is stale, not a "
            "shot; restart the device(s) and resubmit"
        )


def geecs_t0_sync(
    devices: Sequence[Any],
    window_s: float = 0.2,
    retries: int = 2,
    retry_wait_s: float = 1.2,
):
    """Plan stub: seed every device's shot-ID tracker from one physical shot.

    Parameters
    ----------
    devices:
        Sync devices exposing ``last_acq_timestamp`` and ``seed_shot_id()``
        (e.g. :class:`~geecs_bluesky.devices.ca.generic_detector.CaGenericDetector`).
    window_s:
        Acceptance window for the timestamp spread.  Default ``0.2`` —
        comfortably above NTP skew (~50 ms) and far below a 1 s trigger
        period.
    retries:
        Re-check attempts when a device has no cached timestamp yet or the
        spread exceeds the window (e.g. frames still propagating after the
        trigger was just disarmed).
    retry_wait_s:
        Plan-level sleep between attempts.

    Returns
    -------
    dict[str, float]
        GEECS device name → captured t0 ``acq_timestamp``.  Record this in
        run metadata as ``device_t0s``.

    Raises
    ------
    GeecsT0SyncError
        If a common physical shot cannot be established.  Never proceed
        unseeded — shot IDs from unsynchronized t0s are not comparable.
    """
    # Liveness gate first (#664): a dead device's cache would "seed" —
    # see _refuse_disconnected_devices.
    yield from _refuse_disconnected_devices(devices)
    last_error = ""
    timestamps: dict[str, float | None] = {}
    for attempt in range(retries + 1):
        if attempt > 0:
            yield from bps.sleep(retry_wait_s)
        timestamps = {
            getattr(dev, "_geecs_device_name", dev.name): dev.last_acq_timestamp
            for dev in devices
        }
        missing = [name for name, ts in timestamps.items() if ts is None]
        if missing:
            last_error = f"no cached acq_timestamp for: {', '.join(missing)}"
            logger.info("t0 sync attempt %d: %s", attempt + 1, last_error)
            continue
        values = [ts for ts in timestamps.values() if ts is not None]
        spread = max(values) - min(values)
        if spread > window_s:
            # Name the laggards: a dead/off device serves a stale cached
            # timestamp forever, and "spread exceeds window" alone doesn't
            # say which of N devices to go look at.
            newest = max(values)
            newest_name = next(name for name, ts in timestamps.items() if ts == newest)
            stale = ", ".join(
                f"{name} ({newest - ts:.3f}s behind {newest_name})"
                for name, ts in sorted(timestamps.items(), key=lambda kv: kv[1] or 0.0)
                if ts is not None and newest - ts > window_s
            )
            last_error = (
                f"acq_timestamp spread {spread:.3f}s exceeds window "
                f"{window_s:.3f}s — cached frames are not from one trigger; "
                f"stale device(s): {stale}"
            )
            logger.info("t0 sync attempt %d: %s", attempt + 1, last_error)
            continue
        for dev in devices:
            name = getattr(dev, "_geecs_device_name", dev.name)
            dev.seed_shot_id(timestamps[name])
        logger.info(
            "t0 sync complete: %d devices seeded, spread %.3fs", len(devices), spread
        )
        return {name: ts for name, ts in timestamps.items() if ts is not None}

    raise GeecsT0SyncError(
        f"t0 sync failed after {retries + 1} attempts: {last_error}",
        timestamps=timestamps,
        window_s=window_s,
    )
