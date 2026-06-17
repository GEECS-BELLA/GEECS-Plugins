"""geecs_t0_sync — coordinated t0 capture for cross-device shot matching.

Formalizes the fast t0 procedure from the legacy DAQ: with the shot-control
outputs disarmed (devices no longer firing), every sync device's TCP cache
holds the frame from the last physical trigger.  The control machines are
NTP-synced far better than the trigger period, so if the cached
``acq_timestamp`` values all fall within a small acceptance window they were
produced by the **same physical trigger** — each device's value becomes its
own t0 (physical shot 1), and shot IDs derived from those t0s are directly
comparable across devices (see
:class:`~geecs_bluesky.devices.shot_id.ShotIdTracker`).

Run this stage once per free-run scan, after ``open_run`` is prepared but
before the first trigger arm.  Strict-mode scans may run it too so their rows
carry comparable shot IDs; if skipped there, devices self-seed on their first
awaited shot.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import bluesky.plan_stubs as bps

from geecs_bluesky.exceptions import GeecsT0SyncError

logger = logging.getLogger(__name__)


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
        (e.g. :class:`~geecs_bluesky.devices.generic_detector.GeecsGenericDetector`).
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
            last_error = (
                f"acq_timestamp spread {spread:.3f}s exceeds window "
                f"{window_s:.3f}s — cached frames are not from one trigger"
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
