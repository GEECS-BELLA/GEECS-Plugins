"""ShotIdTracker — per-device physical trigger-opportunity numbering.

A device's *shot ID* is the index of the external-trigger tick its acquisition
belongs to.  It is derived from the device's own ``acq_timestamp`` history, so
it is immune to clock skew between control machines: two devices saw the same
physical trigger if and only if their shot IDs are equal (given t0s captured
on the same physical shot — see :mod:`geecs_bluesky.plans.t0_sync`).

The ID advances **incrementally**::

    delta = round((acq_timestamp - last_acq_timestamp) * rep_rate_hz)
    shot_id = last_shot_id + max(delta, 1)

rather than absolutely from t0.  Absolute derivation accumulates rep-rate
error over a run (a 0.05% rate mismatch misquantizes after ~30 minutes at
1 Hz); incremental derivation resets the error on every shot.

Shot IDs are matching machinery and diagnostics, **not** a file-join key —
files join to events by device ``acq_timestamp``.  Jumps greater than 1
across stage-move dead time are expected; cross-device matching is equality,
never consecutiveness.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ShotIdTracker:
    """Incremental shot-ID derivation for one sync device.

    Parameters
    ----------
    rep_rate_hz:
        Free-running external trigger repetition rate in Hz.  Must be > 0.

    Usage::

        tracker = ShotIdTracker(rep_rate_hz=1.0)
        tracker.seed(t0_acq_timestamp)            # from the t0-sync stage
        shot_id = tracker.update(acq_timestamp)   # after each shot
    """

    def __init__(self, rep_rate_hz: float) -> None:
        if rep_rate_hz <= 0:
            raise ValueError(f"rep_rate_hz must be > 0, got {rep_rate_hz}")
        self._rep_rate_hz = float(rep_rate_hz)
        self._t0_acq_timestamp: float | None = None
        self._last_acq_timestamp: float | None = None
        self._last_shot_id: int | None = None

    @property
    def rep_rate_hz(self) -> float:
        """Configured external trigger repetition rate in Hz."""
        return self._rep_rate_hz

    @property
    def is_seeded(self) -> bool:
        """Whether a t0 has been captured."""
        return self._t0_acq_timestamp is not None

    @property
    def t0_acq_timestamp(self) -> float | None:
        """Device ``acq_timestamp`` defined as physical shot 1."""
        return self._t0_acq_timestamp

    @property
    def current_shot_id(self) -> int | None:
        """Shot ID of the most recent :meth:`update` (or 1 right after seeding)."""
        return self._last_shot_id

    def seed(self, t0_acq_timestamp: float) -> None:
        """Define ``t0_acq_timestamp`` as physical shot 1.

        Re-seeding resets the tracker (e.g. a fresh t0-sync stage).
        """
        self._t0_acq_timestamp = float(t0_acq_timestamp)
        self._last_acq_timestamp = float(t0_acq_timestamp)
        self._last_shot_id = 1

    def update(self, acq_timestamp: float) -> int | None:
        """Advance to ``acq_timestamp`` and return its shot ID.

        Idempotent for a repeated timestamp (device timed out — no new shot):
        returns the unchanged current ID.  Returns ``None`` when unseeded.
        A timestamp earlier than the last seen one is logged and ignored.
        """
        if self._last_acq_timestamp is None or self._last_shot_id is None:
            return None
        ts = float(acq_timestamp)
        if ts == self._last_acq_timestamp:
            return self._last_shot_id
        if ts < self._last_acq_timestamp:
            logger.warning(
                "acq_timestamp went backwards (%s < %s); keeping shot_id=%d",
                ts,
                self._last_acq_timestamp,
                self._last_shot_id,
            )
            return self._last_shot_id
        delta = round((ts - self._last_acq_timestamp) * self._rep_rate_hz)
        self._last_shot_id += max(delta, 1)
        self._last_acq_timestamp = ts
        return self._last_shot_id
