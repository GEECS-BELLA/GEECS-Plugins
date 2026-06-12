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
from typing import Any

from bluesky.protocols import Reading
from event_model import DataKey

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
        shot_id = self.peek(acq_timestamp)
        if shot_id is not None and float(acq_timestamp) > self._last_acq_timestamp:  # type: ignore[operator]
            self._last_acq_timestamp = float(acq_timestamp)
            self._last_shot_id = shot_id
        return shot_id

    def peek(self, acq_timestamp: float) -> int | None:
        """Return the shot ID :meth:`update` would assign, without advancing.

        Used by grace-wait loops to test whether a device's cache has reached
        a target shot yet, without committing tracker state.
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
        return self._last_shot_id + max(delta, 1)


class ShotIdSupport:
    """Mixin for GEECS devices that derive shot IDs from their ``acq_timestamp``.

    Cooperates with :class:`~geecs_bluesky.devices.geecs_device.GeecsDevice`
    (uses its ``_shot_cache``).  Hosts gain ``configure_shot_id()`` /
    ``seed_shot_id()`` and the ``last_acq_timestamp`` accessor used by the
    coordinated t0-sync stage (:func:`~geecs_bluesky.plans.t0_sync.geecs_t0_sync`).
    """

    _acq_timestamp_variable: str = "acq_timestamp"
    _shot_id_tracker: ShotIdTracker | None = None

    def configure_shot_id(
        self,
        rep_rate_hz: float,
        t0_acq_timestamp: float | None = None,
    ) -> None:
        """Enable shot-ID derivation and the sync-device companion columns.

        Parameters
        ----------
        rep_rate_hz:
            Free-running external trigger repetition rate in Hz.  Invalid
            values disable shot IDs (with a warning) rather than raising, so
            a misconfigured scan still collects data.
        t0_acq_timestamp:
            Optional device acquisition timestamp for physical shot 1.
            Normally seeded later by the coordinated t0-sync stage.
        """
        name = getattr(self, "name", "?")
        if rep_rate_hz <= 0:
            logger.warning(
                "Shot IDs disabled for %s: invalid rep_rate_hz=%s",
                name,
                rep_rate_hz,
            )
            self._shot_id_tracker = None
            return
        self._shot_id_tracker = ShotIdTracker(rep_rate_hz)
        if t0_acq_timestamp is not None:
            self._shot_id_tracker.seed(t0_acq_timestamp)
        logger.info(
            "Shot IDs configured for %s: rep_rate_hz=%s, t0=%s",
            name,
            rep_rate_hz,
            t0_acq_timestamp if t0_acq_timestamp is not None else "(deferred)",
        )

    def seed_shot_id(self, t0_acq_timestamp: float) -> None:
        """Define ``t0_acq_timestamp`` as this device's physical shot 1.

        Called by the coordinated t0-sync stage.  Requires
        :meth:`configure_shot_id` first.
        """
        if self._shot_id_tracker is None:
            raise RuntimeError(
                f"{getattr(self, 'name', '?')}: configure_shot_id() must be "
                "called before seed_shot_id()"
            )
        self._shot_id_tracker.seed(t0_acq_timestamp)

    @property
    def shot_id_tracker(self) -> ShotIdTracker | None:
        """Shot-ID tracker, or ``None`` when shot IDs are not configured."""
        return self._shot_id_tracker

    @property
    def last_acq_timestamp(self) -> float | None:
        """Most recent ``acq_timestamp`` from the TCP cache, if any."""
        cache = getattr(self, "_shot_cache", None)
        if cache is None:
            return None
        return coerce_timestamp(cache.get(self._acq_timestamp_variable))

    def _shot_id_datakeys(self) -> dict[str, DataKey]:
        """Describe the derived companion columns (schema contract v1).

        ``shot_id`` and ``shot_offset`` are dtype ``number``, not ``integer``:
        the missing case is NaN.
        """
        prefix = getattr(self, "name", "")
        keys: dict[str, DataKey] = {}
        for suffix, dtype in (
            ("t0_acq_timestamp", "number"),
            ("shot_id", "number"),
            ("shot_offset", "number"),
            ("valid", "boolean"),
        ):
            keys[f"{prefix}-{suffix}"] = {
                "source": f"derived://{prefix}/{suffix}",
                "dtype": dtype,
                "shape": [],
            }
        return keys

    def _emit_shot_id_readings(
        self,
        reading: dict[str, Reading],
        event_timestamp: float,
        shot_id: int | None,
        shot_offset: int | None,
    ) -> None:
        """Add the companion-column Readings to *reading* in place.

        Keys are stable: unavailable values are NaN, with ``valid=False``.
        ``valid`` is ``shot_offset == 0`` — this device's data belongs to the
        row's physical shot.
        """
        prefix = getattr(self, "name", "")
        tracker = self._shot_id_tracker
        t0 = tracker.t0_acq_timestamp if tracker is not None else None
        values: dict[str, Any] = {
            "t0_acq_timestamp": t0 if t0 is not None else float("nan"),
            "shot_id": shot_id if shot_id is not None else float("nan"),
            "shot_offset": shot_offset if shot_offset is not None else float("nan"),
            "valid": shot_offset == 0,
        }
        for suffix, value in values.items():
            reading[f"{prefix}-{suffix}"] = Reading(
                value=value,
                timestamp=event_timestamp,
                alarm_severity=0,
            )


def coerce_timestamp(value: Any) -> float | None:
    """Return ``value`` as float, or ``None`` when unavailable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
