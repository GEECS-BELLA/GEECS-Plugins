"""Typed exception hierarchy for GeecsBluesky (scan level).

Wire-protocol and database exceptions (``GeecsError`` base,
``GeecsConnectionError``, ``GeecsCommandError`` and subclasses,
``GeecsDeviceNotFoundError``) live in :mod:`geecs_ca_gateway.exceptions` — the
GEECS access layer — and are re-exported here so existing imports keep working.
Scan-level exceptions below subclass the same ``GeecsError`` base.
"""

from __future__ import annotations

from geecs_ca_gateway.exceptions import (
    GeecsCommandError,
    GeecsCommandFailedError,
    GeecsCommandRejectedError,
    GeecsConnectionError,
    GeecsDeviceNotFoundError,
    GeecsError,
)

__all__ = [
    "GeecsError",
    "GeecsConnectionError",
    "GeecsCommandError",
    "GeecsCommandRejectedError",
    "GeecsCommandFailedError",
    "GeecsDeviceNotFoundError",
    "GeecsTriggerTimeoutError",
    "GeecsQuiescenceTimeoutError",
    "GeecsMotorTimeoutError",
    "GeecsT0SyncError",
    "GeecsConfigurationError",
    "GeecsDeviceDownError",
    "GeecsStaleDevicesError",
]


# ---------------------------------------------------------------------------
# Timeout errors — operation started but did not complete in time
# ---------------------------------------------------------------------------


class GeecsTriggerTimeoutError(GeecsError):
    """``acq_timestamp`` did not advance within the trigger timeout.

    Raised by :class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable`
    when no new shot arrives within ``_trigger_timeout`` seconds.  Typical
    causes: DG645 not firing, camera not acquiring, or trigger cable fault.
    """

    def __init__(self, device_name: str, timeout: float, message: str = "") -> None:
        self.device_name = device_name
        self.timeout = timeout
        super().__init__(message or f"{device_name}: no shot within {timeout:.1f}s")


class GeecsQuiescenceTimeoutError(GeecsError):
    """Free-running trigger did not stop within the timeout.

    Raised by :func:`~geecs_bluesky.plans.single_shot.geecs_confirm_quiescent`
    when device ``acq_timestamp`` values keep advancing after the shot
    controller was put in single-shot (``ARMED``) mode — so plan-owned
    single-shot firing cannot safely begin (a residual free-running shot would
    be mistaken for the plan's fired shot).  Typical cause: the ``ARMED`` state
    did not actually switch the trigger source to single-shot mode.
    """

    def __init__(self, timeout: float, message: str = "") -> None:
        self.timeout = timeout
        super().__init__(
            message
            or f"trigger still firing after {timeout:.1f}s; single-shot arm failed"
        )


class GeecsMotorTimeoutError(GeecsError):
    """Motor did not reach the target position within ``move_timeout``.

    Raised by :class:`~geecs_bluesky.devices.motor.GeecsMotor` when the
    position polling loop expires.  Possible causes: stage stall, mechanical
    obstruction, wrong tolerance, or very long move.  Do not auto-retry —
    a stalled stage may need operator intervention.
    """

    def __init__(
        self,
        device_name: str,
        variable: str,
        target: float,
        current: float,
        timeout: float,
    ) -> None:
        self.device_name = device_name
        self.variable = variable
        self.target = target
        self.current = current
        self.timeout = timeout
        super().__init__(
            f"{device_name}/{variable}: position {current} did not reach "
            f"{target} within {timeout:.1f}s"
        )


class GeecsT0SyncError(GeecsError):
    """Coordinated t0 capture could not establish a common physical shot.

    Raised by :func:`~geecs_bluesky.plans.t0_sync.geecs_t0_sync` when device
    ``acq_timestamp`` values are spread wider than the acceptance window (the
    cached frames do not all come from the same physical trigger) or when a
    device has no cached ``acq_timestamp`` at all.  Never proceed unseeded —
    shot IDs from unsynchronized t0s are not comparable across devices.
    """

    def __init__(
        self,
        message: str,
        timestamps: dict[str, float | None] | None = None,
        window_s: float | None = None,
    ) -> None:
        self.timestamps = timestamps or {}
        self.window_s = window_s
        super().__init__(message)


# ---------------------------------------------------------------------------
# Configuration / setup errors
# ---------------------------------------------------------------------------


class GeecsConfigurationError(GeecsError):
    """Runtime configuration is incomplete or inconsistent."""


class GeecsDeviceDownError(GeecsError):
    """A device's gateway ``CONNECTED`` PV reports ``Disconnected``.

    The gateway serves every DB device's PVs whether or not the device's TCP
    stream is up, so CA-connect success never implied device liveness; the
    per-device ``[Experiment:]Device:CONNECTED`` status PV is the
    authoritative signal (PV_CONTRACT.md §1/§5).  Raised (or carried inside
    the pre-claim operator dialog) when that PV says a device is down:
    by ``BlueskyScanner._preflight_check_sync_liveness`` before a scan, and
    by :func:`~geecs_bluesky.plans.single_shot.geecs_single_shot` when a
    no-frame device turns out to be disconnected mid-scan (re-firing cannot
    help a dead device).  The message is operator-facing.
    """

    def __init__(self, message: str, device_name: str | None = None) -> None:
        self.device_name = device_name
        super().__init__(message)


class GeecsStaleDevicesError(GeecsError):
    """Free-run sync device(s) are CONNECTED but have no fresh frames.

    Carried inside the pre-claim operator dialog raised by
    ``BlueskyScanner._preflight_check_sync_liveness`` (free-run mode only,
    after the gateway-liveness stage passed) when cached ``acq_timestamp``
    frames are missing or too old — all-stale means the trigger is probably
    off / not free-running; a stale subset is a per-device acquisition
    problem.  Genuinely *dead* devices are :class:`GeecsDeviceDownError`
    territory (the ``CONNECTED`` PV), not this.  The message is
    operator-facing: it names the stale device(s), how stale they are, and
    what the dialog's options mean.
    """
