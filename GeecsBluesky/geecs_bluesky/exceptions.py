"""Typed exception hierarchy for GeecsBluesky.

All GEECS-specific exceptions inherit from ``GeecsError`` so callers can catch
broadly or narrowly depending on what recovery action is appropriate::

    except GeecsError:          # catch everything GEECS-related
    except GeecsConnectionError:  # only network/transport failures
    except GeecsCommandError:   # device responded but rejected or failed

Design rules
------------
- Devices raise typed exceptions; they know nothing about dialogs or queues.
- Retry policy and dialog logic live in the plan wrapper (``bluesky_scanner``).
- ``GeecsCommandError`` and subclasses always carry ``device_name`` and
  ``variable`` so the error dialog can name the offending device/variable
  without parsing message strings.
"""

from __future__ import annotations


class GeecsError(Exception):
    """Base class for all GeecsBluesky exceptions."""


# ---------------------------------------------------------------------------
# Transport / connectivity
# ---------------------------------------------------------------------------


class GeecsConnectionError(GeecsError):
    """Device is unreachable at the transport level.

    Raised when a UDP or TCP connection cannot be established or a socket
    operation times out before any protocol-level response is received.
    Distinct from :class:`GeecsCommandRejectedError` where the device did
    respond but refused the command.
    """


# ---------------------------------------------------------------------------
# Command errors — device responded but the command did not succeed
# ---------------------------------------------------------------------------


class GeecsCommandError(GeecsError):
    """Base class for errors where the device was reached but the command failed.

    Parameters
    ----------
    device_name:
        GEECS device name (e.g. ``"UC_ModeImager"``).
    variable:
        Variable or command that was attempted.
    message:
        Human-readable description of what went wrong.
    """

    def __init__(self, device_name: str, variable: str, message: str = "") -> None:
        self.device_name = device_name
        self.variable = variable
        super().__init__(
            f"{device_name}/{variable}: {message}"
            if message
            else f"{device_name}/{variable}"
        )


class GeecsCommandRejectedError(GeecsCommandError):
    """Device did not acknowledge the command (no ACK within timeout).

    Typical causes: device process not running, device busy, malformed command,
    or UDP packet lost with no retry opportunity.  Distinct from a *missed*
    packet (handled by the UDP timeout) in that the device may have received
    the command but chosen not to respond.
    """


class GeecsCommandFailedError(GeecsCommandError):
    """Device acknowledged the command but reported an error in the exe response.

    The device received and understood the command but could not execute it
    (e.g. out-of-range value, hardware interlock, internal driver error).
    """


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


# ---------------------------------------------------------------------------
# Configuration / setup errors
# ---------------------------------------------------------------------------


class GeecsDeviceNotFoundError(GeecsError):
    """Device name could not be resolved in the GEECS database.

    Raised by ``GeecsDb.find_device()`` when the device name is not present
    in the MySQL device table.
    """

    def __init__(self, device_name: str) -> None:
        self.device_name = device_name
        super().__init__(f"Device not found in GEECS DB: {device_name!r}")
