"""Typed exceptions for the GEECS access layer (wire protocol + database).

All GEECS-specific exceptions inherit from ``GeecsError`` so callers can catch
broadly or narrowly depending on what recovery action is appropriate::

    except GeecsError:            # catch everything GEECS-related
    except GeecsConnectionError:  # only network/transport failures
    except GeecsCommandError:     # device responded but rejected or failed

Scan-level exceptions (trigger/motor timeouts, t0 sync, configuration) live in
``geecs_bluesky.exceptions``, subclassing :class:`GeecsError` from here.
"""

from __future__ import annotations


class GeecsError(Exception):
    """Base class for all GEECS exceptions."""


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


class GeecsDeviceNotFoundError(GeecsError):
    """Device name could not be resolved in the GEECS database.

    Raised by ``GeecsDb.find_device()`` when the device name is not present
    in the MySQL device table.
    """

    def __init__(self, device_name: str) -> None:
        self.device_name = device_name
        super().__init__(f"Device not found in GEECS DB: {device_name!r}")
