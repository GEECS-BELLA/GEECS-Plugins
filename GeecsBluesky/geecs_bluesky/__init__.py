"""Bluesky / ophyd-async bridge for the GEECS control system.

Devices are CA-backed: they consume the GeecsCAGateway PVs as a standard
EPICS IOC (the gateway is the only component that speaks GEECS TCP/UDP).

The device re-exports below are **lazy** (PEP 562): importing the package
— in particular :mod:`geecs_bluesky.qs_client`, the RE Manager client any
GEECS client uses — must stay light, and the device family pulls
aioca/ophyd-async at import.  ``from geecs_bluesky import CaMotor`` still
works; it just pays the device-family import at that moment.
"""

from .epics_env import apply_epics_address_config

# Must run before any device import: the device family pulls in aioca (via
# ophyd-async), and libca reads EPICS_CA_ADDR_LIST when the CA context is
# created at that import.  Every ``geecs_bluesky.devices`` import runs this
# package init first, so laziness preserves the ordering.  Explicit env
# vars win (setdefault semantics).
apply_epics_address_config()

from .exceptions import (  # noqa: E402
    GeecsError,
    GeecsConnectionError,
    GeecsCommandError,
    GeecsCommandRejectedError,
    GeecsCommandFailedError,
    GeecsTriggerTimeoutError,
    GeecsMotorTimeoutError,
    GeecsDeviceNotFoundError,
)

#: Names served lazily from the (heavy) device family.
_DEVICE_EXPORTS = (
    "CaGenericDetector",
    "CaMotor",
    "CaSettable",
    "CaSnapshotReadable",
    "CaTimestampedReadable",
    "CaTriggerable",
)

__all__ = [
    *_DEVICE_EXPORTS,
    "GeecsError",
    "GeecsConnectionError",
    "GeecsCommandError",
    "GeecsCommandRejectedError",
    "GeecsCommandFailedError",
    "GeecsTriggerTimeoutError",
    "GeecsMotorTimeoutError",
    "GeecsDeviceNotFoundError",
]


def __getattr__(name: str):
    """Resolve the device re-exports on first access (PEP 562)."""
    if name in _DEVICE_EXPORTS:
        from . import devices

        return getattr(devices, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Advertise the lazy device names alongside the eager ones."""
    return sorted(set(globals()) | set(_DEVICE_EXPORTS))
