"""Bluesky / ophyd-async bridge for the GEECS control system.

Devices are CA-backed: they consume the GeecsCAGateway PVs as a standard
EPICS IOC (the gateway is the only component that speaks GEECS TCP/UDP).
"""

from .devices import (
    CaGenericDetector,
    CaMotor,
    CaSettable,
    CaSnapshotReadable,
    CaTimestampedReadable,
    CaTriggerable,
)
from .exceptions import (
    GeecsError,
    GeecsConnectionError,
    GeecsCommandError,
    GeecsCommandRejectedError,
    GeecsCommandFailedError,
    GeecsTriggerTimeoutError,
    GeecsMotorTimeoutError,
    GeecsDeviceNotFoundError,
)

__all__ = [
    "CaGenericDetector",
    "CaMotor",
    "CaSettable",
    "CaSnapshotReadable",
    "CaTimestampedReadable",
    "CaTriggerable",
    "GeecsError",
    "GeecsConnectionError",
    "GeecsCommandError",
    "GeecsCommandRejectedError",
    "GeecsCommandFailedError",
    "GeecsTriggerTimeoutError",
    "GeecsMotorTimeoutError",
    "GeecsDeviceNotFoundError",
]
