"""Bluesky / ophyd-async bridge for the GEECS control system.

Devices are CA-backed: they consume the GeecsCAGateway PVs as a standard
EPICS IOC (the gateway is the only component that speaks GEECS TCP/UDP).
"""

from .epics_env import apply_epics_address_config

# Must run before the device imports below: they pull in aioca (via
# ophyd-async), and libca reads EPICS_CA_ADDR_LIST when the CA context is
# created at that import.  Explicit env vars win (setdefault semantics).
apply_epics_address_config()

from .devices import (  # noqa: E402
    CaGenericDetector,
    CaMotor,
    CaSettable,
    CaSnapshotReadable,
    CaTimestampedReadable,
    CaTriggerable,
)
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
