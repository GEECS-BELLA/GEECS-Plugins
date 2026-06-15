"""
Interlock utilities for GEECS experiment control.

Provides lightweight helper utilities for setting up interlocks:
- Monitor condition builders (ThresholdCheck, AlignmentCheck, MultiCheck, CustomCheck)
- Device monitoring group (DeviceMonitorGroup) with built-in staleness detection
- High-level interlock builder facade (InterlockBuilder)

BUILT-IN SAFETY: All interlocks automatically fail-safe (return unsafe) if device
data is stale (no update for staleness_timeout_ms). No separate StalenessGuard needed.
"""

from .monitor_conditions import (
    ThresholdCheck,
    AlignmentCheck,
    MultiCheck,
    CustomCheck,
)
from .device_monitor_group import DeviceMonitorGroup
from .interlock_builder import InterlockBuilder
from .geecs_interlock_server import InterlockServer

__all__ = [
    'ThresholdCheck',
    'AlignmentCheck',
    'MultiCheck',
    'CustomCheck',
    'DeviceMonitorGroup',
    'InterlockBuilder',
    'InterlockServer',
]
