"""
Interlock utilities for GEECS experiment control.

Provides lightweight helper utilities for setting up interlocks:
- Monitor condition builders (ThresholdCheck, AlignmentCheck, MultiCheck, CustomCheck)
- Device monitoring group (DeviceMonitorGroup) with built-in staleness detection
- High-level interlock constructor facade (InterlockConstructor)

BUILT-IN SAFETY: All interlocks automatically fail-safe (return unsafe) if device
data is stale (no update for staleness_timeout_ms). No separate StalenessGuard needed.
"""

from .monitor_conditions import (
    ThresholdCheck,
    AlignmentCheck,
    MultiCheck,
    CustomCheck,
    ConfigurationChangeCheck,
    ExternalStatusCheck,
)
from .device_monitor_group import DeviceMonitorGroup
from .external_status_listener import ExternalStatusListener, default_decode
from .interlock_constructor import InterlockConstructor
from .geecs_interlock_server import InterlockServer
from .interlock_gui import (
    InterlockGUI,
    launch_general_interlock_gui,
    launch_interlock_gui,
)
from .interlock_config import load_configuration, save_configuration

__all__ = [
    "ThresholdCheck",
    "AlignmentCheck",
    "MultiCheck",
    "CustomCheck",
    "ConfigurationChangeCheck",
    "ExternalStatusCheck",
    "DeviceMonitorGroup",
    "ExternalStatusListener",
    "default_decode",
    "InterlockConstructor",
    "InterlockServer",
    "InterlockGUI",
    "launch_general_interlock_gui",
    "launch_interlock_gui",
    "load_configuration",
    "save_configuration",
]
