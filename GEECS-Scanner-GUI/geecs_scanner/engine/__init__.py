"""
Engine package for GEECS-Scanner.

This package re-exports common classes so callers can write:
    from geecs_scanner.engine import ActionManager, DatabaseDictLookup

The legacy scan engine (ScanManager, DataLogger, DeviceManager,
ScanStepExecutor, ScanDataManager, TriggerController, FileMover,
ScanLifecycleStateMachine) was deleted in G1 of the greenfield cutover —
see ``Planning/cutover_strategy/00_overview.md``. BlueskyScanner
(GeecsBluesky) is the only scan backend. What remains here is the
non-scan-path surface: actions, the device-command policy point, the DB
lookup, the event/dialog shims, and the GUI→backend config models.
"""

__all__ = [
    "ActionManager",
    "DeviceCommandExecutor",
    "DatabaseDictLookup",
    "ScanState",
    "ScanEvent",
    "ScanLifecycleEvent",
    "ScanStepEvent",
    "DeviceCommandEvent",
    "ScanErrorEvent",
    "ScanRestoreFailedEvent",
    "ScanDialogEvent",
    "get_full_config_path",
    "visa_config_generator",
]

from .action_manager import ActionManager
from .device_command_executor import DeviceCommandExecutor
from .database_dict_lookup import DatabaseDictLookup
from .scan_events import (
    ScanState,
    ScanEvent,
    ScanLifecycleEvent,
    ScanStepEvent,
    DeviceCommandEvent,
    ScanErrorEvent,
    ScanRestoreFailedEvent,
    ScanDialogEvent,
)
from ..utils.config_utils import get_full_config_path, visa_config_generator
