"""
Models package for GEECS engine.

Re-exports key Pydantic models for action definitions, device configuration,
and scan options.
"""

__all__ = [
    "ActionLibrary",
    "ActionSequence",
    "SetStep",
    "GetStep",
    "WaitStep",
    "ExecuteStep",
    "RunStep",
    "SaveDeviceConfig",
    "DeviceConfig",
    "ScanExecutionConfig",
    "ScanOptions",
]

# The action/save-device models moved to geecs_bluesky.optimization with the
# Xopt/evaluator relocation (they travel with their one live consumer; this
# package deletes whole at M6).  Re-export shims keep the legacy engine and
# its tests importing from the historical path until then.
from geecs_bluesky.optimization._legacy_models_actions import (
    ActionLibrary,
    ActionSequence,
    ExecuteStep,
    GetStep,
    RunStep,
    SetStep,
    WaitStep,
)
from geecs_bluesky.optimization._legacy_models_save_devices import (
    DeviceConfig,
    SaveDeviceConfig,
)
from .scan_execution_config import ScanExecutionConfig
from .scan_options import ScanOptions
