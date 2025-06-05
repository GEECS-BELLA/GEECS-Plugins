from __future__ import annotations
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

from geecs_scanner.data_acquisition.schemas.actions import ActionSequence


class DeviceConfig(BaseModel):
    """
    Configuration for a single device to be used during scan acquisition and saving.
    """
    synchronous: bool = Field(..., description="If True, device is treated as synchronous (step-by-step logging).")
    save_nonscalar_data: bool = Field(..., description="Whether to save non-scalar data such as images.")
    variable_list: List[str] = Field(..., description="List of variables to acquire from the device.")
    post_analysis_class: Optional[str] = Field(
        None,
        description="Deprecated."
    )
    scan_setup: Optional[Dict[str, List[str]]] = Field(
        None,
        description=(
            "Optional dictionary mapping setup variables (e.g., 'Analysis') to a two-element list: "
            "the first element is the value to set before the scan (e.g. 'On'), and the second is the value "
            "to set after the scan (e.g. 'Off')."
        )
    )


class SaveDeviceConfig(BaseModel):
    """
    Full configuration schema for scan-time device saving and setup actions.

    This is typically loaded by ScanManager or DeviceManager. It defines which devices to save,
    which variables to acquire from each device, and any pre-scan setup actions.
    """
    Devices: Dict[str, DeviceConfig] = Field(..., description="Dictionary of device names to their configurations.")
    setup_action: ActionSequence = Field(..., description="Action steps to run before the scan starts.")
    closeout_action: ActionSequence = Field(..., description="Action steps to run after the scan ends.")
