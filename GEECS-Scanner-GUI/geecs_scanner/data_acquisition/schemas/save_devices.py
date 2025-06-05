from __future__ import annotations
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

from geecs_scanner.data_acquisition.schemas.actions import ActionSequence


class DeviceConfig(BaseModel):
    """
    Configuration for a single device to be used during scan acquisition and saving.
    """
    synchronous: Optional[bool] = Field(False, description="If True, device is treated as synchronous (step-by-step logging).")
    save_nonscalar_data: Optional[bool] = Field(False, description="Whether to save non-scalar data such as images.")
    variable_list: Optional[List[str]] = Field(None, description="List of variables to acquire from the device.")
    post_analysis_class: Optional[str] = Field(None,description="Deprecated.")
    scan_setup: Optional[Dict[str, List[str]]] = Field(None,
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
    Devices: Optional[Dict[str, DeviceConfig]] = Field(
        None,
        description="(Optional) Mapping of device name → device config"
    )
    scan_info: Optional[str] = Field('additional description')
    setup_action: Optional[ActionSequence] = Field(
        None,
        description="(Optional) Steps to perform before a scan"
    )
    closeout_action: Optional[ActionSequence] = Field(
        None,
        description="(Optional) Steps to perform after a scan"
    )
