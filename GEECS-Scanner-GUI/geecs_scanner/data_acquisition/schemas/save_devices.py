"""
Schemas for configuring device saving and experimental workflow management.

This module provides comprehensive Pydantic models for defining device
configuration strategies during experimental data acquisition. It supports
flexible device setup, variable tracking, and pre/post-scan actions.

Key Components:
- Device-specific configuration management
- Synchronous and asynchronous logging strategies
- Scalar and non-scalar data handling
- Pre and post-scan action sequences

The save devices system enables:
- Precise device variable tracking
- Configurable data saving strategies
- Automated device setup and teardown
- Flexible experimental workflow definition

Examples
--------
>>> device_config = DeviceConfig(
...     synchronous=True,
...     save_nonscalar_data=True,
...     variable_list=['temperature', 'power']
... )
>>> save_config = SaveDeviceConfig(
...     Devices={'Laser': device_config},
...     scan_info={'experiment': 'beam_characterization'}
... )

See Also
--------
geecs_scanner.data_acquisition.device_manager : Device management system
geecs_scanner.data_acquisition.schemas.actions : Action sequence definitions
"""

from __future__ import annotations
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

from geecs_scanner.data_acquisition.schemas.actions import ActionSequence


class DeviceConfig(BaseModel):
    """
    Comprehensive configuration for a single device during scan acquisition and data saving.

    This model provides granular control over device data collection,
    logging strategies, and variable tracking during experimental workflows.

    Attributes
    ----------
    synchronous : bool, optional
        If True, device is treated as synchronous with step-by-step logging.
        Defaults to False, enabling more flexible asynchronous data collection.
    save_nonscalar_data : bool, optional
        Controls whether non-scalar data (e.g., images) should be saved.
        Defaults to False, focusing on scalar measurements.
    variable_list : List[str], optional
        Explicit list of variables to acquire from the device.
        If None, defaults to device-specific or system-wide variable tracking.
    post_analysis_class : str, optional
        DEPRECATED. Previously used for post-acquisition data processing.
    scan_setup : Dict[str, List[str]], optional
        Configures device-specific setup and teardown values.
        Each key maps to a two-element list: [pre-scan value, post-scan value].

    Notes
    -----
    DeviceConfig enables:
    - Flexible logging mode selection
    - Granular data saving control
    - Explicit variable tracking
    - Device-specific pre/post-scan configuration

    Examples
    --------
    >>> laser_config = DeviceConfig(
    ...     synchronous=True,
    ...     save_nonscalar_data=False,
    ...     variable_list=['power', 'wavelength'],
    ...     scan_setup={'mode': ['standby', 'off']}
    ... )
    >>> # Configures laser device for synchronous, scalar data collection

    See Also
    --------
    SaveDeviceConfig : Overall device saving configuration
    ActionSequence : Define pre/post-scan actions
    """

    synchronous: Optional[bool] = Field(
        False,
        description="If True, device is treated as synchronous (step-by-step logging).",
    )
    save_nonscalar_data: Optional[bool] = Field(
        False, description="Whether to save non-scalar data such as images."
    )
    variable_list: Optional[List[str]] = Field(
        None, description="List of variables to acquire from the device."
    )

    add_all_variables: Optional[bool] = Field(
        False,
        description="If True, include all device variables (overridden by variable_list if provided).",
    )

    post_analysis_class: Optional[str] = Field(
        None, description="Deprecated post-analysis processing class."
    )
    scan_setup: Optional[Dict[str, List[str]]] = Field(
        None,
        description=(
            "Optional dictionary mapping setup variables to pre-scan and post-scan values. "
            "Each entry is a two-element list: [pre-scan value, post-scan value]."
        ),
    )


class SaveDeviceConfig(BaseModel):
    """
    Comprehensive configuration schema for experimental device saving and workflow management.

    This model provides a complete framework for defining device configurations,
    data saving strategies, and pre/post-scan actions during experimental workflows.

    Attributes
    ----------
    Devices : Dict[str, DeviceConfig], optional
        Mapping of device names to their specific configuration.
        Allows granular control over individual device data collection.
    scan_info : Dict[str, str], optional
        Additional metadata or descriptive information about the scan.
    setup_action : ActionSequence, optional
        Sequence of actions to perform before initiating the scan.
        Enables complex pre-scan device preparation and system configuration.
    closeout_action : ActionSequence, optional
        Sequence of actions to perform after completing the scan.
        Supports systematic device shutdown, data processing, or system reset.

    Notes
    -----
    SaveDeviceConfig provides:
    - Comprehensive device configuration management
    - Flexible pre and post-scan action sequences
    - Extensible scan metadata tracking
    - Modular experimental workflow definition

    Examples
    --------
    >>> save_config = SaveDeviceConfig(
    ...     Devices={
    ...         'Laser': DeviceConfig(synchronous=True),
    ...         'Spectrometer': DeviceConfig(save_nonscalar_data=True)
    ...     },
    ...     scan_info={'experiment': 'beam_characterization'},
    ...     setup_action=ActionSequence(steps=[...]),
    ...     closeout_action=ActionSequence(steps=[...])
    ... )
    >>> # Creates a comprehensive device saving configuration

    See Also
    --------
    DeviceConfig : Individual device configuration
    ActionSequence : Define pre/post-scan action sequences
    geecs_scanner.data_acquisition.device_manager : Device management system
    """

    Devices: Optional[Dict[str, DeviceConfig]] = Field(
        None, description="Mapping of device names to their specific configuration."
    )
    scan_info: Optional[Dict[str, str]] = Field(
        None,
        description="Additional metadata or descriptive information about the scan.",
    )
    setup_action: Optional[ActionSequence] = Field(
        None, description="Sequence of actions to perform before initiating the scan."
    )
    closeout_action: Optional[ActionSequence] = Field(
        None, description="Sequence of actions to perform after completing the scan."
    )
