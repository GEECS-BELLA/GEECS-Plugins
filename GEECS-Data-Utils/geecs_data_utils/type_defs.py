"""
Type definitions for GEECS scan configuration and modes.

This module defines the core data structures used throughout the GEECS
plugin suite for configuring and managing experimental scans.

Contains enumerations for scan modes and dataclass definitions for
scan configuration parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Any
from enum import Enum


class ScanMode(str, Enum):
    """
    Enumeration of available scan modes in the GEECS-Scanner-GUI.

    Defines the different types of scans that can be performed,
    from standard parameter sweeps to optimization routines.

    Attributes
    ----------
    STANDARD : str
        Standard 1D parameter scan with defined start, end, step
    NOSCAN : str
        No parameter variation. Data collection for statistics
    OPTIMIZATION : str
        Optimization-driven scan using external optimizer
    BACKGROUND : str
        Background measurement for calibration purposes. Merely sets
        a flag in the scan_info.ini file that can be used for
        downstream purposes
    """

    STANDARD = "standard"
    NOSCAN = "noscan"
    OPTIMIZATION = "optimization"
    BACKGROUND = "background"


@dataclass
class ScanConfig:
    """
    Configuration dataclass for GEECS scan parameters.

    Encapsulates all parameters needed to configure and execute
    a scan in the python-base GEECS-Scanner, including scan mode,
    device settings, parameter ranges, and optimization configurations.

    Attributes
    ----------
    scan_mode : ScanMode
        Type of scan to perform (default is NOSCAN)
    device_var : str, optional
        GEECS device variable to scan (default is None). Note, the
        style should be 'device_name:device_variable', where
        device_variable is the non aliased name
    start : Union[int, float]
        Starting value for parameter scan (default is 0)
    end : Union[int, float]
        Ending value for parameter scan (default is 1)
    step : Union[int, float]
        Step size for parameter scan (default is 1)
    wait_time : float
        Wait time between scan points in seconds (default is 1.0)
    additional_description : str, optional
        Additional description for the scan (default is None). This
        description is added to the scan_info file
    background : bool
        Whether this is a background measurement (default is False)
    optimizer_config_path : Union[str, Path], optional
        Path to optimizer configuration file (default is None)
    optimizer_overrides : Dict[str, Any]
        Override parameters for optimizer (default is empty dict)
        Not currently in use
    evaluator_kwargs : Dict[str, Any]
        Keyword arguments for evaluator (default is empty dict)

    Examples
    --------
    >>> config = ScanConfig(
    ...     scan_mode=ScanMode.STANDARD,
    ...     device_var="U_ESPJetXYZ:Position.Axis 1",
    ...     start=0.1,
    ...     end=1.0,
    ...     step=0.1
    ... )
    >>> print(f"Scanning {config.device_var} from {config.start} to {config.end}")
    Scanning laser_power from 0.1 to 1.0
    """

    scan_mode: ScanMode = ScanMode.NOSCAN
    device_var: Optional[str] = None
    start: Union[int, float] = 0
    end: Union[int, float] = 1
    step: Union[int, float] = 1
    wait_time: float = 1.0
    additional_description: Optional[str] = None
    background: bool = False
    optimizer_config_path: Optional[Union[str, Path]] = None
    optimizer_overrides: Optional[Dict[str, Any]] = field(default_factory=dict)
    evaluator_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
