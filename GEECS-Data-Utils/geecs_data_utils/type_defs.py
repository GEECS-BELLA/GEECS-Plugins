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
from typing import Optional, Union, Dict, Any, List
from enum import Enum
from configparser import ConfigParser
from datetime import date

from pydantic import BaseModel, Field


class ScanTag(BaseModel):
    """
    pydantic model representing a GEECS scan identifier.

    This class provides a structured way to identify and reference
    specific scans within the GEECS acquired data using date and scan number.

    Attributes
    ----------
    year : int
        Year when the scan was performed
    month : int
        Month when the scan was performed (1-12)
    day : int
        Day of the month when the scan was performed
    number : int
        Sequential scan number for that day
    experiment : str
        Name of the experiment (default is None)

    Examples
    --------
    >>> scan = ScanTag(year=2024, month=1, day=15, number=42, experiment="Undulator")
    >>> print(f"Scan {scan.number} from {scan.year}-{scan.month:02d}-{scan.day:02d}")
    Scan 42 from 2024-01-15
    """

    year: int
    month: int
    day: int
    number: int
    experiment: Optional[str] = None

    @property
    def date(self) -> date:
        """
        Return the date associated with the scan.

        Returns
        -------
        date
            The date extracted from the year, month, and day fields.
        """
        return date(self.year, self.month, self.day)


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


class DeviceDump(BaseModel):
    """
    Represents a single device record in an ECS Live Dump file.

    Attributes
    ----------
    name : str
        Device name (from "Device Name" field).
    shot_number : Optional[int]
        Shot number (from "shot #" field).
    parameters : Dict[str, str]
        All other key-value pairs from the ECS dump for this device.
    """

    name: str
    shot_number: Optional[int] = Field(None, alias="shot #")
    parameters: Dict[str, str] = {}

    model_config = {
        "extra": "allow",
        "populate_by_name": True,
    }


class ECSDump(BaseModel):
    """
    Represents the full ECS Live Dump file for a scan.

    Attributes
    ----------
    experiment_name : Optional[str]
        The name of the experiment from the [Experiment] section.
    devices : List[DeviceDump]
        All parsed devices from the dump file.
    """

    experiment_name: Optional[str]
    devices: List[DeviceDump]


def parse_ecs_dump(path: Path) -> Optional[ECSDump]:
    """
    Parse ECS Live Dump file and return an ECSDump object.

    Parameters
    ----------
    path : Path
        Path to ECS dump .txt file.

    Returns
    -------
    Optional[ECSDump]
        Parsed experiment name and list of DeviceDump entries, or None if file doesn't exist.
    """
    if not path.exists():
        return None

    parser = ConfigParser()
    parser.optionxform = str  # preserve case
    parser.read(path)

    experiment = parser.get("Experiment", "Expt Name", fallback=None)
    if experiment:
        experiment = experiment.strip('"')

    devices = []
    for section in parser.sections():
        if not section.startswith("Device "):
            continue

        raw_items = dict(parser.items(section))
        name = raw_items.pop("Device Name", "").strip('"')
        shot = raw_items.pop("shot #", None)
        device = DeviceDump(
            name=name,
            shot_number=int(shot) if shot and shot.isdigit() else None,
            parameters={k: v.strip('"') for k, v in raw_items.items()},
        )
        devices.append(device)

    return ECSDump(experiment_name=experiment, devices=devices)
