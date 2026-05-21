"""
Type definitions for GEECS scan configuration and modes.

This module defines the core data structures used throughout the GEECS
plugin suite for configuring and managing experimental scans.

Contains enumerations for scan modes and Pydantic models for scan
configuration parameters.
"""

from __future__ import annotations

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


class ScanConfig(BaseModel):
    """
    Pydantic model for GEECS scan parameters.

    Describes a scan completely enough to write ``ScanInfo{scan}.ini`` and
    to be parsed back by analysis tools.  All fields have defaults so that
    partial construction (e.g. noscan with only ``wait_time``) is valid.

    Attributes
    ----------
    scan_mode : ScanMode
        Type of scan to perform.
    device_var : str, optional
        ``"device_name:variable_name"`` for the scanned variable.
    start : int or float
        Starting value for a parameter sweep.
    end : int or float
        Ending value for a parameter sweep.
    step : int or float
        Step size for a parameter sweep.
    wait_time : float
        Acquisition time per step in seconds.
    additional_description : str, optional
        Free-text appended to ``ScanStartInfo`` in the scan_info file.
    background : bool
        Flags this scan as a background measurement.
    optimizer_config_path : str or Path, optional
        Path to the Xopt optimizer configuration YAML.
    optimizer_overrides : dict
        Per-run overrides applied on top of the optimizer config.
    evaluator_kwargs : dict
        Extra keyword arguments forwarded to the evaluator constructor.
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
    optimizer_overrides: Optional[Dict[str, Any]] = Field(default_factory=dict)
    evaluator_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


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
