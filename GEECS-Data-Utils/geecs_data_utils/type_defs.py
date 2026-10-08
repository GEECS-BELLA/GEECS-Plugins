"""
Type definitions for GEECS scan configuration and modes.

This module defines the core data structures used throughout the GEECS
plugin suite for configuring and managing experimental scans.

Contains enumerations for scan modes and Pydantic models for scan
configuration parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List
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
    Enumeration of scan modes, as recorded in ``ScanInfoScanNNN.ini``.

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
