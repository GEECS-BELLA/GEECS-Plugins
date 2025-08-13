"""
Scan scans_database entry schema.

This module defines structured models for representing scan entries and
associated metadata in the GEECS data ecosystem. These models are used
to serialize, organize, and manage experimental scan data, including
scalar logs, TDMS files, non-scalar device data, scan metadata parsed
from INI files, and ECS dumps.

Classes
-------
ScanMetadata
    Represents structured scan metadata extracted from INI-style config files.
ScanEntry
    Represents a full scan record with associated files and metadata.

Notes
-----
These models are primarily intended for use in building and querying a
structured scan scans_database using the `ScanDatabase` class. The models use
Pydantic for data validation and serialization.
"""

from __future__ import annotations
from typing import Optional, List, Dict
from datetime import date
from pydantic import BaseModel
import pyarrow as pa

from geecs_data_utils.utils import ScanTag
from geecs_data_utils.type_defs import ECSDump


# -----------------------------------------------------------------------------
# ScanMetadata
# -----------------------------------------------------------------------------


class ScanMetadata(BaseModel):
    """
    Encapsulates scan-level metadata parsed from the scan info file (INI).

    Attributes
    ----------
    scan_parameter : Optional[str]
        Name of the scanned variable (e.g., "U_S3V:Current").
    start : Optional[float]
        Start value of the scan range.
    end : Optional[float]
        End value of the scan range.
    step_size : Optional[float]
        Step size used for the scan.
    shots_per_step : Optional[float]
        Number of shots taken at each scan point.
    scan_mode : Optional[str]
        Type of scan (e.g., "standard", "background").
    scan_description : Optional[str]
        user input description of the scan.
    background : Optional[bool]
        Whether this scan was flagged as a background measurement.
    raw_fields : Dict[str, str]
        Raw key-value fields from the scan info file.
    """

    scan_parameter: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    step_size: Optional[float] = None
    shots_per_step: Optional[float] = None
    scan_mode: Optional[str] = None
    scan_description: Optional[str] = None
    background: Optional[bool] = None
    raw_fields: Dict[str, str] = {}

    @classmethod
    def from_ini_dict(cls, ini_data: Dict[str, str]) -> ScanMetadata:
        """
        Construct a ScanMetadata object from an INI-style dictionary.

        This method parses known scan metadata fields from a dictionary
        produced by reading a scan's INI configuration file. Numeric and
        boolean fields are converted to the appropriate Python types where
        possible. Any unrecognized fields are preserved in `raw_fields`.

        Parameters
        ----------
        ini_data : dict of str to str
            Dictionary parsed from a scan info (.ini) file, where all keys
            and values are strings.

        Returns
        -------
        ScanMetadata
            A validated ScanMetadata object with typed and structured
            metadata fields populated from the input dictionary.

        Notes
        -----
        - Missing keys are returned as ``None`` for optional fields.
        - Boolean values for ``background`` are parsed case-insensitively
          and accept ``"true"``, ``"1"``, ``"yes"``, and ``"y"`` as True.
        - All original key-value pairs from the INI file are stored in
          ``raw_fields`` for completeness.
        """

        def parse_float(key: str) -> Optional[float]:
            return float(ini_data[key]) if key in ini_data else None

        def parse_bool(key: str) -> Optional[bool]:
            if key not in ini_data:
                return None
            v = ini_data.get(key, "")
            return str(v).strip().lower() in {"true", "1", "yes", "y"}

        return cls(
            scan_parameter=ini_data.get("Scan Parameter"),
            start=parse_float("Start"),
            end=parse_float("End"),
            step_size=parse_float("Step size"),
            shots_per_step=parse_float("Shots per step"),
            scan_mode=ini_data.get("ScanMode"),
            scan_description=ini_data.get("ScanStartInfo"),
            background=parse_bool("Background"),
            raw_fields=ini_data,
        )


# -----------------------------------------------------------------------------
# ScanEntry
# -----------------------------------------------------------------------------


class ScanEntry(BaseModel):
    """
    Represents a single scan and its associated metadata and state.

    This model stores scan date components (`year`, `month`, `day`) as integers for
    Parquet/Hive compatibility, along with scan number, experiment name, file paths,
    device lists, and metadata.

    Attributes
    ----------
    year : int
        Four-digit year of the scan (e.g., 2025).
    month : int
        Month of the scan, in range [1, 12].
    day : int
        Day of the month of the scan, in range [1, 31].
    number : int
        Sequential scan number for that day.
    experiment : str
        Name of the experiment associated with this scan.
    scalar_data_file : Optional[str]
        Path to scalar data file (.txt), if available.
    tdms_file : Optional[str]
        Path to TDMS file, if present.
    non_scalar_devices : list of str
        Device names with saved image or scope data.
    scan_metadata : dict
        Parsed scan metadata (from ScanInfo file).
    ecs_dump : Optional[dict]
        Parsed ECS dump containing device states, if present.
    has_analysis_dir : bool
        Whether an associated analysis directory exists for the scan.
    notes : Optional[str]
        Optional notes or annotations about this scan.
    """

    year: int
    month: int
    day: int
    number: int
    experiment: str
    scalar_data_file: Optional[str]
    tdms_file: Optional[str]
    non_scalar_devices: List[str]
    scan_metadata: dict
    ecs_dump: Optional[dict]
    has_analysis_dir: bool
    notes: Optional[str] = None

    @property
    def date(self) -> date:
        """
        The scan date as a `datetime.date` object.

        Returns
        -------
        datetime.date
            The date corresponding to (`year`, `month`, `day`).
        """
        return date(self.year, self.month, self.day)

    model_config = {"arbitrary_types_allowed": True}

SCHEMA_DTYPE_MAP = {
    "year": "Int16",
    "month": "Int8",
    "day": "Int8",
    "number": "Int32",
    "experiment": "string[pyarrow]",
    "scalar_data_file": "string[pyarrow]",
    "tdms_file": "string[pyarrow]",
    "non_scalar_devices": pa.list_(pa.string()),
    "scan_metadata": "string[pyarrow]",
    "ecs_dump": "string[pyarrow]",
    "has_analysis_dir": "boolean",
    "notes": "string[pyarrow]",
}
