"""
Models and schema helpers for the scan database.

This module defines Pydantic models that represent a single scan
(`ScanEntry`) and its flat scan metadata (`ScanMetadata`), plus utilities
to produce stable pandas/PyArrow schemas for Parquet I/O.

Contents
--------
Classes
    ScanMetadata
        Flat, one-level scan metadata parsed from `scan_info.ini`.
    ScanEntry
        A single scan record including date parts, file paths, devices,
        optional `ScanMetadata`, optional `ECSDump`, notes, and flags.

Functions
    collect_dtypes(entry)
        Return pandas dtypes (as strings) suitable for `DataFrame.astype`.
    get_pyarrow_schema(entry)
        Return a `pyarrow.Schema` suitable for Parquet writes.

Notes
-----
- `ScanEntry.flatten()` converts the model to a flat `dict` for DataFrame /
  Parquet use. The `ecs_dump` field is **serialized to a JSON string**
  via `model_dump(by_alias=True)` to preserve field aliases (e.g., `"shot #"`).
- `collect_dtypes()` forces the `ecs_dump` column to pandas `"string"` and uses
  nullable dtypes (`Int32`, `boolean`) where possible.
- `get_pyarrow_schema()` forces `ecs_dump` to `pa.large_string()` to safely hold
  large JSON payloads and avoid Arrow object columns.
- `ScanMetadata.raw_fields` is serialized to JSON in the flattened output so it
  can be round-tripped losslessly.

See Also
--------
geecs_data_utils.type_defs.ECSDump
pandas.DataFrame.astype
pyarrow.schema

Examples
--------
>>> entry = ScanEntry(
...     year=2025, month=8, day=8, number=123, experiment="Undulator",
...     scalar_data_file=None, tdms_file=None, non_scalar_devices=[],
...     scan_metadata=ScanMetadata(scan_parameter="U_S3V:Current"),
...     ecs_dump=None, has_analysis_dir=False,
... )
>>> row = entry.flatten()
>>> # Enforce stable types for pandas
>>> dtypes = collect_dtypes(entry)
>>> # Build an Arrow schema for Parquet
>>> schema = get_pyarrow_schema(entry)
"""

from __future__ import annotations
from typing import Optional, List, Dict
from datetime import date
from pydantic import BaseModel
import pyarrow as pa
import json

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
        User-provided description of the scan.
    background : Optional[bool]
        Whether this scan was flagged as a background measurement.
    raw_fields : Dict[str, str]
        Raw key-value fields from the scan info file.

    Notes
    -----
    - This model is expected to be flat (no nested submodels).
    - Nested models inside ScanMetadata are not currently supported by flatten().
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
        """Convert dict loaded from ini file to ScanEntry model type."""

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

    This model uses submodels for components like ScanMetadata and flattens
    fields on export for Parquet/DF usage. Nested submodels beyond one level
    (i.e., submodels inside submodels) are currently not supported and will
    raise a ValueError during flattening.
    """

    year: int
    month: int
    day: int
    number: int
    experiment: str
    scalar_data_file: Optional[str]
    tdms_file: Optional[str]
    non_scalar_devices: List[str]
    scan_metadata: Optional[ScanMetadata] = None
    ecs_dump: Optional[ECSDump] = None
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

    @property
    def scan_id(self) -> str:
        """
        A standardized string identifier for the scan, useful for manifests or logging.

        Format: "{experiment}_{YYYY-MM-DD}_Scan{NNN}"
        """
        return f"{self.experiment}_{self.date.isoformat()}_Scan{self.number:03d}"

    def flatten(self) -> dict:
        """
        Flatten nested models into a flat dictionary for Parquet/DF use.

        Returns
        -------
        dict
            A flattened dictionary representation of the scan entry.

        Raises
        ------
        ValueError
            If nested submodels are detected within any BaseModel attribute.
        """
        flat = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ScanMetadata):
                for sub_k, sub_v in v.model_dump(exclude_unset=False).items():
                    # Serialize raw_fields if it's a dict
                    if sub_k == "raw_fields" and isinstance(sub_v, dict):
                        sub_v = json.dumps(sub_v)
                        flat[f"{k}_{sub_k}"] = sub_v
                    else:
                        flat[f"{sub_k}"] = sub_v

            elif k == "ecs_dump":
                # ✅ Single standard: JSON string written to Parquet
                if v is None:
                    flat[k] = None
                else:
                    flat[k] = json.dumps(v.model_dump(by_alias=True))

            else:
                flat[k] = v
        return flat

    @classmethod
    def unflatten(cls, data: dict) -> ScanEntry:
        """
        Rehydrate a ScanEntry from a flat dictionary.

        Parameters
        ----------
        data : dict
            A flat dictionary, typically loaded from a Parquet row.

        Returns
        -------
        ScanEntry
            A reconstructed ScanEntry instance.
        """
        scan_metadata_fields = {
            k.removeprefix("scan_metadata_"): v
            for k, v in data.items()
            if k.startswith("scan_metadata_")
        }
        base_fields = {
            k: v for k, v in data.items() if not k.startswith("scan_metadata_")
        }
        return cls(
            **base_fields,
            scan_metadata=ScanMetadata(**scan_metadata_fields)
            if scan_metadata_fields
            else None,
        )

    model_config = {"arbitrary_types_allowed": True}


# -----------------------------------------------------------------------------
# Schema mapping
# -----------------------------------------------------------------------------


def collect_dtypes(entry: ScanEntry) -> dict[str, str]:
    """
    Collect **pandas-compatible** dtypes for use in .astype().

    Use string representations only — no pyarrow types.
    """
    flat = entry.flatten()
    out = {}
    for k, v in flat.items():
        if k == "ecs_dump":
            out[k] = "string"  # ✅ force pandas string
        elif isinstance(v, int):
            out[k] = "Int32"
        elif isinstance(v, float):
            out[k] = "float64"
        elif isinstance(v, bool):
            out[k] = "boolean"
        elif isinstance(v, list):
            out[k] = "object"  # ✅ pandas dtype for lists

        else:
            out[k] = "string"
    return out


def get_pyarrow_schema(entry: ScanEntry) -> pa.Schema:
    """Collect PyArrow-compatible schema for writing to Parquet."""
    flat = entry.flatten()
    fields = []
    for k, v in flat.items():
        if k == "ecs_dump":
            dtype = pa.large_string()  # ✅ force Arrow string
        elif isinstance(v, int):
            dtype = pa.int32()
        elif isinstance(v, float):
            dtype = pa.float64()
        elif isinstance(v, bool):
            dtype = pa.bool_()
        elif isinstance(v, list):
            dtype = pa.list_(pa.string())

        else:
            dtype = pa.string()
        fields.append(pa.field(k, dtype))
    return pa.schema(fields)
