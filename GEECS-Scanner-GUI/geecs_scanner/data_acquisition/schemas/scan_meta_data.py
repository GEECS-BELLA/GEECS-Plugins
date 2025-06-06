"""
scan_info_model.py

Defines a Pydantic model for the `[Scan Info]` section of an INI configuration file.
Each field corresponds to one entry in that section, with appropriate types and
aliases matching the INI keys.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ScanInfo(BaseModel):
    """
    Configuration schema for the `[Scan Info]` section of an INI file.

    Attributes:
        scan_no: The integer scan number.
        scan_start_info: Optional text for scan start info (often empty).
        scan_parameter: The scan parameter string (e.g., device or axis name).
        start: Starting floating-point value of the scan variable.
        end: Ending floating-point value of the scan variable.
        step_size: Floating-point increment for each scan step.
        shots_per_step: Integer number of shots collected at each step.
        scan_end_info: Optional text for scan end info (often empty).
        background: Boolean flag indicating whether to include a background measurement.
    """

    scan_no: int = Field(
        ...,
        alias="Scan No",
        description="The integer scan number."
    )

    scan_start_info: Optional[str] = Field(
        "",
        alias="ScanStartInfo",
        description="Optional text for scan start info."
    )

    scan_parameter: str = Field(
        ...,
        alias="Scan Parameter",
        description="The scan parameter string (e.g., device name or axis)."
    )

    start: float = Field(
        ...,
        alias="Start",
        description="Starting floating-point value of the scan variable."
    )

    end: float = Field(
        ...,
        alias="End",
        description="Ending floating-point value of the scan variable."
    )

    step_size: float = Field(
        ...,
        alias="Step size",
        description="Floating-point increment for each scan step."
    )

    shots_per_step: int = Field(
        ...,
        alias="Shots per step",
        description="Integer number of shots collected at each step."
    )

    scan_end_info: Optional[str] = Field(
        "",
        alias="ScanEndInfo",
        description="Optional text for scan end info."
    )

    background: Optional[bool] = Field(
        False,
        alias="Background",
        description="Flag indicating whether to include a background measurement."
    )

    class Config:
        """
        Pydantic configuration: allow using field names or aliases interchangeably.
        """
        allow_population_by_field_name = True
