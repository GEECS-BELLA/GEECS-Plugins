"""
Utility functions and classes for GEECS data handling.

This module provides common utilities for working with GEECS scan data,
including path handling, date parsing, and data structure definitions.

The module contains helper functions for converting between different
data formats and standardized data structures used throughout the
GEECS plugin suite.
"""

from __future__ import annotations

import os
import logging

from pathlib import Path
from dateutil.parser import parse as dateparse
from typing import Optional, Union
from datetime import date
from pydantic import BaseModel

# module‐level logger
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

# support both strings and real Path objects
SysPath = Union[str, bytes, os.PathLike, Path]


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


def month_to_int(month: Union[str, int]) -> int:
    """
    Convert month representation to integer.

    Converts various month representations (integer, string name, etc.)
    to a standardized integer format (1-12).

    Parameters
    ----------
    month : Union[str, int]
        Month representation to convert. Can be an integer (1-12)
        or a string that can be parsed as a date.

    Returns
    -------
    int
        Integer representation of the month (1-12)

    Raises
    ------
    ValueError
        If the month cannot be converted to a valid integer (1-12)

    Examples
    --------
    >>> month_to_int(3)
    3
    >>> month_to_int("March")
    3
    >>> month_to_int("2024-03-15")
    3
    """
    try:
        month_int = int(month)
        if 1 <= month_int <= 12:
            return month_int
    except ValueError:
        pass

    if isinstance(month, str):
        return dateparse(month).month
    else:
        raise ValueError(f"'{month}' is not a valid month")


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration file."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


if __name__ == "__main__":
    pass
