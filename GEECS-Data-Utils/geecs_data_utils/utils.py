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
import re

from pathlib import Path
from dateutil.parser import parse as dateparse
from typing import Union

# module‐level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# support both strings and real Path objects
SysPath = Union[str, bytes, os.PathLike, Path]


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


def timestamp_from_string(string: str) -> float:
    """Extract epoch-seconds timestamp from a GEECS filename string.

    Parameters
    ----------
    string : str
        Filename or string containing a timestamp in the form ``_<digits>.<3-digits>.<ext>``.

    Returns
    -------
    float
        Epoch seconds extracted from the string.

    Raises
    ------
    ValueError
        If the pattern is not found in *string*.
    """
    pattern = r"_(\d+\.\d{3})\.[^.]+$"
    match = re.search(pattern, string)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract timestamp from '{string}'")


def timestamp_from_filename(file: Path) -> float:
    """Extract epoch-seconds timestamp from a GEECS data file's name.

    Parameters
    ----------
    file : Path
        Path whose filename encodes a timestamp (see :func:`timestamp_from_string`).

    Returns
    -------
    float
        Epoch seconds.
    """
    return timestamp_from_string(file.name)


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration file."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


if __name__ == "__main__":
    pass
