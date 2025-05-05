from __future__ import annotations

import os
import logging

from pathlib import Path
from dateutil.parser import parse as dateparse
from typing import Optional, Union, NamedTuple

# module‐level logger
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

# support both strings and real Path objects
SysPath = Union[str, bytes, os.PathLike, Path]



class ScanTag(NamedTuple):
    year: int
    month: int
    day: int
    number: int
    experiment: Optional[str] = None

def month_to_int(month: Union[str, int]) -> int:
    """ :return: an integer representing the given month """
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

if __name__ == '__main__':
    pass