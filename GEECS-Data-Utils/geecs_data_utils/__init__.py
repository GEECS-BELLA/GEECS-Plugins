"""
GEECS Data Utilities Package.

This package provides utilities for working with GEECS experimental data,
including path management, scan data handling, and configuration utilities.

The package contains modules for:
- Scan data management and access
- Path configuration and resolution
- Data type definitions and utilities
- Common utility functions

Examples
--------
>>> from geecs_data_utils import ScanData, ScanPaths
>>> scan_paths = ScanData(year=2024, month=1, day=15, scan_number=42)
"""

from geecs_data_utils.scan_data import ScanData
from geecs_data_utils.scan_paths import ScanPaths
from geecs_data_utils.utils import ConfigurationError, SysPath
from geecs_data_utils.geecs_paths_config import GeecsPathsConfig
from geecs_data_utils.type_defs import ScanMode, ScanConfig, ScanTag

__all__ = [
    "ScanData",
    "ScanPaths",
    "ScanTag",
    "ConfigurationError",
    "SysPath",
    "GeecsPathsConfig",
    "ScanMode",
    "ScanConfig",
]
