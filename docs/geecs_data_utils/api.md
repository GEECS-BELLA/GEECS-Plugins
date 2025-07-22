# GEECS Data Utils - API Reference

This page contains the complete API reference for the `geecs_data_utils` module. The module provides utilities for working with GEECS experimental data, including path management, scan data handling, and configuration utilities.

## Module Overview

The `geecs_data_utils` package contains the following modules:

- **utils**: Core utility classes and functions including `ScanTag` and error handling
- **type_defs**: Type definitions and enums for scan configurations
- **geecs_paths_config**: Path configuration management for GEECS data
- **scan_paths**: Path resolution and management for scan data
- **scan_data**: High-level interface for loading and working with scan data

---

## geecs_data_utils.utils

Core utility classes and functions used throughout the package.

::: geecs_data_utils.utils.ScanTag
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 3
      members: false

::: geecs_data_utils.utils.ConfigurationError
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 3
      members: false

::: geecs_data_utils.utils.month_to_int
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 3

---

## geecs_data_utils.type_defs

Type definitions and configuration classes for scan operations.

::: geecs_data_utils.type_defs.ScanMode
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 3
      members: false

::: geecs_data_utils.type_defs.ScanConfig
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 3
      members: false

---

## geecs_data_utils.geecs_paths_config

Configuration management for GEECS data paths and server addresses.

::: geecs_data_utils.geecs_paths_config.GeecsPathsConfig
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3
      filters: ["!^_"]
      show_if_no_docstring: false
      group_by_category: true
      show_signature_annotations: true

---

## geecs_data_utils.scan_paths

Path resolution and management for scan data files and directories.

::: geecs_data_utils.scan_paths.ScanPaths
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3
      filters: ["!^_"]
      show_if_no_docstring: false
      group_by_category: true
      show_signature_annotations: true

---

## geecs_data_utils.scan_data

High-level interface for loading and working with GEECS scan data.

::: geecs_data_utils.scan_data.read_geecs_tdms
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 3

::: geecs_data_utils.scan_data.geecs_tdms_dict_to_panda
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      heading_level: 3

::: geecs_data_utils.scan_data.ScanData
    options:
      show_source: true
      show_root_heading: false
      show_root_toc_entry: true
      merge_init_into_class: true
      heading_level: 3
      filters: ["!^_"]
      show_if_no_docstring: false
      group_by_category: true
      show_signature_annotations: true

---

## Usage Examples

### Basic Usage

```python
from geecs_data_utils import ScanData, ScanPaths

# Load scan data for a specific scan
scan_data = ScanData(year=2024, month=1, day=15, scan_number=42)

# Access scan paths
scan_paths = ScanPaths(year=2024, month=1, day=15, scan_number=42)
folder_path = scan_paths.get_scan_folder_path()
```

### Working with Scan Tags

```python
from geecs_data_utils import ScanTag

# Create a scan tag
tag = ScanTag(year=2024, month=1, day=15, scan_number=42)
print(f"Scan: {tag.year}-{tag.month:02d}-{tag.day:02d}_{tag.scan_number:03d}")
```

### Configuration Management

```python
from geecs_data_utils import GeecsPathsConfig

# Initialize paths configuration
config = GeecsPathsConfig(experiment_name="HTU")
