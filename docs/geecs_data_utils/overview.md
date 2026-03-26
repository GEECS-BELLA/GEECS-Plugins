# GEECS Data Utils

GEECS Data Utils handles the basics of finding and loading GEECS experiment data: resolving scan folder paths from a date and scan number, loading s-files (the per-shot metadata TSV that GEECS records for every scan), and providing the common data structures that other packages in this suite build on.

This package is **typically used as a dependency** — Scan Analysis and the GEECS Python API pull it in automatically. You might use it directly when writing scripts to load and explore scan data outside of those frameworks.

---

## Core Functionality

**Path Resolution** — Given an experiment name, date, and scan number, resolve the correct data folder on disk. Handles the GEECS directory convention so you don't have to hard-code paths.

```python
from geecs_data_utils import ScanTag
from geecs_data_utils.scan_paths import get_scan_folder

tag = ScanTag(year=2024, month=6, day=15, number=42)
folder = get_scan_folder(experiment="Undulator", scan_tag=tag)
```

**S-File Loading** — Load the s-file for a scan as a pandas DataFrame, with shot numbers, timestamps, and all recorded device variables.

**Type Definitions** — Common types (`ScanTag`, path aliases, etc.) used consistently across the plugin suite.

---

## Examples

| Notebook | What it covers |
|---|---|
| [Basic Usage](examples/basic_usage.ipynb) | Loading scan data, resolving paths, reading s-files |
| [Scan Database Utils](examples/scans_database_utils.ipynb) | Querying and navigating the scan database |

---

## API Reference

- [Core Modules](api/core_modules.md)
