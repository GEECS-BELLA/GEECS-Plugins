# GEECS Plugin Suite

This is the documentation for **GEECS-Plugins**, a collection of Python tools built for laser-plasma experiments at [Lawrence Berkeley National Laboratory's BELLA Center](https://bella.lbl.gov/). The facility uses **GEECS** (Generalized Equipment and Experiment Control System) for hardware control and data acquisition — these packages extend that ecosystem with a Python-native interface for scanning, image analysis, and automated post-processing.

---

## Projects

### [GEECS Scanner GUI](geecs_scanner/overview.md)
A PyQt5-based data acquisition interface that replaces or supplements Master Control for running scans. Supports composite scan variables, pre/post-scan action sequences, multi-scan batch execution, and parameter optimization via [Xopt](https://xopt.xopt.org/). Designed for experimentalists who want more control and automation than Master Control provides.

### [Image Analysis](image_analysis/overview.md)
YAML-configured image processing and analysis for camera data. Provides a modular pipeline (background subtraction, masking, filtering, beam profile fitting, etc.) that works across multiple file formats (`.png`, `.hdf5`, `.himg`, `.tsv`). Designed for offline post-analysis; also supports online analysis via LabVIEW Point Grey Camera devices.

### [Scan Analysis](scan_analysis/overview.md)
Orchestrates image analysis across complete experimental scans — handling shot binning, rendering summary figures, appending results to the s-file, and optionally uploading to a Google Doc e-log. Can run interactively on a finished scan or as a live watcher that automatically processes scans as they complete.

### [GEECS Python API](geecs_python_api/overview.md)
The low-level Python interface to the GEECS control system — device communication, data access, and hardware control. Most users interact with this indirectly through the Scanner GUI or Scan Analysis packages, but it is also available for direct scripting.

### [GEECS Data Utils](geecs_data_utils/overview.md)
Path resolution and data loading utilities for GEECS experiment data. Resolves scan folder locations, loads s-files, and provides the common data structures used across the other packages. Typically used as a dependency rather than directly.

---

## Where to Start

The best entry point for most packages is the **examples** — each one is a Jupyter notebook demonstrating real usage end-to-end.

| I want to... | Start here |
|---|---|
| Analyze images from a camera device | [Image Analysis — Basic Offline Analysis](image_analysis/examples/basic_offline_analysis.ipynb) |
| Run analysis across a scan (2D image data) | [Scan Analysis — Basic Usage](scan_analysis/examples/basic_usage.ipynb) |
| Set up config-driven automated scan analysis | [Scan Analysis — Config-Based Workflow](scan_analysis/examples/config_based_scan_analysis.ipynb) |
| Watch for and process scans automatically (live) | [Scan Analysis — Live Watch](scan_analysis/examples/live_watch.ipynb) |
| Upload analysis results to a Google Doc e-log | [Scan Analysis — GDoc Upload](scan_analysis/examples/gdoc_upload.ipynb) |
| Run a parameter optimization scan | [Scanner GUI — Optimization Example](geecs_scanner/examples/optimization/optimization_example.ipynb) |
| Load and navigate GEECS scan data | [Data Utils — Basic Usage](geecs_data_utils/examples/basic_usage.ipynb) |

---

*GEECS — Copyright (c) 2016, The Regents of the University of California, through Lawrence Berkeley National Laboratory*
