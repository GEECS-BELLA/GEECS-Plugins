# Welcome to the GEECS Plugin Suite Documentation

This site documents the GEECS-Plugins monorepo, a collection of Python tools for the Generalized Equipment and Experiment Control System (GEECS) used at Lawrence Berkeley National Laboratory's BELLA facility.

## Core Projects

### [GEECS Scanner GUI](geecs_scanner/overview.md)
A modular PyQt5-based interface for experiment control and scan management. Provides an alternative to Master Control with flexible data acquisition, automated scan sequences, and composite variable support.

**Key Features:**
- Opt-in device framework for reliable data collection
- Automated pre/post-scan actions
- Multi-scan sequencing with presets
- Composite scan variables

### [Image Analysis](image_analysis/overview.md)
Central repository for online and offline analysis of experimental images from BELLA experiments. Provides analyzers for beam parameter extraction and visual diagnostics.

**Key Features:**
- Real-time image analysis integration with GEECS devices
- Offline analysis tools for post-processing
- Support for LabVIEW image formats
- Extensible analyzer framework

### [Scan Analysis](scan_analysis/overview.md)
Tools for analyzing complete experimental scans, often incorporating image analysis for individual shots. Designed for cross-device analysis and automated scan processing.

**Key Features:**
- Multi-device data correlation
- Automated scan discovery and analysis
- Extensible analyzer framework
- Integration with Google Docs logging

## Additional Tools

The monorepo also includes several supporting projects:

- **GEECS-PythonAPI**: Core API for interfacing with GEECS control systems
- **GEECS-Data-Utils**: Utilities for working with GEECS data structures and file formats
- **LivePostProcessing**: Real-time data processing tools
- **Xopt-GEECS**: Integration with Xopt optimization library

## Getting Started

1. Check the [Installation Guide](installation.md) for setup instructions
2. Explore individual project documentation for detailed usage
3. Review API references for development information

---

*GEECS (Generalized Equipment and Experiment Control System) - Copyright (c) 2016, The Regents of the University of California, through Lawrence Berkeley National Laboratory*
