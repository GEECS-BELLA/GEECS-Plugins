# Welcome to the GEECS Plugin Suite Documentation

This site documents the GEECS-Plugins monorepo, a collection of Python tools for the Generalized Equipment and Experiment Control System (GEECS) used at Lawrence Berkeley National Laboratory's BELLA facility. There are several projects included which are briefly described below.

## Core Projects

### [GEECS Scanner GUI](geecs_scanner/overview.md)
A modular PyQt5-based interface for data acquisition. Provides an alternative to Master Control with flexible data acquisition, automated scan sequences, composite variable support, parameter optimization through Xopt, and more.

### [Image Analysis](image_analysis/overview.md)
Central repository for online and offline analysis of experimental images from BELLA experiments. Online analyzers can be used in conjunction with Point Grey Camera devices for live analysis, but they must be python 3.7 compatible. Offline analyzers have greater flexibility can be used in post analysis routines.

### [Scan Analysis](scan_analysis/overview.md)
Tools for analyzing complete experimental scans, often incorporating image analysis for individual shots. Designed for cross-device analysis and automated scan processing.

### [GEECS Python API](geecs_python_api/overview.md)
A general python API to interface with the GEECS control system software

### [GEECS Data Utils](geecs_data_utils/overview.md)
Some general utils to establish paths and access data recorded by GEECS control system

## Getting Started

1. Explore individual project documentation for detailed usage
2. Review API references for development information

---

*GEECS (Generalized Equipment and Experiment Control System) - Copyright (c) 2016, The Regents of the University of California, through Lawrence Berkeley National Laboratory*
