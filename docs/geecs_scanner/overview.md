# GEECS Scanner GUI

The GEECS Scanner GUI is a Python-based data acquisition module that provides a flexible alternative to Master Control for data collection. Built with PyQt5, it offers a modular interface for experiment control with enhanced customization options.

## Overview

While Master Control scans save everything but can crash or slow down when any device encounters an error, the GEECS Scanner operates on an "opt-in" framework. This approach provides several advantages:

- **Python Flexibility**: Easy to extend with additional features and automation

### Key Features

- **Scan Management**: Many of the standard scan modes (e.g. 'no scan', 1D parameter scans) with additional support for multi scan execution and optimization 'scans'.

- **Composite Variables**: Combine multiple device parameters into single scan variables using arbitrary mathematical relations.

- **Automated Actions**: Pre and post-scan automation sequences using user defined action sequences allow complex configuration changes before/after scans.

- **Timing and Synchronization**: Data synchronization handled through hardware timestamps rather than shot number.

- **Configuration GUIs**: GUIs available to create the necessary config files (e.g. 'save devices', 'multi scan', etc.)

## Architecture

The GUI is organized into several core components:

- **Main Window** (`GEECSScanner.py`): Primary interface for scan configuration
- **Run Control** (`RunControl.py`): Interface between GUI and backend scan management
- **Element Editor**: Device and action configuration interface
- **Multi-Scanner**: Batch scan execution interface
- **Backend Integration**: Uses `geecs_python_api` for device communication

## Use Cases

The GEECS Scanner GUI is ideal for:

- Reliable, automated scanning with minimal intervention
- Coordinated control of multiple experimental parameters
- Custom scan patterns and automated sequences
- Python-based extensibility for custom features
- Paramter optimization using [Xopt](https://xopt.xopt.org/)

## Getting Started

1. **Installation**: Follow the [Installation & Setup](installation.md) guide
2. **Configuration**: Set up experiment configuration and timing
3. **Tutorial**: Work through the [Tutorial](tutorial.md) for hands-on learning
4. **Advanced Usage**: Explore multi-scanning and custom actions

## Related Documentation

- [Installation & Setup](installation.md) - Detailed setup instructions
- [Tutorial](tutorial.md) - Step-by-step usage guide

---

*Originally developed for HTU but designed to be generally usable by any experiment using the GEECS/Master Control environment.*
