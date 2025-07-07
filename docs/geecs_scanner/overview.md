# GEECS Scanner GUI

The GEECS Scanner GUI is a Python-based data acquisition module that provides a flexible alternative to Master Control for data collection. Built with PyQt5, it offers a modular interface for experiment control with enhanced customization options.

## Overview

While Master Control scans save everything but can crash or slow down when any device encounters an error, the GEECS Scanner operates on an "opt-in" framework. This approach provides several advantages:

- **Python Flexibility**: Easy to extend with additional features and automation

## Key Features

### Automated Actions
- Pre and post-scan automation sequences
- Configurable device state management
- Custom action library integration

### Composite Variables
- Define scan variables relative to current values
- Combine multiple device parameters into single scan variables
- Support for complex multi-device coordination

### Scan Management
- **NoScan (Statistics)**: Hold settings constant for specified number of shots
- **1D Scans**: Vary parameters from start to end with configurable step sizes
- **Multi-Scan Sequences**: Chain multiple scan presets together
- **Scan Presets**: Save and load complete scan configurations

### Timing and Synchronization
- Configurable timing setup for scan/standby/off modes
- Shot controller integration
- Flexible device synchronization

## Architecture

The GUI is organized into several core components:

- **Main Window** (`GEECSScanner.py`): Primary interface for scan configuration
- **Run Control** (`RunControl.py`): Interface between GUI and backend scan management
- **Element Editor**: Device and action configuration interface
- **Multi-Scanner**: Batch scan execution interface
- **Backend Integration**: Uses `geecs_python_api` for device communication

## Use Cases

The GEECS Scanner GUI is ideal for:

- **Routine Data Collection**: Reliable, automated scanning with minimal intervention
- **Complex Multi-Device Experiments**: Coordinated control of multiple experimental parameters
- **Flexible Scan Sequences**: Custom scan patterns and automated sequences
- **Development and Testing**: Python-based extensibility for custom features

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
