# Image Analysis Overview

The Image Analysis package provides a toolkit designed to make image processing accessible and configurable, allowing researchers to easily analyze camera data with minimal setup. It is also intended to be easily extensible enable integration a new processing pipelines or new analysis algorithims. The framework is generic enough that it can be used for any type of array 2D data (e.g. HasoLift .himg data).

This package contains a fair bit of general framework that makes it flexible to use for various analysis workflows for various experiments. The inention is to continue to grow the available list of analyzers (both general and specific) to cover a wide range of use cases at BELLA Center.

ImageAnalyzers are leveraged by Array2DAnalysis (which is part of ScanAnalysis module), to automate full image analysis workflows across 'scans' including binnging, rendering, appending to 'sFile' etc. Further extensions enable direct and automatic logging of the rendered results to "LogMaker" e-logs.

The best way to get started is to look at the [examples folder](examples/)

## Key Features

- **Configuration-Driven Analysis**: YAML-based camera configurations for reproducible analysis workflows
- **Modular Processing Pipeline**: Composable image processing steps including background subtraction, masking, filtering, and transforms
- **Extensible Architecture**: Easy development of custom analyzers, new processing methods, etc.
- **Multiple File Format Support**: Handle various experimental data formats including .tsv, .hdf5, .png, .himg, .has
- **Integration with ScanAnalysis/Array2DAnalysis**: existing framework to iterate analyzers across scans, with image binnig, rendering etc.
- **LabVIEW Integration**: Secondary support for online analysis with GEECS Point Grey Camera devices

## Package Architecture

The architecture is currently under a bit of development. A few important concepts:

- **analyzers vs offline_analyzers:** The 'analyzers' module is reserved for Labview compatible analyzer while 'offline_analyzers' are used for analyzing already recorded data. In principle, either could be used for both purposes, but, the very tight restrictions for LabView compatiblity are overly restrictive for general analysis purposes. See the note below on labview analyzers. It is expected the majority of devleopment will be with offline_analyzers, which are generally formatted to analyze recorded data.

- **processing vs algorithms vs utils/tools:** These three different modules serve similar types of purposes but are somewhat distinct. The general breakdown is meant to be the following. Functionality that 'takes in' an image and 'returns' an image is classified as processing, e.g. background substraction, filtering etc. The alogrithms module is reserved for functionality that takes in a processed image, performs some analysis and returns scalar type or 1D array type data. Utils is meant for fucntionality that is more like a data type translation, e.g. path-like object to image.
These aren't strict rules, but general guidelines as to where various functionalities should reside.

*Core Components*

- **Configuration System** (`config_loader.py`): Load and validate YAML camera configurations
- **Processing Pipeline** (`processing/`): Modular image processing operations
- **Offline Analyzers** (`offline_analyzers/`): High-level analysis classes for common workflows
- **Base Classes** (`base.py`): Foundation classes for building custom analyzers

*Processing Modules*

- **Background Management** (`processing/background.py`): Static, dynamic, and hybrid background subtraction
- **Masking Operations** (`processing/masking.py`): Crosshair masking, ROI cropping, circular masks
- **Filtering** (`processing/filtering.py`): Noise reduction and image enhancement
- **Transforms** (`processing/transforms.py`): Geometric transformations and corrections
- **Thresholding** (`processing/thresholding.py`): Intensity-based image segmentation
- **and more** we continually add new features

*Ready-to-Use Analyzers*

- **StandardAnalyzer**: Base class with configurable processing pipeline
- **BeamAnalyzer**: Specialized for electron beam profile analysis
- **Custom Analyzers**: Framework for experiment-specific analysis routines

## Examples

The [examples folder](examples/) contains Jupyter notebooks demonstrating:


## A note on integration with Point Grey Camera device type
The earliest implementation of this package was to enable 'online' analysis in the backend of the Point Grey Camera driver. This approach is functional, but limited. LabView 2020 only explicitly supports python 3.6 and does not support virtual environments. It has also been noted that the implementation can be unreliable. We often observe python related errors when a device with python analysis is left on for long periods of time.

See the [examples](examples/) for detailed configuration workflows.



## Important Notes

!!! warning "LabVIEW PNG Images"
    When working with PNG images saved by LabVIEW, always use the `read_imaq_png_image()` function instead of standard image loading libraries. LabVIEW saves PNG images with a unique bit-shift that can cause unexpected scaling if not handled properly.

!!! tip "Configuration Management"
    Store camera configurations in version control alongside your analysis scripts to ensure reproducible results and easy collaboration.

---

*Designed for reliable offline analysis of GEECS experimental data, with specialized support for BELLA facility requirements.*
