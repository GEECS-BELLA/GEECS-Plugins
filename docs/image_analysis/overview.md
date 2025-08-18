# Image Analysis

The Image Analysis module is a flexible toolkit for developing custom analysis routines for various types of array type data. Custom ImageAnalyzers should be derived from base class (base.ImageAnalyzer).

Contacts: Sam, Chris

## Key Features

- **Online Analysis**: Direct integration with GEECS Point Grey Camera devices using the derived LabviewImageAnalyzer class. Note: this functionality has been tested and is functional, but not 100% reliable. Often, a GEECS Point Grey Camera device running python analysis for long periods (>1 day) seems to trigger python related errors.

- **Post-Processing**: Detailed analysis after experiment completion
- **Customizable Image Rendering**:

- **LabVIEW PNG Support**: Proper handling of LabVIEW's unique PNG bit-shifting
- **Multiple Camera Types**: Support for various experimental camera systems and file types including: .tsv, .hdf5, .png, .himg, .has

- **Custom Analyzers**: Easy development of experiment-specific analysis routines
- **Algorithm Library**: Collection of common image processing algorithms
- **Third-Party Integration**: Support for external SDKs and libraries
- **Modular Design**: Mix and match analysis components as needed

## Architecture

The module is organized into several key components:

### Core Components
- **Base Classes** (`base.py`): Foundation classes for all analyzers
- **Analyzers**: Specific analysis implementations for different experiments
- **Algorithms**: Low-level image processing routines
- **Tools**: Utility functions and helper classes

### Utilities
- **Image Utils** (`utils.py`): Common image processing utilities

## Important Notes

!!! warning "LabVIEW PNG Images"
    When working with PNG images saved by LabVIEW, always use the `read_imaq_png_image()` function instead of standard image loading libraries. LabVIEW saves PNG images with a unique bit-shift that can cause unexpected scaling if not handled properly.

---

*Designed to work with any GEECS experiment, with specialized support for BELLA facility requirements.*
