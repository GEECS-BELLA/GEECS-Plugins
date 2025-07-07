# Image Analysis - Installation & Setup

This guide covers the installation and setup process for the Image Analysis module, including both standalone usage and integration with GEECS devices.

## Installation Options

### Option 1: Development Installation, i.e. 'offline analysis'

For development and standalone analysis preferred method is to use poetry:

1. **Navigate to Project Directory**:
   ```bash
   cd path/to/GEECS-Plugins/ImageAnalysis
   ```

2. **Install with Poetry**:
   ```bash
   poetry install
   ```

3. **Alternative: Direct Installation**:
   ```bash
   pip install -e .
   ```

### Option 2: Device Computer Installation, i.e. "Camera Server", for online analysis

Note, Labview 2018 does not support virtual environments nor python above 3.6 (though, 3.7 seems to work ok)

Required python installation:
**Python 3.7.9 (32-bit)**: For device computer integration
   - Download from: [Python 3.7.9](https://www.python.org/downloads/release/python-379/)
   - **Important**: Use the 32-bit version for LabVIEW compatibility
   - Check "Add python 3.7.9 to PATH" during installation

For integration with GEECS Point Grey Camera devices:
Navigate to 'server' version of the geecs software and install pacakge with pip:
Example for HTU/Undulator experiment
   ```bash
   cd Z:
   cd '.\software\control-all-loasis\HTU\Active Version\GEECS-Plugins\'
   ```
**Install Package**:
   ```bash
   py -3.7-32 -m pip install ./ImageAnalysis
   ```

## Basic Usage

### Standalone Image Analysis

```python
from image_analysis.utils import read_imaq_image
from image_analysis.offline_analyzers.Undulator.EBeamProfile import EBeamProfileAnalyzer

# Load image (handles LabVIEW PNG format)
image = read_imaq_image('beam_image.png')

# Create analyzer
analyzer = EBeamProfileAnalyzer(camera_name="UC_ALineEBeam3")

# Analyze image
results = analyzer.analyze_image(image)
print(f"results: {results}")

```

## Integration with LabVIEW/GEECS
see other doc

---

*For additional support, consult the tutorial slides: [GEECS-Python Analysis Tutorial (Coding)](https://docs.google.com/presentation/d/1RU251CXiWsM73NsBJtd_jdQ7tNdZbKSX9x-xBv1vtJw/edit?usp=sharing)*
